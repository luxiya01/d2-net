import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from lib.utils import upscale_positions


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self, use_cuda=True, num_channels=512):
        super(DenseFeatureExtractionModule, self).__init__()

        model = models.vgg16()
        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv4_3_idx = vgg16_layers.index('conv4_3')

        if num_channels == 512:
            model = list(model.features.children())[:conv4_3_idx + 1]
        else:
            model = list(model.features.children())[:conv4_3_idx]
            model.append(
                nn.Conv2d(512,
                          num_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        self.model = nn.Sequential(*model)

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_channels = num_channels

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        return output


class HardDetectionModule(nn.Module):
    def __init__(self, local_max_size=3):
        super(HardDetectionModule, self).__init__()
        self.local_max_size = local_max_size

    def forward(self, batch):
        batch = F.relu(batch)

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)

        detection = torch.min(is_depth_wise_max, is_local_max)
        grid_keypoints = torch.nonzero(detection)[:, 2:]
        keypoints = upscale_positions(grid_keypoints, scaling_steps=3)

        return grid_keypoints, keypoints


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)

        batch = F.relu(batch)

        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / (max_per_sample + 1e-7).view(b, 1, 1, 1))
        sum_exp = (
            self.soft_local_max_size**2 *
            F.avg_pool2d(F.pad(exp, [self.pad] * 4, mode='constant', value=1.),
                         self.soft_local_max_size,
                         stride=1))
        local_max_score = exp / (sum_exp + 1e-7)

        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / (depth_wise_max + 1e-7).unsqueeze(1)

        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]

        score = score / torch.sum(
            (score + 1e-7).view(b, -1), dim=1).view(b, 1, 1)

        return score


class D2Net(nn.Module):
    def __init__(self, model_file, use_cuda=True, num_channels=512):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            use_cuda=use_cuda, num_channels=num_channels)

        self.hard_detection = HardDetectionModule()
        self.soft_detection = SoftDetectionModule()

        if use_cuda:
            self.load_state_dict(torch.load(model_file)['model'], strict=True)
        else:
            self.load_state_dict(torch.load(model_file,
                                            map_location='cpu')['model'],
                                 strict=True)

    def forward(self, batch):
        dense_features = self.dense_feature_extraction(batch)

        grid_keypoints, keypoints = self.hard_detection(dense_features)
        scores = self.soft_detection(dense_features)

        return {
            'dense_features': dense_features,
            'grid_keypoints': grid_keypoints,
            'keypoints': keypoints,
            'scores': scores
        }
