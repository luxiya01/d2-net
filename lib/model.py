import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class DenseFeatureExtractionModule(nn.Module):
    def __init__(self,
                 finetune_feature_extraction=False,
                 use_cuda=True,
                 num_channels=512):
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

        model = list(model.features.children())[:conv4_3_idx + 1]
        if num_channels != 512:
            model.append(
                nn.Conv2d(512,
                          num_channels,
                          kernel_size=1,
                          stride=1,
                          padding=0))
        self.model = nn.Sequential(*model)

        self.num_channels = num_channels

        # Fix forward parameters
        for param in self.model.parameters():
            param.requires_grad = False
        if finetune_feature_extraction:
            # Unlock conv4_3
            if num_channels == 512:
                for param in list(self.model.parameters())[-2:]:
                    param.requires_grad = True
            # Unlock conv4_3 and bottleneck
            else:
                for param in list(self.model.parameters())[-3:]:
                    param.requires_grad = True

        if use_cuda:
            self.model = self.model.cuda()

    def forward(self, batch):
        output = self.model(batch)
        return output


class SoftDetectionModule(nn.Module):
    def __init__(self, soft_local_max_size=3, ignore_score_edges=False):
        super(SoftDetectionModule, self).__init__()

        self.soft_local_max_size = soft_local_max_size

        self.pad = self.soft_local_max_size // 2

        self.ignore_score_edges = ignore_score_edges

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

        if self.ignore_score_edges:
            score[:, :2, :] = 0
            score[:, -2:, :] = 0
            score[:, :, :2] = 0
            score[:, :, -2:] = 0

        return score


class D2Net(nn.Module):
    def __init__(self,
                 model_file=None,
                 use_cuda=True,
                 ignore_score_edges=False,
                 num_channels=512):
        super(D2Net, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModule(
            finetune_feature_extraction=True,
            use_cuda=use_cuda,
            num_channels=num_channels)

        self.detection = SoftDetectionModule(
            ignore_score_edges=ignore_score_edges)

        if model_file is not None:
            if use_cuda:
                self.load_state_dict(torch.load(model_file)['model'],
                                     strict=False)
            else:
                self.load_state_dict(torch.load(model_file,
                                                map_location='cpu')['model'],
                                     strict=False)

        self.ignore_score_edges = ignore_score_edges

    def forward(self, batch):
        b = batch['image1'].size(0)

        # input shape: (num_images, c_0, h_0, w_0)
        # output(dense_features) shape: (num_images, c_f, h_f, w_f)
        dense_features = self.dense_feature_extraction(
            torch.cat([batch['image1'], batch['image2']], dim=0))

        # scores shape: (num_images, h_f, w_f)
        scores = self.detection(dense_features)

        # dense_feature1 shape: (1, c_f, h_f, w_f)
        dense_features1 = dense_features[:b, :, :, :]
        # dense_feature2 shape: (1, c_f, h_f, w_f)
        dense_features2 = dense_features[b:, :, :, :]

        # score1 shape: (1, h_f, w_f)
        scores1 = scores[:b, :, :]
        # score1 shape: (1, h_f, w_f)
        scores2 = scores[b:, :, :]

        return {
            'dense_features1': dense_features1,
            'scores1': scores1,
            'dense_features2': dense_features2,
            'scores2': scores2
        }
