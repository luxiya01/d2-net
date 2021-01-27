import argparse

import os
import numpy as np

from matplotlib import pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from PIL import Image

from lib.model_detection import D2Net
from lib.utils import image_net_mean_std, show_tensor_image

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--data_dir',
    type=str,
    required=True,
    help=
    'path to a directory containing a subdirectory named "patches" with npz data files'
)
parser.add_argument('--feat_dir',
                    type=str,
                    required=True,
                    help='directory name for the resulting features')
parser.add_argument(
    '--store_separate_pngs',
    action='store_true',
    default=False,
    help=
    'store images as separate png files. If False, images will be logged to tensorboard'
)

parser.add_argument('--model_file',
                    type=str,
                    default='models/d2_tf.pth',
                    help='path to the full model')
parser.add_argument('--output_extension',
                    type=str,
                    default='.d2-net',
                    help='extension for the output')
parser.add_argument('--num_channels',
                    type=int,
                    default=512,
                    help='number of channels for the final output features')
parser.add_argument(
    '--img_type',
    type=str,
    default='unnormalised',
    help=
    'Image type used to extract features: (norm_intensity_artefact_removed, norm_intensity, unnorm_intensity)'
)

args = parser.parse_args()

print(args)

# Creating CNN model
model = D2Net(model_file=args.model_file,
              use_cuda=use_cuda,
              num_channels=args.num_channels)

# Tensorboard logging
log_dir = os.path.join(args.feat_dir, 'logs')
if args.store_separate_pngs:
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
else:
    writer = SummaryWriter(log_dir)

# Process the patches directory
image_dir = os.path.join(args.data_dir, os.path.join('images', args.img_type))
files = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]

mean, std = image_net_mean_std()
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)])

fig = plt.figure(figsize=(10, 5), constrained_layout=True)

for i, filename in tqdm(enumerate(files), total=len(files)):
    idx = (os.path.basename(os.path.normpath(filename))).split('.')[0]
    image = Image.open(filename).convert('RGB')
    image = np.array(image)

    input_image = data_transform(image).unsqueeze(0).to(device).float()

    with torch.no_grad():
        output = model(input_image)
        # dense_features.shape = (num_channels, h, w)
        dense_features = output['dense_features'].squeeze()
        normalized_features = F.normalize(dense_features, dim=0).cpu().numpy()
        scores = output['scores'].squeeze().cpu().numpy()
        grid_keypoints = output['grid_keypoints'].cpu().numpy()
        keypoints = output['keypoints'].cpu().numpy()

        # keypoint features
        grid_pos_x = grid_keypoints[:, 0]
        grid_pos_y = grid_keypoints[:, 1]
        keypoint_features = normalized_features[:, grid_pos_x, grid_pos_y].T
        keypoint_scores = scores[grid_pos_x, grid_pos_y]

    # i, j -> u, v (numpy conv -> opencv conv)
    keypoints = keypoints[:, [1, 0]]
    grid_keypoints = grid_keypoints[:, [1, 0]]

    store_path = os.path.join(args.feat_dir, str(idx) + args.output_extension)
    with open(store_path, 'wb') as output_file:
        np.savez(output_file,
                 keypoints=keypoints,
                 scores=keypoint_scores,
                 descriptors=keypoint_features)

    # Logging
    gs = fig.add_gridspec(1, 3)
    ax_orig_img = fig.add_subplot(gs[0, 0])
    ax_orig_img.imshow(image, cmap='Greys')
    ax_orig_img.set_title(f'Original: {idx}')
    ax_orig_img.axis('off')

    ax_preprocessed_img = fig.add_subplot(gs[0, 1])
    preprocessed_img = show_tensor_image(input_image.squeeze(0), mean, std)
    ax_preprocessed_img.imshow(preprocessed_img, cmap='Greys')
    ax_preprocessed_img.scatter(x=[kp[0] for kp in keypoints],
                                y=[kp[1] for kp in keypoints],
                                s=1,
                                c='y')
    ax_preprocessed_img.set_title(f'Preprocessed: {idx}')
    ax_preprocessed_img.axis('off')

    ax_soft_detection = fig.add_subplot(gs[0, 2])

    ax_soft_detection.imshow(scores, cmap='Reds')
    ax_soft_detection.scatter(x=grid_pos_y, y=grid_pos_x, s=3, c='k')
    ax_soft_detection.set_title(f'Soft detection score: {idx}')
    ax_soft_detection.axis('off')
    if args.store_separate_pngs:
        model_name = os.path.basename(os.path.normpath(args.model_file))
        plt.savefig(os.path.join(log_dir, f'{idx}.png'))
    else:  # store to tensorboard
        writer.add_figure(f'model_{args.model_file}', fig, global_step=i)
    plt.clf()
