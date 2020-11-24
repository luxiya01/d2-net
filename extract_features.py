import argparse

import os
import numpy as np

from matplotlib import pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model import D2Net as D2NetSoftDetection
from lib.model_test import D2Net
from lib.utils import image_net_mean_std, show_tensor_image
from lib.pyramid import process_multiscale

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
parser.add_argument('--log_dir',
                    type=str,
                    required=True,
                    help='path to tensorboard logging dir')
parser.add_argument(
    '--img_type',
    type=str,
    default='norm_intensity_artefact_removed',
    help=
    'Image type used to extract features: (norm_intensity_artefact_removed, norm_intensity, unnorm_intensity)'
)
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

parser.add_argument('--max_edge',
                    type=int,
                    default=1600,
                    help='maximum image size at network input')
parser.add_argument('--max_sum_edges',
                    type=int,
                    default=2800,
                    help='maximum sum of image sizes at network input')

parser.add_argument('--output_extension',
                    type=str,
                    default='.d2-net',
                    help='extension for the output')
parser.add_argument('--output_type',
                    type=str,
                    default='npz',
                    help='output file type (npz or mat)')

parser.add_argument('--multiscale',
                    dest='multiscale',
                    action='store_true',
                    help='extract multiscale features')
parser.set_defaults(multiscale=False)

parser.add_argument(
    '--no-relu',
    dest='use_relu',
    action='store_false',
    help='remove ReLU after the dense feature extraction module')
parser.set_defaults(use_relu=True)

args = parser.parse_args()

print(args)

# Creating CNN model
model = D2Net(model_file=args.model_file,
              use_relu=args.use_relu,
              use_cuda=use_cuda)
soft_detection_model = D2NetSoftDetection(model_file=args.model_file,
                                          use_cuda=use_cuda)

# Tensorboard logging
if args.store_separate_pngs:
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
else:
    writer = SummaryWriter(args.log_dir)

# Process the patches directory
patches_dir = os.path.join(args.data_dir, 'patches')
files = [
    os.path.join(patches_dir, x) for x in os.listdir(patches_dir)
    if x.split('.')[-1] == 'npz' and x.split('.')[0].isnumeric()
]

outpath = os.path.join(args.data_dir, f'{args.feat_dir}')
if not os.path.exists(outpath):
    os.mkdir(outpath)
print(outpath)

mean, std = image_net_mean_std()
data_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=mean, std=std)])

for i, filename in tqdm(enumerate(files), total=len(files)):
    print(f'>> Generating features for path = {filename}')
    data = np.load(filename, allow_pickle=True)

    # keep range in (0, 1)
    image = data[args.img_type]

    idx = data['ids']

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image
    if max(resized_image.shape) > args.max_edge:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_edge / max(resized_image.shape)).astype('float')
    if sum(resized_image.shape[:2]) > args.max_sum_edges:
        resized_image = scipy.misc.imresize(
            resized_image,
            args.max_sum_edges / sum(resized_image.shape[:2])).astype('float')

    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    # image range -> (0, 1)
    resized_image = (resized_image - resized_image.min()) / (
        resized_image.max() - resized_image.min())
    input_image = data_transform(resized_image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Hard detection
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                input_image, model)
        else:
            keypoints, scores, descriptors = process_multiscale(input_image,
                                                                model,
                                                                scales=[1])
        # Soft detection score map
        batch = {
            'image1': input_image,
            'image2': input_image,
        }
        soft_detection_res = soft_detection_model(batch)
        soft_detection_scores = soft_detection_res['scores1'].cpu()
        soft_detection_scores = np.transpose(soft_detection_scores,
                                             [1, 2, 0]).squeeze()

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]

    store_path = os.path.join(outpath, str(idx) + args.output_extension)

    if args.output_type == 'npz':
        with open(store_path, 'wb') as output_file:
            np.savez(output_file,
                     keypoints=keypoints,
                     scores=scores,
                     descriptors=descriptors)
    elif args.output_type == 'mat':
        with open(store_path, 'wb') as output_file:
            scipy.io.savemat(
                output_file, {
                    'keypoints': keypoints,
                    'scores': scores,
                    'descriptors': descriptors
                })
    else:
        raise ValueError('Unknown output type.')

    # Tensorboard logging

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    ax_orig_img = fig.add_subplot(gs[0, 0])
    ax_orig_img.imshow(resized_image, cmap='Greys')
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
    ax_soft_detection.imshow(soft_detection_scores, cmap='Reds')
    ax_soft_detection.set_title(f'Soft detection score: {idx}')
    ax_soft_detection.axis('off')
    if args.store_separate_pngs:
        model_name = os.path.basename(os.path.normpath(args.model_file))
        plt.savefig(os.path.join(args.log_dir, f'{idx}.png'))
    else:  # store to tensorboard
        writer.add_figure(f'model_{args.model_file}', fig, global_step=i)
