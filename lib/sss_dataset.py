"""Custom PyTorch Dataset class for loading side scan sonar data"""

import os
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from sss_data_processing.src.correspondence_getter import CorrespondenceGetter
from sss_data_processing.src.utils import load_correspondence


class SSSDataset(Dataset):
    """Side scan sonar image patches dataset. Correspondences are given in numpy
    convention (x-axis pointing downwards, y-axis pointing to the right.)"""
    def __init__(
        self,
        data_dir,
        data_indices_file,
        remove_trivial_pairs,
        pos_round_to=5,  #default round to .2
        transform=None,
        remove_edge_corr=False,
        one_channel=False,
        plot_gt_correspondence=False,
        max_num_corr=1000,
        img_type='norm_intensity_artefact_removed',
        min_overlap=.3,
        max_overlap=.99):
        """
        Args:
            - data_dir: path to the sss patch data
            - data_indices_file: path to the text files describing the patch indices
              to include in the dataset (e.g. for separating training and
              testing)
            - remove_trivial_pairs: if True, trivial overlapping pair of images
              (patches cropped out from overlapping regions of the same sss
              image) are removed from the dataset
            - pos_round_to: the decimals that the physical positions will be rounded to
              (e.g. 1 = round to closest integer, 2 = round to closest .5, 10 =
              round to closest .1)
            - one_channel: if True, grayscale image is kept as is, else the
              image is converted into three channels (RGB) by copying the
              grayscale channel
            - plot_gt_correspondence: plot groundtruth correspondences
            - max_num_corr: max pairs of correspondences given for a pair of
              images
            - min_overlap: minimum amount of overlap required to be considered
              as an overlapping image pair
            - max_overlap: maximum amount of overlap required to be considered
              as an overlapping image pair
        """
        self.data_dir = data_dir
        self.patches_dir = os.path.join(self.data_dir, 'patches')
        self.img_type = img_type
        self.one_channel = one_channel
        self.correspondence_getter = CorrespondenceGetter(
            self.data_dir, pos_round_to, data_indices_file)
        self.overlapping_pairs = self.correspondence_getter.get_all_pairs_with_target_overlap(
            min_overlap=min_overlap,
            max_overlap=max_overlap,
            remove_trivial_pairs=remove_trivial_pairs)
        self.plot_gt_correspondence = plot_gt_correspondence
        self.max_num_corr = max_num_corr
        self.remove_edge_corr = remove_edge_corr
        if not transform:
            transform = transforms.Compose([transforms.ToTensor()])
        self.transform = transform

    def _load_image(self, idx):
        """Given a patch index, load the correponding img_type (norm_intensity or
        unnorm_intensity), convert from grayscale to rgb if not one_channel.
        Keep range in (0, 1)"""
        patch_name = '.'.join([str(idx), 'npz'])
        image = np.load(os.path.join(self.patches_dir,
                                     patch_name))[self.img_type]
        # image range -> (0, 1)
        image = (image - image.min()) / (image.max() - image.min())

        image = image[:, :, np.newaxis]
        # grayscale -> 3 color channels
        if not self.one_channel:
            image = np.repeat(image, 3, -1)
        return image

    def __len__(self):
        """Number of overlapping pairs in the dataset"""
        return len(self.overlapping_pairs)

    def _sample_correspondences(self, idx1, idx2):
        """Sample correspondences <= max_num_corr for image pair (idx1, idx2).
        The correspondences are given by CorrespondenceGetter in OpenCV
        convention."""
        # Get correspondences in OpenCV convention (shape: [num_corr, 2])
        try:
            pos, corr1, corr2 = load_correspondence(
                os.path.join(self.correspondence_getter.out_dir,
                             f'{idx1}_{idx2}.corr'))
        except Exception as e:
            print(
                f'Calculate correspondences for ({idx1}, {idx2}) on the fly...'
            )
            pos, corr1, corr2 = self.correspondence_getter.get_correspondence(
                idx1, idx2, store=True)
        if self.remove_edge_corr:
            lower = 16
            upper = 240
            valid_idx = np.where(
                np.logical_and(
                    np.logical_and(
                        np.logical_and(corr1[:, 0] > lower,
                                       corr1[:, 0] < upper),
                        np.logical_and(corr1[:, 1] > lower,
                                       corr1[:, 1] < upper)),
                    np.logical_and(
                        np.logical_and(corr2[:, 0] > lower,
                                       corr2[:, 0] < upper),
                        np.logical_and(corr2[:, 1] > lower,
                                       corr2[:, 1] < upper))))
            pos, corr1, corr2 = pos[valid_idx], corr1[valid_idx], corr2[
                valid_idx]
        num_corr = corr1.shape[0]

        # Sample corresopndences
        if self.max_num_corr is None or num_corr <= self.max_num_corr:
            idx = range(num_corr)
        else:
            idx = random.sample(range(num_corr), k=self.max_num_corr)
        pos, corr1, corr2 = pos[idx], corr1[idx], corr2[idx]

        return pos, corr1, corr2

    def _plot_correspondences(self, idx1, idx2, corr1, corr2):
        self.correspondence_getter.plot_correspondence(idx1,
                                                       idx2,
                                                       corr1,
                                                       corr2,
                                                       plot_keypoints=True,
                                                       store=True)

    def __getitem__(self, pair_idx):
        """Note that the correspondences are flipped from OpenCV convention
        given by the CorrespondenceGetter to numpy convention that will be used
        by the networks"""
        idx1, idx2 = self.overlapping_pairs[pair_idx]
        image1 = self.transform(self._load_image(idx1))
        image2 = self.transform(self._load_image(idx2))

        pos, corr1, corr2 = self._sample_correspondences(idx1, idx2)

        if self.plot_gt_correspondence:
            self._plot_correspondences(idx1, idx2, corr1, corr2)

        # Convert from OpenCV convention to Numpy convention
        corr1 = corr1[:, [1, 0]]
        corr2 = corr2[:, [1, 0]]

        return {
            'idx1': idx1,
            'idx2': idx2,
            'image1': image1.float(),
            'image2': image2.float(),
            'overlap': self.correspondence_getter.overlap_matrix[idx1, idx2],
            'pos': torch.from_numpy(pos.astype(np.float32)),
            # Correspondences in OpenCV convention
            'corr1': torch.from_numpy(corr1.astype(np.float32)),
            'corr2': torch.from_numpy(corr2.astype(np.float32))
        }
