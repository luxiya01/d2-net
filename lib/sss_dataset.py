"""Custom PyTorch Dataset class for loading side scan sonar data"""

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from lib.utils import preprocess_image
from sss_data_processing.src.correspondence_getter import CorrespondenceGetter


class SSSDataset(Dataset):
    """Side scan sonar image patches dataset. Correspondences are given in numpy
    convention (x-axis pointing downwards, y-axis pointing to the right.)"""
    def __init__(self,
                 data_dir,
                 data_indices_file,
                 remove_trivial_pairs,
                 pos_round_to,
                 max_num_corr=1000,
                 preprocessing='torch',
                 img_type='norm_intensity',
                 min_overlap=.3):
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
            - max_num_corr: max pairs of correspondences given for a pair of
              images
            - min_overlap: minimum amount of overlap required to be considered
              as an overlapping image pair
        """
        self.data_dir = data_dir
        self.patches_dir = os.path.join(self.data_dir, 'patches')
        self.img_type = img_type
        self.preprocessing = preprocessing
        self.correspondence_getter = CorrespondenceGetter(
            self.data_dir, pos_round_to, data_indices_file)
        self.overlapping_pairs = self.correspondence_getter.get_all_pairs_with_target_overlap(
            min_overlap=min_overlap,
            max_overlap=.99,
            remove_trivial_pairs=remove_trivial_pairs)
        self.max_num_corr = max_num_corr

    def _load_image(self, idx):
        """Given a patch index, load the correponding img_type (norm_intensity or
        unnorm_intensity), scale up intensities to (0, 255) and convert from
        grayscale to rgb."""
        patch_name = '.'.join([str(idx), 'npz'])
        image = np.load(os.path.join(self.patches_dir,
                                     patch_name))[self.img_type]

        # image intensities (0, 1)-> (0, 255)
        image *= 255

        # grayscale -> 3 color channels
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)
        return image

    def __len__(self):
        """Number of overlapping pairs in the dataset"""
        return len(self.overlapping_pairs)

    def __getitem__(self, pair_idx):
        """Note that the correspondences are flipped from OpenCV convention
        given by the CorrespondenceGetter to numpy convention that will be used
        by the networks"""
        idx1, idx2 = self.overlapping_pairs[pair_idx]
        image1 = preprocess_image(self._load_image(idx1),
                                  preprocessing=self.preprocessing)
        image2 = preprocess_image(self._load_image(idx2),
                                  preprocessing=self.preprocessing)

        # Get correspondences in OpenCV convention (shape: [num_corr, 2])
        pos, corr1, corr2 = self.correspondence_getter.get_correspondence(
            idx1, idx2)
        num_corr = corr1.shape[0]

        # Select corresopndences
        if num_corr <= self.max_num_corr:
            idx = range(num_corr)
        else:
            idx = random.sample(range(num_corr), k=self.max_num_corr)
        pos, corr1, corr2 = pos[idx], corr1[idx], corr2[idx]

        # Convert from OpenCV convention to Numpy convention
        corr1 = corr1[:, [1, 0]]
        corr2 = corr2[:, [1, 0]]

        return {
            'idx1': idx1,
            'idx2': idx2,
            'image1': torch.from_numpy(image1.astype(np.float32)),
            'image2': torch.from_numpy(image2.astype(np.float32)),
            'overlap': self.correspondence_getter.overlap_matrix[idx1, idx2],
            'pos': torch.from_numpy(pos.astype(np.float32)),
            'corr1': torch.from_numpy(corr1.astype(np.float32)),
            'corr2': torch.from_numpy(corr2.astype(np.float32))
        }
