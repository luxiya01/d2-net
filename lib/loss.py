import matplotlib
import matplotlib.pyplot as plt
import random

import numpy as np

import torch
import torch.nn.functional as F

from lib.utils import (grid_positions, upscale_positions, downscale_positions,
                       savefig, imshow_image)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')


def loss_function(model,
                  batch,
                  device,
                  margin=1,
                  safe_radius=4,
                  scaling_steps=3,
                  min_num_corr=128,
                  max_num_corr=1000,
                  plot=False):
    output = model({
        'image1': batch['image1'].to(device),
        'image2': batch['image2'].to(device)
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):
        # Network output
        dense_features1 = output['dense_features1'][idx_in_batch]
        c, h1, w1 = dense_features1.size()
        scores1 = output['scores1'][idx_in_batch].view(-1)

        dense_features2 = output['dense_features2'][idx_in_batch]
        _, h2, w2 = dense_features2.size()
        scores2 = output['scores2'][idx_in_batch].view(-1)

        all_descriptors1 = F.normalize(dense_features1.view(c, -1), dim=0)
        descriptors1 = all_descriptors1
        all_descriptors2 = F.normalize(dense_features2.view(c, -1), dim=0)
        descriptors2 = all_descriptors2

        # Sample GT correspondences (assume already in numpy convention)
        corr1 = batch['corr1'][idx_in_batch].to(device)  # [num_corr, 2]
        corr2 = batch['corr2'][idx_in_batch].to(device)  # [num_corr, 2]
        num_corr = corr1.shape[0]
        # Skip the pair if not enough GT correspondences are available
        if num_corr < min_num_corr:
            continue
        if num_corr <= max_num_corr:
            idx = range(num_corr)
        else:
            idx = random.sample(range(num_corr), k=max_num_corr)
        pos1 = corr1[idx].T
        fmap_pos1 = downscale_positions(pos1, scaling_steps).round().int()
        pos2 = corr2[idx].T
        fmap_pos2 = downscale_positions(pos2, scaling_steps).round().int()

        # Descriptors at the corresponding positions
        ids_pos1 = fmap_pos_to_idx(fmap_pos1, w1)
        descriptors1 = descriptors1[:, ids_pos1]
        scores1 = scores1[ids_pos1]

        ids_pos2 = fmap_pos_to_idx(fmap_pos2, w2)
        descriptors2 = descriptors2[:, ids_pos2]
        scores2 = scores2[ids_pos2]

        positive_distance = 2 - 2 * (descriptors1.t().unsqueeze(
            1) @ descriptors2.t().unsqueeze(2)).squeeze()

        all_fmap_pos2 = grid_positions(h2, w2, device)
        position_distance = torch.max(torch.abs(
            fmap_pos2.unsqueeze(2).float() - all_fmap_pos2.unsqueeze(1)),
                                      dim=0)[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors1.t() @ all_descriptors2)
        negative_distance2 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1)[0]

        all_fmap_pos1 = grid_positions(h1, w1, device)
        position_distance = torch.max(torch.abs(
            fmap_pos1.unsqueeze(2).float() - all_fmap_pos1.unsqueeze(1)),
                                      dim=0)[0]
        is_out_of_safe_radius = position_distance > safe_radius
        distance_matrix = 2 - 2 * (descriptors2.t() @ all_descriptors1)
        negative_distance1 = torch.min(
            distance_matrix + (1 - is_out_of_safe_radius.float()) * 10.,
            dim=1)[0]

        diff = positive_distance - torch.min(negative_distance1,
                                             negative_distance2)

        loss = loss + (torch.sum(scores1 * scores2 * F.relu(margin + diff)) /
                       torch.sum(scores1 * scores2))

        has_grad = True
        n_valid_samples += 1

        if plot and batch['batch_idx'] % batch['log_interval'] == 0:
            #TODO: remove one of these - only one of them is correct...
            plot_network_res(pos1,
                             pos2,
                             batch,
                             idx_in_batch,
                             output,
                             flip_xy=True)
            plot_network_res(pos1,
                             pos2,
                             batch,
                             idx_in_batch,
                             output,
                             flip_xy=False)

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def plot_network_res(pos1, pos2, batch, idx_in_batch, output, flip_xy):
    pos1_aux = pos1.cpu().numpy()
    pos2_aux = pos2.cpu().numpy()
    k = pos1_aux.shape[1]
    col = np.random.rand(k, 3)
    n_sp = 4
    plt.figure()
    plt.subplot(1, n_sp, 1)
    im1 = imshow_image(batch['image1'][idx_in_batch].cpu().numpy(),
                       preprocessing=batch['preprocessing'])
    plt.imshow(im1)

    if flip_xy:
        plt.scatter(pos1_aux[1, :],
                    pos1_aux[0, :],
                    s=0.25**2,
                    c=col,
                    marker=',',
                    alpha=0.5)
    else:
        plt.scatter(pos1_aux[0, :],
                    pos1_aux[1, :],
                    s=0.25**2,
                    c=col,
                    marker=',',
                    alpha=0.5)
    plt.axis('off')
    plt.subplot(1, n_sp, 2)
    plt.imshow(output['scores1'][idx_in_batch].data.cpu().numpy(), cmap='Reds')
    plt.axis('off')
    plt.subplot(1, n_sp, 3)
    im2 = imshow_image(batch['image2'][idx_in_batch].cpu().numpy(),
                       preprocessing=batch['preprocessing'])
    plt.imshow(im2)
    if flip_xy:
        plt.scatter(pos2_aux[1, :],
                    pos2_aux[0, :],
                    s=0.25**2,
                    c=col,
                    marker=',',
                    alpha=0.5)
    else:
        plt.scatter(pos2_aux[0, :],
                    pos2_aux[1, :],
                    s=0.25**2,
                    c=col,
                    marker=',',
                    alpha=0.5)
    plt.axis('off')
    plt.subplot(1, n_sp, 4)
    plt.imshow(output['scores2'][idx_in_batch].data.cpu().numpy(), cmap='Reds')
    plt.axis('off')
    savefig('train_vis/%s.%02d.%02d.%d.%d.%d.overlap_%02d_flipxy_%s.png' %
            ('train' if batch['train'] else 'valid', batch['epoch_idx'],
             batch['batch_idx'] // batch['log_interval'], idx_in_batch,
             batch['idx1'][idx_in_batch], batch['idx2'][idx_in_batch],
             batch['overlap'][idx_in_batch] * 100, str(flip_xy)),
            dpi=300)
    plt.close()


def fmap_pos_to_idx(fmap_pos, w):
    """Given a tensor of feature map positions, return the corresponding
    flattened index (used to index the corresponding descriptors)"""
    ids = fmap_pos[0, :] * w + fmap_pos[1, :]
    return ids.long()
