import matplotlib
import matplotlib.pyplot as plt
import random

import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from lib.utils import (grid_positions, upscale_positions, downscale_positions,
                       savefig, imshow_image)
from lib.exceptions import NoGradientError, EmptyTensorError

matplotlib.use('Agg')


def loss_function(model,
                  batch,
                  device,
                  writer,
                  margin=1,
                  safe_radius=30,
                  scaling_steps=3,
                  min_num_corr=1):
    output = model({
        'image1': batch['image1'].to(device),
        'image2': batch['image2'].to(device)
    })

    loss = torch.tensor(np.array([0], dtype=np.float32), device=device)
    has_grad = False

    n_valid_samples = 0
    for idx_in_batch in range(batch['image1'].size(0)):
        # patch indices
        idx1 = batch['idx1'][idx_in_batch]
        idx2 = batch['idx2'][idx_in_batch]
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

        pos1 = corr1.T
        fmap_pos1 = downscale_positions(pos1, scaling_steps).round().int()
        pos2 = corr2.T
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

        if batch['batch_idx'] % batch['log_interval'] == 0:
            # log image to tensorboard
            fig = plot_intermediate_results(pos1, pos2, fmap_pos1, fmap_pos2,
                                            output, batch, idx_in_batch,
                                            scaling_steps)
            writer.add_figure(
                f'GT correspondences ({idx1}, {idx2}) safe_radius = {safe_radius}',
                fig,
                global_step=batch['epoch_idx'])

    if not has_grad:
        raise NoGradientError

    loss = loss / n_valid_samples

    return loss


def pos_to_matches(pos1_aux, pos2_aux, idx):
    kp1 = [
        cv2.KeyPoint(x=pos1_aux[1, i], y=pos1_aux[0, i], _size=.25**2)
        for i in idx
    ]
    kp2 = [
        cv2.KeyPoint(x=pos2_aux[1, i], y=pos2_aux[0, i], _size=.25**2)
        for i in idx
    ]
    matches = [cv2.DMatch(i, i, 0) for i in idx]
    return kp1, kp2, matches


def plot_intermediate_results(pos1, pos2, fmap_pos1, fmap_pos2, output, batch,
                              idx_in_batch, scaling_steps):
    idx1 = batch['idx1'][idx_in_batch]
    idx2 = batch['idx2'][idx_in_batch]

    pos1_aux = pos1.cpu().numpy()
    pos2_aux = pos2.cpu().numpy()
    idx = range(pos1_aux.shape[1])

    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    ax_img1 = fig.add_subplot(gs[0, 0])
    img1 = imshow_image(batch['image1'][idx_in_batch].cpu().numpy(),
                        preprocessing=batch['preprocessing'])
    ax_img1.imshow(img1)
    ax_img1.set_title(f'Image1: {idx1}')
    ax_img1.axis('off')

    ax_img2 = fig.add_subplot(gs[0, 1])
    img2 = imshow_image(batch['image2'][idx_in_batch].cpu().numpy(),
                        preprocessing=batch['preprocessing'])
    ax_img2.imshow(img2)
    ax_img2.set_title(f'Image2: {idx2}')
    ax_img2.axis('off')

    ax_img_match = fig.add_subplot(gs[0, 2:])
    kp1, kp2, matches = pos_to_matches(pos1_aux, pos2_aux, idx)
    img_match = cv2.drawMatches(img1,
                                kp1,
                                img2,
                                kp2,
                                matches,
                                None,
                                matchColor=(145, 232, 144, .5))
    ax_img_match.imshow(img_match)
    ax_img_match.set_title('GT correspondences')
    ax_img_match.axis('off')
    print(f'img_match range: {img_match.min()}, {img_match.max()}')

    ax_fmap1 = fig.add_subplot(gs[1, 0])
    img_fmap1 = output['scores1'][idx_in_batch].data.cpu().numpy()
    ax_fmap1.imshow(img_fmap1, cmap='Reds')
    ax_fmap1.set_title(f'Soft detection scores image1: {idx1}')
    ax_fmap1.axis('off')

    ax_fmap2 = fig.add_subplot(gs[1, 1])
    img_fmap2 = output['scores2'][idx_in_batch].data.cpu().numpy()
    ax_fmap2.imshow(img_fmap2, cmap='Reds')
    ax_fmap2.set_title(f'Soft detection scores image2: {idx2}')
    ax_fmap2.axis('off')

    red_cm = matplotlib.cm.get_cmap('Reds')
    ax_fmap_match = fig.add_subplot(gs[1, 2:])
    norm_img_fmap1 = (img_fmap1 - img_fmap1.min()) / (img_fmap1.max() -
                                                      img_fmap1.min())
    mapped_img_fmap1 = red_cm(norm_img_fmap1)
    mapped_img_fmap1 = np.round(mapped_img_fmap1 * 255).astype(np.uint8)
    mapped_img_fmap1 = cv2.resize(cv2.cvtColor(mapped_img_fmap1,
                                               cv2.COLOR_RGB2BGR),
                                  dsize=tuple(img1.shape[:2]),
                                  interpolation=cv2.INTER_NEAREST)
    norm_img_fmap2 = (img_fmap2 - img_fmap2.min()) / (img_fmap2.max() -
                                                      img_fmap2.min())
    mapped_img_fmap2 = red_cm(norm_img_fmap2)
    mapped_img_fmap2 = np.round(mapped_img_fmap2 * 255).astype(np.uint8)
    mapped_img_fmap2 = cv2.resize(cv2.cvtColor(mapped_img_fmap2,
                                               cv2.COLOR_RGB2BGR),
                                  dsize=tuple(img2.shape[:2]),
                                  interpolation=cv2.INTER_NEAREST)

    fmap_kp1, fmap_kp2, fmap_matches = pos_to_matches(
        upscale_positions(fmap_pos1, scaling_steps),
        upscale_positions(fmap_pos2, scaling_steps), idx)
    print(
        f'img_fmap1 range: {mapped_img_fmap1.min()}, {mapped_img_fmap1.max()}')
    img_fmap_match = cv2.drawMatches(cv2.cvtColor(mapped_img_fmap1,
                                                  cv2.COLOR_RGB2BGR),
                                     kp1,
                                     cv2.cvtColor(mapped_img_fmap2,
                                                  cv2.COLOR_RGB2BGR),
                                     kp2,
                                     fmap_matches,
                                     None,
                                     matchColor=(0, 22, 120, .5))

    print(
        f'img_fmap_match range: {img_fmap_match.min()}, {img_fmap_match.max()}'
    )
    ax_fmap_match.imshow(img_fmap_match)
    ax_fmap_match.set_title('GT correspondences in feature map')
    ax_fmap_match.axis('off')

    tmpfig = plt.figure(2)
    plt.imshow(red_cm(img_fmap2))
    plt.savefig('mapped_img_fmap2_matplotlib.png')

    return fig


def fmap_pos_to_idx(fmap_pos, w):
    """Given a tensor of feature map positions, return the corresponding
    flattened index (used to index the corresponding descriptors)"""
    ids = fmap_pos[0, :] * w + fmap_pos[1, :]
    return ids.long()


def plot_network_res(pos1, pos2, batch, idx_in_batch, output):
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

    plt.scatter(pos1_aux[1, :],
                pos1_aux[0, :],
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
    plt.scatter(pos2_aux[1, :],
                pos2_aux[0, :],
                s=0.25**2,
                c=col,
                marker=',',
                alpha=0.5)
    plt.axis('off')
    plt.subplot(1, n_sp, 4)
    plt.imshow(output['scores2'][idx_in_batch].data.cpu().numpy(), cmap='Reds')
    plt.axis('off')
    savefig('train_vis/%s.%02d.%02d.%d.%d.%d.overlap_%02d.png' %
            ('train' if batch['train'] else 'valid', batch['epoch_idx'],
             batch['batch_idx'] // batch['log_interval'], idx_in_batch,
             batch['idx1'][idx_in_batch], batch['idx2'][idx_in_batch],
             batch['overlap'][idx_in_batch] * 100),
            dpi=300)
    plt.close()
