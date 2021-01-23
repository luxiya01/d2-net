import numpy as np

import cv2
import os

import shutil

import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm

import warnings

from lib.sss_dataset import SSSDataset
from lib.exceptions import NoGradientError
from lib.loss import loss_function
from lib.model import D2Net
from lib.utils import image_net_mean_std
import parse_args

# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Seed
torch.manual_seed(1)
if use_cuda:
    torch.cuda.manual_seed(1)
np.random.seed(1)

args = parse_args.get_args()
print(args)

# Create writer for logging to tensorboard
writer = SummaryWriter(args.log_dir)

# Creating CNN model
finetune_feature_extraction = True
if args.train_from_scratch:
    finetune_feature_extraction = False
model = D2Net(model_file=args.model_file,
              use_cuda=use_cuda,
              finetune_feature_extraction=finetune_feature_extraction,
              ignore_score_edges=args.ignore_score_edges,
              num_channels=args.num_channels)

# Optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=args.lr)
# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=2,
                                      gamma=.5,
                                      verbose=True)

# Dataset
mean, std = image_net_mean_std()
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.2),
    transforms.Normalize(mean=mean, std=std)
])


def get_log_img_pairs(dataset, log_img_len=10):
    indices = []
    log_img_len = min(len(dataset), log_img_len)
    for i in range(log_img_len):
        data = dataset[i]
        indices.append((data['idx1'], data['idx2']))
    return indices


training_dataset = SSSDataset(data_dir=args.data_dir,
                              ignore_edges=args.ignore_score_edges,
                              data_indices_file=args.data_indices_file,
                              remove_trivial_pairs=args.remove_trivial_pairs,
                              transform=data_transform,
                              img_type=args.img_type,
                              min_overlap=args.min_overlap,
                              max_overlap=.99,
                              max_num_corr=args.max_num_corr,
                              pos_round_to=5)
training_log_img_pairs = get_log_img_pairs(training_dataset)

print(f'Num training data: {len(training_dataset)}')
if args.use_validation:
    num_validation = max(int(len(training_dataset) * args.validation_size), 1)
    num_train = len(training_dataset) - num_validation
    print(f'num training data: {num_train}, validation data: {num_validation}')

    training_dataset, validation_dataset = torch.utils.data.random_split(
        training_dataset, lengths=[num_train, num_validation])
    training_log_img_pairs = get_log_img_pairs(training_dataset)
    validation_log_img_pairs = get_log_img_pairs(validation_dataset)

    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       shuffle=False)

training_dataloader = DataLoader(training_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=True)


# Define epoch function
def process_epoch(epoch_idx,
                  model,
                  loss_function,
                  optimizer,
                  dataloader,
                  device,
                  log_file,
                  args,
                  log_img_pairs,
                  train=True):
    epoch_losses = []

    torch.set_grad_enabled(train)

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, batch in progress_bar:
        if train:
            optimizer.zero_grad()

        batch['train'] = train
        batch['epoch_idx'] = epoch_idx
        batch['batch_idx'] = batch_idx
        batch['batch_size'] = args.batch_size
        batch['log_interval'] = args.log_interval
        batch['global_step'] = epoch_idx * len(dataloader) + batch_idx
        batch['mean'] = mean
        batch['std'] = std

        batch['log_img'] = False
        patch_indices = (batch['idx1'][0], batch['idx2'][0])
        if patch_indices in log_img_pairs:
            batch['log_img'] = True

        try:
            loss = loss_function(model,
                                 batch,
                                 device,
                                 writer=writer,
                                 safe_radius=args.safe_radius,
                                 margin=args.margin,
                                 ignore_score_edges=args.ignore_score_edges)
        except NoGradientError:
            continue

        current_loss = loss.data.cpu().numpy()[0]
        epoch_losses.append(current_loss)

        progress_bar.set_postfix(loss=('%.10f' % np.mean(epoch_losses)))

        if batch_idx % args.log_interval == 0:
            log_file.write('[%s] epoch %d - batch %d / %d - avg_loss: %f\n' %
                           ('train' if train else 'valid', epoch_idx,
                            batch_idx, len(dataloader), np.mean(epoch_losses)))
        # log to tensorboard
        writer.add_scalar('average [%s] loss' %
                          ('train' if train else 'valid'),
                          np.mean(epoch_losses),
                          global_step=batch['global_step'])
        writer.add_scalar('exact [%s] loss' % ('train' if train else 'valid'),
                          current_loss,
                          global_step=batch['global_step'])

        if train:
            loss.backward()
            optimizer.step()

    log_file.write(
        '[%s] epoch %d - avg_loss: %f\n' %
        ('train' if train else 'valid', epoch_idx, np.mean(epoch_losses)))
    log_file.flush()
    writer.flush()

    return np.mean(epoch_losses)


# Create the checkpoint directory
checkpoint_directory = os.path.join(args.log_dir, 'checkpoints')
if os.path.isdir(checkpoint_directory):
    print('[Warning] Checkpoint directory already exists.')
else:
    os.mkdir(checkpoint_directory)

# Open the log file for writing
log_file_path = os.path.join(args.log_dir, 'log.txt')
if os.path.exists(log_file_path):
    print(f'[Warning] Log file {log_file_path} already exists.')
log_file = open(log_file_path, 'a+')

# Initialize the history
train_loss_history = []
validation_loss_history = []

if args.use_validation:
    min_validation_loss = process_epoch(0,
                                        model,
                                        loss_function,
                                        optimizer,
                                        validation_dataloader,
                                        device,
                                        log_file,
                                        args,
                                        log_img_pairs=validation_log_img_pairs,
                                        train=False)
# Start the training
for epoch_idx in range(args.num_epochs):
    # Process epoch
    train_loss_history.append(
        process_epoch(epoch_idx,
                      model,
                      loss_function,
                      optimizer,
                      training_dataloader,
                      device,
                      log_file,
                      args,
                      log_img_pairs=training_log_img_pairs))

    if args.use_validation:
        validation_loss_history.append(
            process_epoch(epoch_idx + 1,
                          model,
                          loss_function,
                          optimizer,
                          validation_dataloader,
                          device,
                          log_file,
                          args,
                          log_img_pairs=validation_log_img_pairs,
                          train=False))

    # Save the current checkpoint
    checkpoint_path = os.path.join(
        checkpoint_directory,
        '%s.%02d.pth' % (args.checkpoint_prefix, epoch_idx))
    checkpoint = {
        'args': args,
        'epoch_idx': epoch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'validation_loss_history': validation_loss_history
    }
    torch.save(checkpoint, checkpoint_path)
    if (args.use_validation
            and validation_loss_history[-1] < min_validation_loss):
        min_validation_loss = validation_loss_history[-1]
        best_checkpoint_path = os.path.join(
            checkpoint_directory,
            '%s.best.pth.epoch%s' % (args.checkpoint_prefix, epoch_idx))
        shutil.copy(checkpoint_path, best_checkpoint_path)

    # Reduce the learning rate according to LR schedule
    scheduler.step()

# Close the log file
log_file.close()
writer.close()
