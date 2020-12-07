import argparse


# Argument parsing
def get_args():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--data_dir',
                        type=str,
                        required=True,
                        help='path to the dataset')
    parser.add_argument(
        '--data_indices_file',
        type=str,
        required=True,
        help=
        'path to the data indices file containing patch indices used for training'
    )
    parser.add_argument(
        '--remove_trivial_pairs',
        action='store_true',
        help=
        'whether trivial overlapping pairs should be removed from the dataset')
    parser.add_argument(
        '--img_type',
        type=str,
        default='unnormalised',
        help=
        'image type used (norm_intensity, norm_intensity_artefact_removed or unnorm_intensity)'
    )
    parser.add_argument(
        '--min_overlap',
        type=float,
        default=.1,
        help='minimum overlap required to be considered as an overlapping pair'
    )
    parser.add_argument(
        '--max_num_corr',
        type=int,
        default=500,
        help=
        'maximum number of correspondence pairs fed into the network per image pair'
    )

    parser.add_argument('--model_file',
                        type=str,
                        default=None,
                        help='path to the full model')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='number of training epochs')
    parser.add_argument(
        '--init_epoch',
        type=int,
        default=0,
        help=
        'initial epoch number (used when resume training from a checkpoint)')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='initial learning rate')
    parser.add_argument(
        '--safe_radius',
        type=int,
        default=4,
        help=
        'area (in feature map space) from which the negative samples will not be drawn'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=1.,
        help=
        'margin used in the loss function, the loss function requires dist(neg) > dist(pos) + margin'
    )
    parser.add_argument(
        '--ignore_score_edges',
        action='store_true',
        help='set detection scores at the edge of the feature map to 0')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='number of workers for data loading')

    parser.add_argument('--use_validation',
                        dest='use_validation',
                        action='store_true',
                        help='use the validation split')
    parser.set_defaults(use_validation=False)
    parser.add_argument(
        '--validation_size',
        type=float,
        default=.05,
        help='percentage of the dataset used as validation set')

    parser.add_argument('--log_interval',
                        type=int,
                        default=50,
                        help='loss logging interval')
    parser.add_argument('--log_dir',
                        type=str,
                        required=True,
                        help='logging directory')
    parser.add_argument('--checkpoint_prefix',
                        type=str,
                        default='d2',
                        help='prefix for training checkpoints')

    args = parser.parse_args()
    return args
