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
        default='norm_intensity',
        help='image type used (norm_intensity or unnorm_intensity)')
    parser.add_argument(
        '--min_overlap',
        type=float,
        default=.3,
        help='minimum overlap required to be considered as an overlapping pair'
    )
    parser.add_argument(
        '--max_overlap',
        type=float,
        default=.99,
        help='maximum overlap required to be considered as an overlapping pair'
    )
    parser.add_argument(
        '--pos_round_to',
        type=int,
        default=5,
        help=
        'the decimals that the physical positions will be rounded to during correspondence calculation, pos_round_to = 5 -> round to closest 0.2'
    )

    parser.add_argument('--preprocessing',
                        type=str,
                        default=None,
                        help='image preprocessing (None, caffe or torch)')
    parser.add_argument('--model_file',
                        type=str,
                        default='models/d2_tf.pth',
                        help='path to the full model')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='number of training epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='number of workers for data loading')

    parser.add_argument('--use_validation',
                        dest='use_validation',
                        action='store_true',
                        help='use the validation split')
    parser.set_defaults(use_validation=False)
    parser.add_argument(
        '--validation_size',
        type=float,
        default=.1,
        help='percentage of the dataset used as validation set')

    parser.add_argument('--log_interval',
                        type=int,
                        default=50,
                        help='loss logging interval')
    parser.add_argument('--log_file',
                        type=str,
                        default='log.txt',
                        help='loss logging file')
    parser.add_argument('--log_dir',
                        type=str,
                        required=True,
                        help='logging directory')

    parser.add_argument('--checkpoint_directory',
                        type=str,
                        default='checkpoints',
                        help='directory for training checkpoints')
    parser.add_argument('--checkpoint_prefix',
                        type=str,
                        default='d2',
                        help='prefix for training checkpoints')

    args = parser.parse_args()
    return args