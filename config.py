import argparse


def config(name='NNDL Final Project'):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument('--index_device', '-d', default=4, type=int,
                        help='-1: using CPU; nonnegative num x: using x-th GPU')
    parser.add_argument('--path_saving', default='./save', type=str,
                        help='path for saving all files, containing sub-directories, e.g., checkpoints, runs, logs')
    parser.add_argument('--name_experiment', '-nexp', default='tmp', type=str)
    parser.add_argument('--delete_previous_results', '-dpr', default=False, action='store_true',
                        help='whether delete results of previous experiments')
    parser.add_argument('--is_validation', default=False, action='store_true')
    parser.add_argument('--epoch_validation', default=1, type=int)
    parser.add_argument('--save_all_models', default=False, type=bool)
    parser.add_argument('--save_better_models', default=False, type=bool)
    parser.add_argument('--random_seed', type=int, default=233, help='random seed')

    # model
    parser.add_argument('--network', default='resnet18')

    # dataset
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--batch_size_training', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--batch_size_validation', '-bs_v', default=128, type=int)
    parser.add_argument('--num_workers_training', '-nwt', default=4, type=int)
    parser.add_argument('--num_workers_validation', '-nwv', default=4, type=int)
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs to train (default: 20)')

    # optimizer
    parser.add_argument('--name_optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--name_scheduler', type=str, default='multisteplr')

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_nesterov', default=True, action='store_false')
    parser.add_argument('--scheduler_multisteplr_milestones', default=(60, 120, 160))
    parser.add_argument('--scheduler_multisteplr_gamma', type=float, default=0.2)

    parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
    parser.add_argument('--cutmix', action='store_true', default=False, help='apply cutmix')
    parser.add_argument('--mixup', action='store_true', default=False, help='apply mixup')

    parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16, help='length of the holes')

    parser.add_argument('--beta', type=float, default=1.0, help='parameter for cutmix')
    parser.add_argument('--cutmix_prob', type=float, default=1.0, help='parameter for cutmix')

    parser.add_argument('--alpha', type=float, default=1.0, help='parameter for mixup')

    args = parser.parse_args()

    return args
