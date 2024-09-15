import argparse
import torch
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=True,
                    choices=['acm', 'yelp', 'dblp', 'mag'])
args, unknown = parser.parse_known_args()
dataset = args.dataset
# dataset = 'acm'


def acm_params():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='acm',
                        choices=['acm', 'yelp', 'dblp', 'mag'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=False)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0)
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-r', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-augment_type', type=str, default='random',
                        choices=['random', 'generative'])
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    parser.add_argument('-lr_dis', type=float, default=0.001)
    parser.add_argument('-aug_lambda', type=float, default=0.01)

    # GSL Module
    parser.add_argument('-k', type=int, default=15)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0001)
    parser.add_argument('-hidden_dim_cls', type=int, default=64)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    args = parser.parse_args()

    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='dblp',
                        choices=['acm', 'yelp', 'dblp', 'mag'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=False)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.01)
    parser.add_argument('-w_decay', type=float, default=0.0)
    parser.add_argument('-hidden_dim', type=int, default=64)
    parser.add_argument('-rep_dim', type=int, default=32)
    parser.add_argument('-proj_dim', type=int, default=32)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-r', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-augment_type', type=str, default='random',
                        choices=['random', 'generative'])
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    parser.add_argument('-lr_dis', type=float, default=0.001)
    parser.add_argument('-aug_lambda', type=float, default=1.)

    # GSL Module
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0001)
    parser.add_argument('-hidden_dim_cls', type=int, default=64)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    args = parser.parse_args()

    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='yelp',
                        choices=['acm', 'yelp', 'dblp', 'mag'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=False)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-w_decay', type=float, default=0)
    parser.add_argument('-hidden_dim', type=int, default=128)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-contrast_batch_size', type=int, default=0)
    parser.add_argument('-nlayers', type=int, default=2)
    parser.add_argument('-r', type=int, default=2)

    # GCL Module -Augmentation
    parser.add_argument('-augment_type', type=str, default='random',
                        choices=['random', 'generative'])
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    parser.add_argument('-lr_dis', type=float, default=0.001)
    parser.add_argument('-aug_lambda', type=float, default=1.)

    # GSL Module
    parser.add_argument('-k', type=int, default=15)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=200)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.001)
    parser.add_argument('-hidden_dim_cls', type=int, default=32)
    parser.add_argument('-dropout_cls', type=float, default=0.5)
    parser.add_argument('-dropedge_cls', type=float, default=0.)
    parser.add_argument('-nlayers_cls', type=int, default=2)
    parser.add_argument('-patience_cls', type=int, default=10)

    args = parser.parse_args()

    return args

def mag_params():
    parser = argparse.ArgumentParser()
    # Experimental setting
    parser.add_argument('-dataset', type=str, default='mag',
                        choices=['acm', 'yelp', 'dblp', 'mag'])
    parser.add_argument('-ntrials', type=int, default=5)
    parser.add_argument('-sparse', type=int, default=True)
    parser.add_argument('-eval_freq', type=int, default=50)
    parser.add_argument('-downstream_task', type=str, default='classification',
                        choices=['classification', 'clustering'])
    parser.add_argument('-gpu', type=int, default=0)

    # GCL Module - Framework
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-w_decay', type=float, default=0)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-rep_dim', type=int, default=64)
    parser.add_argument('-proj_dim', type=int, default=64)
    parser.add_argument('-dropout', type=float, default=0.)
    parser.add_argument('-contrast_batch_size', type=int, default=2560)
    parser.add_argument('-nlayers', type=int, default=3)
    parser.add_argument('-r', type=int, default=3)

    # GCL Module -Augmentation
    parser.add_argument('-augment_type', type=str, default='random',
                        choices=['random', 'generative'])
    parser.add_argument('-dropedge_rate', type=float, default=0.5)
    parser.add_argument('-lr_dis', type=float, default=0)
    parser.add_argument('-aug_lambda', type=float, default=0)

    # GSL Module
    parser.add_argument('-k', type=int, default=15)
    parser.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])

    # Evaluation Network (Classification)
    parser.add_argument('-epochs_cls', type=int, default=500)
    parser.add_argument('-lr_cls', type=float, default=0.001)
    parser.add_argument('-w_decay_cls', type=float, default=0.0001)
    parser.add_argument('-hidden_dim_cls', type=int, default=256)
    parser.add_argument('-dropout_cls', type=float, default=0.)
    parser.add_argument('-dropedge_cls', type=float, default=0.25)
    parser.add_argument('-nlayers_cls', type=int, default=3)
    parser.add_argument('-patience_cls', type=int, default=10)

    args = parser.parse_args()

    return args


def set_params():
    if dataset == "acm":
        args = acm_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == 'yelp':
        args = yelp_params()
    elif dataset == 'mag':
        args = mag_params()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return args
