import argparse
import time

import numpy as np
import torch

#from networks import old_preprocess_images
import utils
from paac import PAACLearner
from train import get_network_and_environment_creator, eval_network, evaluate


def print_dict(d, name=None):
    title = ' '.join(['=='*10, '{}','=='*10])
    if name is not None:
        title.format(name)

    print(title)
    for k in sorted(d.keys()):
        print('  ', k,':', d[k])
    print('='*len(title))


def fix_args_for_test(args, train_args):
    for k, v in train_args.items():
        if getattr(args, k, None) == None: #this includes cases where args.k is None
            setattr(args, k, v)

    args.max_global_steps = 0
    args.random_seed = np.random.randint(1000)

    if args.framework == 'vizdoom':
        args.reward_coef = 1.
        args.step_delay = 0.15
    elif args.framework == 'atari':
        args.random_start = True
        args.single_life_episodes = False
        args.step_delay = 0
    else:
        args.step_delay = 5.
    return args


def load_trained_network(net_creator, checkpoint_path):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    network = net_creator()
    network.load_state_dict(checkpoint['network_state_dict'])
    return network, checkpoint['last_step']

if __name__=='__main__':
    devices = ['gpu', 'cpu'] if torch.cuda.is_available() else ['cpu']
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('folder', type=str, help="Folder with a trained model.")
    parser.add_argument('-tc', '--test_count', default=1, type=int, help="Number of episodes to test the model", dest="test_count")
    parser.add_argument('-g', '--greedy', action='store_true', help='Determines whether to use a stochastic or deterministic policy')
    parser.add_argument('-d', '--device', default=devices[0], type=str, choices=devices,
        help="Device to be used ('cpu' or 'gpu'). Use CUDA_VISIBLE_DEVICES to specify a particular gpu", dest="device")
    parser.add_argument('-v', '--visualize', action='store_true')


    args = parser.parse_args()
    train_args = utils.load_args(folder=args.folder)
    args = fix_args_for_test(args, train_args)

    checkpoint_path = utils.join_path(
        args.folder, PAACLearner.CHECKPOINT_SUBDIR, PAACLearner.CHECKPOINT_LAST
    )
    net_creator, env_creator = get_network_and_environment_creator(args)
    network, steps_trained = load_trained_network(net_creator, checkpoint_path)

    use_rnn = hasattr(network, 'get_initial_state')

    print_dict(vars(args), 'ARGS')
    print('Model was trained for {} steps'.format(steps_trained))
    if args.visualize:
        num_steps, rewards = evaluate.visual_eval(
            network, env_creator, args.greedy,
            use_rnn, args.test_count, verbose=1, delay=args.step_delay)
    else:
        num_steps, rewards = eval_network(
            network, env_creator, args.test_count,
            use_rnn, greedy=args.greedy)

    print('Perfromed {0} tests for {1}.'.format(args.test_count, args.game))
    print('Mean number of steps: {0:.3f}'.format(np.mean(num_steps)))
    print('Mean R: {0:.2f}'.format(np.mean(rewards)), end=' | ')
    print('Max R: {0:.2f}'.format(np.max(rewards)), end=' | ')
    print('Min R: {0:.2f}'.format(np.min(rewards)), end=' | ')
    print('Std of R: {0:.2f}'.format(np.std(rewards)))
