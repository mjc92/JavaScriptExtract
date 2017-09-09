import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.backends.cudnn as cudnn

cudnn.benchmark = True
import datetime
import torch
import argparse
import sys
from packages.train import train
from packages.test import test
from packages.functions import str2bool
from packages.copy_model import copy_model

parser = argparse.ArgumentParser()

# arguments related to the dataset
parser.add_argument('--train_root', type=str, default='data/types/all/all_train.txt', help='data file')
parser.add_argument('--val_root', type=str, default='data/types/all/all_val.txt', help='data file')
parser.add_argument('--test_root', type=str, default='data/types/all/all_test.txt', help='data file')
parser.add_argument('--dict_root', type=str, default='data/types/all/dict_1000.json',
                    help='directory of dictionary file')
parser.add_argument('--save_dir', type=str, default='saves', help='where to save model & info')
parser.add_argument("--max_oovs", type=int, default=20,
                    help='max number of OOVs to accept in a sample')

# arguments related to model training and inference
parser.add_argument("--mode", type=str, help='train/test mode. Error if unspecified')
parser.add_argument("--epochs", type=int, default=20, help='Number of epochs. Set by default to 20')
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--batch", type=int, default=64, help='batch size')
parser.add_argument("--k", type=int, default=5, help='for top-k accuracy')
parser.add_argument("--single", type=str2bool, default=True,
                    help='whether to include the task of selecting from multiple lines')
parser.add_argument("--cuda", type=str2bool, default=True, help='whether to use cuda')
parser.add_argument("--log", type=str2bool, default=False, help='whether to use tensorboard')
parser.add_argument("--load", type=str, default=None, help='whether to load from a previous model')
parser.add_argument("--copy", type=str2bool, default=False, help='whether to copy all related files to a folder')

# arguments related to the model structure itself
parser.add_argument("--hidden", type=int, default=256, help='size of hidden dimension')
parser.add_argument("--embed", type=int, default=256, help='size of embedded word dimension')
parser.add_argument("--n_layers", type=int, default=2, help='number of layers for transformer model')
parser.add_argument("--n_head", type=int, default=8, help='number of heads for transformer model')
parser.add_argument("--max_in_seq", type=int, default=100, help='max length of input')
parser.add_argument("--max_out_seq", type=int, default=100, help='max length of output')
parser.add_argument("--similarity", type=str, default='lstm_cosine', help='similarity measure to use')
parser.add_argument("--encoder", type=str, default='lstm', help='encoder type to use')

# etc
parser.add_argument("--gpu", type=int, default=0, help='which gpu to use')
parser.add_argument("--val_freq", type=int, default=10, help='how many times to perform evaluation')
parser.add_argument("--save_freq", type=int, default=1000, help='how many steps for a save')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# if not os.path.exists(args.save_dir):
#     os.mkdir(args.save_dir)




def main(args):
    if args.copy:
        copy_model(args)
    if args.mode == 'train':
        print("Train mode")
        train(args)
    elif args.mode =='train_sim':
        from packages.train_sim import train_sim
        train_sim(args)
    elif args.mode == 'test':
        print("Test mode")
        test(args)
    else:
        print("Error: please specify --mode as 'train' or 'test'")


if __name__ == "__main__":
    main(args)
