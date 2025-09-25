from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# from pygcn.utils import load_data, accuracy
import math
from torch.utils.data import TensorDataset, DataLoader
# from model1 import *
# from DGCM import *
from model import Model
from dataset import *
from sklearn.preprocessing import MinMaxScaler
import random
import os
import logging
from torch.utils.tensorboard import SummaryWriter
# from gpu_mem_track import MemTracker
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from ipdb import set_trace


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                        help='train the model')
parser.add_argument('--test', action='store_true',
                        help='test on test set')

parser.add_argument('--train_data_path', default='data/train',
                                help='the dir to store training dataset')
parser.add_argument('--valid_data_path', default='data/valid',
                                help='the dir to store validation dataset')
parser.add_argument('--test_data_path', default='data/test',
                                help='the dir to store testing dataset')
parser.add_argument('--summary_dir', default='output/writer/',
                                help='')


parser.add_argument('--input_feature_size', type=int, default=216, help='input node embedding size.')
parser.add_argument('--hidden_feature_size', type=int, default=150, help='hidden state size.')
parser.add_argument('--out_feature_size', type=int, default=45, help='out feature size.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the size of data in one epoch')
parser.add_argument('--patience', type=int, default=5,
                    help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
parser.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')
parser.add_argument('--eval_freq', type=int, default=300,
                                help='the frequency of evaluating on the valid set when training')

parser.add_argument('--query_sni2fea', default='pretrained_files/query_snippet2feature.npy',
                                help='a nested dict of query2snippet2feature')
parser.add_argument('--query_title2id', default='pretrained_files/query_title2id.npy',
                                help='a nested dict of query2title2id')
parser.add_argument('--img_fea', default='pretrained_files/img_feature_160.npy',
                                help='a dictionary of img_id to feature')
parser.add_argument('--all_sess_adj', default='pretrained_files/all_sess_adj.npy',
                                help='a dictionary of session to adjacent matrix')
parser.add_argument('--all_sess_img_id', default='pretrained_files/all_sess_img_id.npy',
                                help='a dictionary of session to img_id')
parser.add_argument('--all_sess_show_idx', default='pretrained_files/all_sess_show_idx.npy',
                                help='a dictionary of session to its documents index')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#     model.cuda()


def test(args, data):
    logger = logging.getLogger("DGCM")
    logger.info('Starting to test...')
    # model = Model(args, args.input_feature_size, args.hidden_feature_size, args.out_feature_size)
    
    logger.info('Reloading the model...')
    # model.load_model(model_dir=args.model_dir)
    model = torch.load('models/DGCM')
    model.test(data)
    logger.info('Done with model testing!')

def train(args, data):
    logger = logging.getLogger("DGCM")
    
    logger.info('Initialize the model...')
    model = Model(args, args.input_feature_size, args.hidden_feature_size, args.out_feature_size)
    
    logger.info('Training the model...')
    for epoch in range(args.epochs):
        model.train(epoch, data)
    logger.info('Done with model training!')

def run():
  train_data, val_data, test_data = ClickDataset(args, args.train_data_path), ClickDataset(args, args.valid_data_path), ClickDataset(args, args.test_data_path)
  # dataset = ClickDataset(args, data_path=args.data_path)
  # train_data, val_data, test_data = dataset[:len(dataset)*0.8], dataset[len(dataset)*0.8:len(dataset)*0.9], dataset[len(dataset)*0.9:]
  #   writer = SummaryWriter(log_dir="output/writer/")
  #   writer.add_scalar(tag='loss/lr', scalar_value=args.lr, global_step=0)
  
  if args.train:
    #   for epoch in range(args.epochs):
    train(args, train_data)
  if args.test:
    test(args, test_data)
#   writer.close()

  # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  # learning_rate = args.lr
  # query_sni2fea = np.load('npy/query_snippet2feature.npy', allow_pickle=True).item()
  # query_title2id = np.load('npy/query_title2id.npy', allow_pickle=True).item()
  # img_fea = np.load('npy/img_feature_160.npy', allow_pickle=True).item()
  # all_sess_adj = np.load('npy/id_bert_add_title_11/sess_adj.npy', allow_pickle=True).item()
  # all_sess_img_id = np.load('npy/id_bert_add_title_11/sess_img_id.npy', allow_pickle=True).item()
  # all_sess_show_idx = np.load('npy/id_bert_add_title_11/sess_show_idx.npy', allow_pickle=True).item()
  # patience = 0


if __name__ == '__main__':
    run()
