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
from model44 import *
from sklearn.preprocessing import MinMaxScaler
import random
import os
from torch.utils.tensorboard import SummaryWriter
# from gpu_mem_track import MemTracker
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from ipdb import set_trace


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                        help='train the model')
parser.add_argument('--test', action='store_true',
                        help='test on test set')

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
parser.add_argument('--batch_size', type=int, default=128,
                    help='the size of data in one epoch')
parser.add_argument('--patience', type=int, default=5,
                    help='lr half when more than the patience times of evaluation\' loss don\'t decrease')
parser.add_argument('--lr_decay', type=float, default=0.5,
                                help='lr decay')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    model.cuda()


def test(args, data):
    logger.info('Initialize the model...')
    model = Model(args, args.input_feature_size, args.hidden_feature_size, args.out_feature_size)
    
    logger.info('Reloading the model...')
    model.load_model(model_dir=args.model_dir)
    model.test(data)
    logger.info('Done with model testing!')

def train(args, data):
    logger = logging.getLogger("DGCM")
    logger.info('Checking the data files...')
    if not args.train_file:
        assert 'train data file does not exist.'
    
    logger.info('Initialize the model...')
    model = Model(args, args.input_feature_size, args.hidden_feature_size, args.out_feature_size)
    
    logger.info('Training the model...')
    model.train(data)
    logger.info('Done with model training!')

def run():
  dataset = ClickDataset(args, data_path=args.data_path)
  train_data, val_data, test_data = dataset[:len(dataset)*0.8], dataset[len(dataset)*0.8:len(dataset)*0.9], dataset[len(dataset)*0.9:]

  if args.train:
      train(args, train_data)
  if args.test:
      test(args, test_data)

  # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  # learning_rate = args.lr
  # query_sni2fea = np.load('npy/query_snippet2feature.npy', allow_pickle=True).item()
  # query_title2id = np.load('npy/query_title2id.npy', allow_pickle=True).item()
  # img_fea = np.load('npy/img_feature_160.npy', allow_pickle=True).item()
  # all_sess_adj = np.load('npy/id_bert_add_title_11/sess_adj.npy', allow_pickle=True).item()
  # all_sess_img_id = np.load('npy/id_bert_add_title_11/sess_img_id.npy', allow_pickle=True).item()
  # all_sess_show_idx = np.load('npy/id_bert_add_title_11/sess_show_idx.npy', allow_pickle=True).item()
  # patience = 0

  # t_total = time.time()
  # writer = SummaryWriter(log_dir="results/")
  # writer.add_scalar(tag='loss/lr', scalar_value=learning_rate, global_step=0)
  
  # for epoch in range(args.epochs): # args.epochs
  #     # test()
  #     train(epoch)
  # writer.close()
  
  # print("Optimization Finished!")
  # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

if __name__ == '__main__':
    run()
