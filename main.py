'''
Author: Little-Chen
Github: https://github.com/Little-Chen-T
Date: 2021-11-21 17:04:04
LastEditors: Little-Chen
LastEditTime: 2021-11-21 17:13:22
'''
# coding=utf-8

from utils.datasets import*
from utils.evaluate import*
from utils.others import*
from utils.train import*
from src.model import*
import torch
import numpy as np
import argparse
import os
import logging
import time
import datetime

def parse_args():
    desc = "Pytorch implementation of gesture recognition on CNN"
    parser = argparse.ArgumentParser(description=desc)

    # data
    parser.add_argument("--data_path", type=str, default='data/gestures', help="path of data")
    parser.add_argument("--batch_size_train", type=int, default=512, help="batch size of train data")
    parser.add_argument("--batch_size_test", type=int, default=500, help="batch size of test data")

    # result
    parser.add_argument('--results_path', type=str, default='/results',
                        help='the path of output images (generated and reconstructed)')
    parser.add_argument('--log_interval', type=float, default=50, help='interval between logging')

    # model
    parser.add_argument('--learn_rate', type=float, default=1e-2, help='learn_rate')
    parser.add_argument('--epoch', type=int, default=5, help='epoch')

    # others
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--no_cuda', type=int, default=0, help='use GPU or not')
    parser.add_argument('--GPU', type=int, default=-1, help='GPU')
  
    return check_args(parser.parse_args())

def main(args):
    
    #logging 
    args.results_path = os.getcwd() + '/' + args.results_path + '/' + 'epoch_' + str(args.epoch) + '_lr_' + str(args.learn_rate) + '_' + 'batch_size_train_' + str(args.batch_size_train) + '_' + str(datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    logging_path = args.results_path + '/log.txt'

    if not os.path.exists(args.results_path):
        
        os.makedirs(args.results_path)

    logging.basicConfig(filename=logging_path, level=logging.INFO)

    # set seed
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.cuda.set_device(args.GPU)
    device = torch.device("cuda" if use_cuda else "cpu")
    logging.info("used device: " + str(device))

    # get data
    train_dataset,test_dataset = load_data(args)
    train_dataloader,test_dataloader = get_dataloder(train_dataset,test_dataset, args)

    # get model
    model = MY_CNN();
    model.to(device)

    # set optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    time_start = time.time()
    train(model, optimizer, train_dataloader, test_dataloader,device,  args)
    time_end = time.time()

    logging.info("\nTraining time: " + str(time_end-time_start) + " seconds")

if __name__ == '__main__':

    # parse arguments
    args = parse_args()
    if args is None:
        exit()
    # main
    main(args)
