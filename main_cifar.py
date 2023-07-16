import numpy as np
import torch.backends.cudnn as cudnn
import torch
import random
import argparse
import logging
import time
import os
import sys
from train import *
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import TensorDataset
from controller_rnn import *
import matplotlib.pyplot as plt
from utils import *
# Change the import path to your own pre-trained models
from pre_trained.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from pre_trained.cifar10_models.resnet import resnet18, resnet34
from pre_trained.cifar10_models.mobilenetv2 import mobilenet_v2
# Untrained model


# lr=0.001, epochs=100, max_val_iter=5, lmd=0, PATH='./'
# num_epoch_rnn
# batch_size_rnn
# max_iter_rnn



parser = argparse.ArgumentParser('model')
parser.add_argument('--batch_size_rl', type=int, default=5)
parser.add_argument('--k', type=int, default=1)
parser.add_argument('--max_iter', type=int, default=25)
parser.add_argument('--max_val_iter', type=int, default=2)
parser.add_argument('--filter_flag', type=int, default=1) # 0 - conv filter; 1 - all filters; 2 - conv para; 3- all para
parser.add_argument('--net', type=int, default=0) # 0 - vgg11_bn; 1 - resnet18; 2 - mobilenetv2
# parser.add_argument('--lmd', type=float, default=0)
parser.add_argument('--dataset', type=int, default=0) # 0 - cifar10; 1 - cifar100;
parser.add_argument('--lr_rl', type=float, default=0.01)
parser.add_argument('--num_epoch_rl', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--num_epoch_cnn', type=int, default=100)
parser.add_argument('--eps', type=float, default=8e-5)
parser.add_argument('--min_w', type=float, default=-0.14)
parser.add_argument('--max_w', type=float, default=0.24)
parser.add_argument('--PATH', type=str, default='./result/mobilenet_cifar_rnn.pth')
args = parser.parse_args()



def main():
# %% ========= Select GPU ==========
#     os.environ['CUDA_VISIBLE_DEVICES']='1'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: Could not find GPU! Use CPU only")

    # ========================== Fix seed for reproduce====================

    # seed_val = 66
    # random.seed(seed_val)
    # np.random.seed(seed_val)
    # torch.manual_seed(seed_val)
    # torch.cuda.manual_seed_all(seed_val)


    # %% ========= Get data ==========
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])
    # # transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=1)
    # Category information also needs to be given by us
    # classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')




    # %% ========= Define Net ==========


    # lmd = np.linspace(0,1, num=5).tolist()
    lmd=[0.0]
    # max_val_iter = 5
    B = -7e-4
    # B = [0 for _ in range(args.k)]
    # B_list = np.random.uniform(-6e-4, 3e-5, 32).tolist()
    if args.net == 0:
        B_list = [-7e-4] #vgg-11
        args.min_w = -0.13
        args.max_w = 0.19
        args.eps = 1e-4
        args.k = 1
        args.save_ori = './res_data/cifar_vgg11_ori.npy'
        args.save_new = './res_data/cifar_vgg11_new.npy'
        layer_list = [64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096, 10] #vgg_11

    if args.net == 1:
        B_list = [-0.0012] #resnet-18
        args.min_w = -0.14
        args.max_w = 0.24
        args.eps = 8e-5
        args.k = 1
        args.save_ori = './res_data/cifar_res18_ori.npy'
        args.save_new = './res_data/cifar_res18_new.npy'
        layer_list = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 10]  # resnet_18

    if args.net == 2:
        # B_list = [-2.5e-4] #mobilenet_v2
        B_list = [-0.00067]
        args.min_w = -0.13
        args.max_w = 0.19
        args.eps = 1.5e-4
        args.k = 5
        args.save_ori = './res_data/cifar_mob_ori.npy'
        args.save_new = './res_data/cifar_mob_new.npy'
        # layer_list = [32, 32, 96, 144, 144, 192, 192, 192, 384, 384, 384, 384, 576, 576, 576, 960, 960, 960, 10]
        layer_list = [32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192,
                      64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96,
                      576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280, 10]


    for i in range(len(lmd)):
        print('\n=============== Training with lambda = %s '%(lmd[i]))


        if args.filter_flag < 2:
            control = Controller_rnn(device, layer_list, B_list)
            control.to(device)
            loss_list, record = control.policy_gradient(args, lmd[i], trainloader, testloader)
            print('loss = %s\n' % loss_list)
            print('acc = %s\n' % record[0])
            print('change = %s\n' % record[-1])

        else:
            if args.net == 0:
                net = vgg11_bn(pretrained=True)
            if args.net == 1:
                net = resnet18(pretrained=True)
            if args.net == 2:
                net = mobilenet_v2(pretrained=True)
            net.to(device)
            layer_filters = [[4], [44], [10], [1], [5], [34], [49], [30], [10], [3], [6]]
            best_val_accuracy, net_dict_new, idx_list, ori_w, new_w, num_modi = Trainer(args, layer_filters, B, net, trainloader, testloader, device)
            np.save('ori_w_280.npy', ori_w)
            np.save('new_w_280.npy', new_w)



if __name__ == '__main__':
    main()
