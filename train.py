# TODO: 1) locate weights; 2) check the magnitude of modified weights

import torch
import numpy as np
import random
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
# from sample import *
import matplotlib.pyplot as plt
from utils import *
# from model import *
from pre_trained.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from pre_trained.cifar10_models.resnet import resnet18, resnet34
from pre_trained.cifar10_models.mobilenetv2 import mobilenet_v2



def Trainer(arg, lmd, layer_filters, B, net, trainloader, testloader,device):

    plotloss = []
    plotauc = []
    val_accuracy = 100.

    # %% ========= setting ==========
    # Use the cross entropy loss function in the neural network toolbox nn
    criterion = nn.CrossEntropyLoss()
    # Use SGD (stochastic gradient descent) optimization, learning rate is 0.001, momentum is 0.9

    optimizer = optim.Adam(net.parameters(), lr=arg.lr)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)


    mask, new_dict, total_w, idx_list, ori_w, cnt_w, layer_modi= modify_layer(layer_filters, B, arg.eps, net, testloader,device)

    print('layer_filters = %s, B = %s' % (layer_filters, B))



    for epoch in range(arg.num_epoch_cnn):  # Specify how many epochs to cycle through the training
        # set to the eval mode to fix the paramaters of batchnorm
        net.eval()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0

        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Initialize the grad value of the parameter to
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # Use cross entropy to calculate loss for output and labels
            loss_c = criterion(outputs, targets)
            # print('loss_c \n',loss_c)
            # l1 norm
            l1_norm = 0


            for name, para in net.named_parameters():

                # tmp = (para-ori_dict[name])*mask[name]
                tmp = para  * mask[name]
                l1_norm += tmp.abs().sum()

            # print('l1_norm \n',l1_norm)
            # l1_norm = [abs(para*mask[name]).sum() ]

            #loss
            loss = -loss_c + lmd * l1_norm
            # print('loss \n', loss)
            # Backpropagation
            loss.backward()
            for name, para in net.named_parameters():
                para.grad *= mask[name].long()


            optimizer.step()
            with torch.no_grad():
                for name, para in net.named_parameters():
                    # param.clamp_(-arg.clip, arg.clip)
                    if arg.filter_flag < 2:
                        para.data = (1-mask[name]) * para + (mask[name]*para).clamp_(arg.min_w, arg.max_w)
                    else:
                        para.data = ~mask[name] * para + (mask[name] * para).clamp_(arg.min_w, arg.max_w)

            # loss.item() converted to numpy
            # loss itself is of Variable type, so use loss.data[0] to get its Tensor, because it is a scalar, so take 0
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)  # Update the number of test pictures
            correct += (predicted == targets).sum()  # Update the number of correctly classified pictures


        min_w = []
        max_w = []
        cnt_w_re = 0
        layer_modi_re = []
        for name, para in net.named_parameters():
            if len(para.shape) > 1 :
                tmp = abs(para - B[0]) * mask[name]
                min_w_tmp = torch.min(tmp)
                max_w_tmp = torch.max(tmp)
                min_w.append(min_w_tmp)
                max_w.append(max_w_tmp)
                modi_w = tmp > arg.eps
                # indexes = modi_w.nonzero()
                # idx_list.append(indexes)
                num_modi_re = modi_w.sum()
                cnt_w_re += num_modi_re
                layer_modi_re.append(num_modi_re.item())
        min_w_v = min(min_w)
        max_w_v = max(max_w)
        print(
            '==========================[epoch:%d] Loss: %.03f | Acc: %.3f%% | count:%d | layer_modi = %s \n| count_re:%d | layer_modi_re = %s'
            % (epoch + 1, loss.item(),  100. * correct / total, cnt_w,  layer_modi, cnt_w_re,  layer_modi_re))

        print('==========================min: %.5f | max: %.5f '
            % (min_w_v, max_w_v))


        acc = inference(net,device,testloader)
        print(f'Val: | Acc: {acc:.5f}')

        if acc < val_accuracy:
            val_accuracy = acc
            val_iter = 0
            new_w = get_weights(net, idx_list)
            # torch.save(net.state_dict(), arg.PATH)

        else:
            val_iter = val_iter + 1
        if val_iter == arg.max_val_iter:

            print("Validation accuracy did not improve for the last {} validation runs. Early stopping..."
                  .format(arg.max_val_iter))
            break


    return val_accuracy, net.state_dict(), idx_list, new_dict, ori_w, new_w, layer_modi
