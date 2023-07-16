

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt

def modify_layer(layer_list,B,eps,pre_model,testloader,device):
    dic = {}
    mask = {}
    idx_list = []
    ori_w = []
    layer_modi = []

    sum = 0
    total = 0
    cnt = 0
    tmp_dict = pre_model.state_dict()
    for name, para in pre_model.named_parameters():
        # only modify conv layer

        if len(para.shape) > 1 and para.shape[-1] > 1: # used for resnet
        # if len(para.shape) > 1 : # used for mobilenet
            tmp = torch.zeros_like(para)
            para_re = torch.zeros_like(para)
            filter_idx = layer_list[cnt]
            # print(para.shape)
            # print('== name =%s, filter_idx =%s'%(name, filter_idx))
            b = B[0]
            for i in range(len(filter_idx)):
                f = filter_idx[i]
                # b = B[i]

                tmp[f] = torch.tensor((abs(para[f]-b)<eps),dtype=torch.float32)
                para_re[f] = torch.tensor((abs(para[f]-b)<eps)*b,dtype=torch.float32)

            mask[name] = tmp

            indexes = torch.nonzero(mask[name], as_tuple=True)
            idx_list.append(indexes)
            w = para[indexes]
            ori_w.extend(w.cpu().detach().numpy())
            # print('Weights before change \n',para)
            # print('Weights after change \n',mask[name])
            total += len(mask[name].flatten())
            num_modi = mask[name].sum()
            layer_modi.append(num_modi.item())
            sum += num_modi
            dic[name] = (1-mask[name])*para + para_re
            cnt += 1
        else:
            mask[name] = torch.zeros_like(para)
            dic[name] = para
            idx_list.append([])
    tmp_dict.update(dic)
    # pre_model.load_state_dict(tmp_dict)
    # new_acc = inference(pre_model,device, testloader)

    # for name, para in pre_model.named_parameters():
    #     if mask[name][44].sum() != 0:
    #         print('utils',name, para[44])
    #         print('mask',mask[name][44].sum())
    #         break
    return mask, tmp_dict, total, idx_list, ori_w, sum, layer_modi




def get_weights(model, idx_list):
    w_list = []
    cnt = 0
    for name, para in model.named_parameters():

        idx = idx_list[cnt]
        w = para[idx]
        # ori_w.extend()
        w_list.extend(w.cpu().detach().numpy())
        cnt += 1
    return w_list

def top_k_idx(model, idx_list, ori_w, new_w):
    w_list = []
    cnt = 0
    dic = {}
    tmp_dict = model.state_dict()
    for name, para in model.named_parameters():

        idx = idx_list[cnt]
        w = para[idx]
        # ori_w.extend()
        w_list.extend(w.cpu().detach().numpy())
        cnt += 1
    return w_list


def inference(net, device, testloader):
    net.to(device)
    net.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total
    return acc


def sparsity(mask,model,B,t):
    sum = 0
    change = 0
    weights = []
    #
    for name, para in model.named_parameters():

        sum += mask[name].sum()
        # delta = [para.flatten()[i]-B for i in range(len(para.flatten())) if mask[name].flatten()[i] is True]
        change += abs(para.flatten()-B>t).sum()
        weights = None
        #weights.extend(para[mask[name].long()].detach().cpu().flatten()-B)

    return weights, change, sum, change/sum


