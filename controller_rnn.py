# add recover loss drop to reward
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from utils import *
from train import Trainer
import argparse
from torch.autograd import Variable
from pre_trained.cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from pre_trained.cifar10_models.resnet import resnet18, resnet34
from pre_trained.cifar10_models.mobilenetv2 import mobilenet_v2




class Controller_rnn(nn.Module):
    '''
    Construct RNN controller

    Args:
        device: using cpu or gpu
        layer_list: list of the output channels of each conv layer
        B_list: list of values uniformly sampling from the weights range with the step epsilon
        embedding_dim: the one-hot encoding dimension of input, decided by the largest number of output channels
        hidden_dim: the hidden dimension of RNNCell
        round: each round samples a filter in each conv layer
        # batch: the input size will be [round, len(layer_list)+1, embedding_dim] if batch is true; otherwise it would be [1, round*(len(layer_list))+1, embedding_dim]
    '''
    def __init__(self, device,  layer_list, B_list, embedding_dim=256, hidden_dim=512, batch=4):
        super(Controller_rnn, self).__init__()

        self.device = device
        # self.trainer = trainer
        self.layer_list = layer_list
        self.B_list = B_list
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch = batch

        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.ReLU())

        # self.batch = batch
        # decode the hidden_out to the desired dimensions for each conv layer
        self.decoders = nn.ModuleList()
        for i in self.layer_list:
            self.decoders.append(nn.Linear(self.hidden_dim, i))
        # decode the B value
        # self.decoders.append(nn.Linear(self.hidden_dim, len(B_list)))
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_dim, 1)
        self.init_parameters()


    def init_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            decoder.bias.data.fill_(0)


    def forward(self, input, h_t):
        '''
        Get the hidden state and logits of output of a time step

        Args:
            input: state


        Returns:
            out_prob: the list of output probabilities (tensor)
            actions: the list of filter indexes
        '''

        # one-hot encoding
        out_prob = []
        actions = []


        out, h = self.rnn(input,h_t)

        for i, decoder in enumerate(self.decoders):
            logits = decoder(out[i])
            # print('======', logits.shape)
            prob = F.softmax(logits, dim=-1)
            # print('======', prob.shape)
            action_index = Categorical(probs=prob).sample()
            # action_index = np.random.choice(action_space, p=action_probs)
            actions.append(action_index.tolist()) # tensor
            # print('shape',torch.log(prob).shape)
            # print('idx', action_index.item())
            log_p = torch.log(prob).gather(1,action_index.reshape(-1,1)).squeeze()
            out_prob.append(log_p)

        return out_prob, actions

    def reset(self,k):
        '''
        Reset the states

        Returns:
             state: one-hot state
        '''

        state = torch.tensor([[random.randint(0,min(self.layer_list)) for _ in range(k)] for _ in range(len(self.layer_list))])
        state = F.one_hot(state, num_classes=self.embedding_dim).to(self.device)

        return state.float()




    def policy_gradient(self, arg, lmd, trainloader, testloader):
        '''
        Apply policy gradient algorithm to update the controller

        Args:
            arg: arguments of trianer
            trainer: used to train the CNN
            optimizer_rnn: controller optimizer
            batch_size: update a batch for stable training
            num_iter: the number of times to update the controller

        Returns:
            loss_list: loss updates of controller
            record: best result found
        '''
        loss_list = []
        record = None
        best = -1
        cnt = 0

        optimizer_rl = optim.Adam(params=self.parameters(), lr=arg.lr_rl)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        h_t = torch.zeros(1, arg.k, self.hidden_dim, dtype=torch.float, device=self.device)



        for ep in range(arg.num_epoch_rl):
            rewards = []
            log_probs = []
            actions = []
            states = []

            optimizer_rl.zero_grad()


            for k in range(arg.batch_size_rl):
                acc_cnn = None
                while acc_cnn == None:
                    state = self.reset(arg.k)
                    actions_p, actions_index = self.forward(state, h_t)

                    # h_t.detach_()
                    layer_filters = actions_index
                    # print('l\n',layer_filters)
                    # B_idx = actions_index[-1]
                    # B =  [self.B_list[n] for n in B_idx]

                    # reset the pre-trained model
                    if arg.dataset == 0:
                        if arg.net == 0:
                            net = vgg11_bn(pretrained=True)
                        if arg.net == 1:
                            net = resnet18(pretrained=True)
                        if arg.net == 2:
                            net = mobilenet_v2(pretrained=True)

                    net.to(device)
                    # get the reward acc

                    acc_cnn, model_dict, idx_list, new_dict, ori_w, new_w, layer_modi = Trainer(arg, lmd,
                                                                                                        layer_filters,
                                                                                                        self.B_list,
                                                                                                        net,
                                                                                                        trainloader,
                                                                                                        testloader,
                                                                                                        self.device)

                net.load_state_dict(new_dict)
                new_acc = inference(net,device,testloader)
                print(f'Recover_Val: | Acc: {new_acc:.5f}')

                reward = -acc_cnn
                rewards.append(reward)
                # states.append(state)

                if reward >= best:
                    # print('reward = %s\n' % reward)
                    # print('best = %s\n' % best)
                    record = [acc_cnn, idx_list,layer_modi]
                    torch.save(model_dict, arg.PATH)
                    np.save('ori_w_vgg11_rnn.npy', ori_w)
                    np.save('new_w_vgg11_rnn.npy', new_w)
                    best = reward
                    cnt = 0
                else:
                    cnt += 1
                # prob = [sum(i) for i in actions_log_p[:-1]]
                # log_prob = sum(prob) + actions_log_p[-1][0]
                if arg.k > 1:
                    log_prob = sum(sum(actions_p))
                else:
                    log_prob = sum(actions_p)
                # print('ac',actions_p)
                # print('log',log_prob)
                log_probs.append(log_prob)


            if cnt > arg.max_iter:
                print("Reward did not improve for the last {} runs. Early stopping..."
                      .format(arg.max_iter))
                break

            # print('reward',rewards)
            b = np.array(rewards).mean()
            # print('==========reward',b)
            tmp = [i-b for i in rewards]
            # r = torch.tensor(tmp)
            # prob_tensor = torch.tensor(log_probs)
            # print('prob_t',prob_tensor[0])

            loss = 0
            for i in range(arg.batch_size_rl):
                loss += tmp[i] * log_probs[i]
            # loss = loss/arg.batch_size_rl
            # loss = sum(log_probs)

            # print('loss',loss)
            loss_list.append(loss.item())
            loss.backward()
            optimizer_rl.step()

            print('+++++++++++++++++++++++++++++++[control_epoch:%d] Loss: %.03f '% (ep + 1, loss.item()))

        return loss_list, record





