from re import X
import string
from joblib.logger import PrintTime
from nltk.util import pr
from numpy.lib.arraypad import pad
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import Tanh
from torch.nn.modules.conv import Conv1d
from torch.nn.utils.rnn import pad_sequence

from models.lstm import LSTM

# nfeat=128, #features.shape[1], 
#                 nhid=8, 
#                 nclass=self.labels + 1, 
#                 dropout=0.5, 
#                 nheads=8, 
#                 alpha=0.2,

class MGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, vocab_size):
        """Dense version of GAT with LSTM."""
        super(MGAT, self).__init__()
        self.dropout = dropout

        # self.textfeat =  LSTM(vocab_size, 24, 64, 1) # TODO
        # self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        
        # self.fc_layers = torch.nn.Sequential(
        #                                     nn.Linear(128, 64),
        #                                     nn.ReLU(),
        #                                     nn.Linear(64, nclass),
        #                                     )
        
        weight = [[1, -1]]
        self.weight_kernel = torch.tensor(weight, dtype=torch.float32).unsqueeze(0).to('cuda')

        self.lstm = nn.LSTM(input_size = 127, hidden_size = nclass, num_layers=1, batch_first=True)

    def forward(self, t, x, adj):
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # t = self.textfeat(t)
        # x = F.dropout(x, self.dropout, training=self.training)
        # # x = F.normalize(x)
        # xt = x * t
        # xhat = torch.zeros(256, 64).long().to('cuda')
        # xhat[:xt.shape[0], :] = xt
        # xhat = xhat.flatten()
        # out = self.fc_layers(xhat).unsqueeze(0)
        # return F.softmax(out, dim=1)
        
        x = x.view(1, 1, -1)
        x = nn.ConstantPad1d((0, 128 - x.shape[2]), 0)(x)
        x = F.conv1d(x, self.weight_kernel, stride=1)
        x = F.relu(x)
        print(x)
        exit()
        lstm_out, (ht, ct) = self.lstm(x)

        return F.softmax(ht[-1], dim=1)

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.W.requires_grad = True
        self.a.requires_grad = True

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
