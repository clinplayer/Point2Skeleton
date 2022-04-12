import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations=1,  # number of relation types (adjacency matrices)
                 K=1,  # GCN is K<=1, else ChebNet
                 adj_sq=False,
                 scale_identity=False):

        super(GraphConv, self).__init__()
        self.fc = nn.Conv1d(in_channels=in_features * K * n_relations, out_channels=out_features, kernel_size=1,
                            bias=False)
        self.n_relations = n_relations

        assert K > 0, ('filter scale must be greater than 0', K)
        self.K = K
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    # L*X or (chebyshev polynomial)*X
    def chebyshev_basis(self, L, X, K):
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])
            Xt = torch.cat(Xt, dim=2)
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X)

    # provide Laplacian matrix in batch #A:(B,N,N)
    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)
        A_hat = A

        if self.K < 2 or self.scale_identity:
            I = torch.eye(N).unsqueeze(0).to('cuda')
            if self.scale_identity:
                I = 2 * I
            if self.K < 2:
                A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, x, A):
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []
        x = x.permute(0, 2, 1)
        for rel in range(self.n_relations):
            L = self.laplacian_batch(A[:, :, :, rel])
            x_hat.append(self.chebyshev_basis(L, x, self.K))
        x = self.fc(torch.cat(x_hat, 2).permute(0, 2, 1))
        return x
