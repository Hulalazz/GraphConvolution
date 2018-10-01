"""The Graph Convolutional Networks at described in Kipf & Welling (2017), Eq.10.

Author: Su Wang.
"""


import torch
from torch.autograd import Variable
import torch.nn as nn
from utils import *


class GraphConvolution(nn.Module):
    """Single graph convolution layer."""
    
    def __init__(self, in_features, hidden_size):
        """Initializer.
        
        Args:
            in_features: embedding/feature size.
            hidden_size: hidden size for the graph convolution layer.
        """
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, hidden_size)
        
    def forward(self, A_hat, X):
        """Graph convolution on the input, see Kipf & Welling (2017), Eq.10."""
        return self.linear(A_hat.mm(X))


class GCN(nn.Module):
    """2-layered Graph Convolutional Network (Kipf & Welling, 2017, Eq.9)."""
    
    def __init__(self, in_features, hidden_size_1, hidden_size_2,
                 drop_prob=0.5):
        super(GCN, self).__init__()
        self.layer_1 = GraphConvolution(in_features, hidden_size_1)
        self.layer_2 = GraphConvolution(hidden_size_1, hidden_size_2)
        self.dropout = nn.Dropout(p=drop_prob)
        self.relu = nn.ReLU()
    
    def forward(self, A_hat, X):
        out = self.layer_1(A_hat, X)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.layer_2(A_hat, out)
        out = self.dropout(out)
        return out # NB: torch.nn.CrossEntropyLoss() takes unnormalized output.
 