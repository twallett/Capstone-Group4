import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        print(f"Initialized weight parameter with shape {self.weight.shape}")
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            print(f"Initialized bias parameter with shape {self.bias.shape}")
        else:
            self.register_parameter('bias', None)
            print("Bias is set to None")

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        print(f"Reset weight parameter with standard deviation {stdv}")
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            print(f"Reset bias parameter with standard deviation {stdv}")

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        print(f"Computed support matrix with shape {support.shape}")

        output = torch.spmm(adj, support)
        print(f"Computed output matrix with shape {output.shape}")

        if self.bias is not None:
            output = output + self.bias
            print("Added bias to output")
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
