import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from qpu_ops import *
from qpu_layers import QPU

def quaternion_power_multi_bias_first(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i**2 + j**2 + k**2 + 1e-12)
    theta = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
    if bias is not None:
        theta = theta + bias
    theta = weight * theta
    
    mul = torch.sin(theta) / norm_v
    r = torch.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k

def qpu_linear_multi_bias_first(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1]
    in_channels = in_channels // 4
    out_channels = weight.shape[0]

    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)

    r, i, j, k = quaternion_power_multi_bias_first(r, i, j, k, weight, bias)
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = quaternion_chained_prod(r, i, j, k, -1)
    return torch.cat([r, i, j, k], dim=-1)

class QPU_MultiBiasFst(QPU):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU_MultiBiasFst, self).__init__(in_features, out_features, bias)
        self.bias = Parameter(torch.Tensor(self.out_features,self.in_features))

    def forward(self, input):
        output = qpu_linear_multi_bias_first(input, self.weight, self.bias)
        return quaternion_normalize(output, dim=-1)

def quaternion_power_multi_bias(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i**2 + j**2 + k**2 + 1e-12)
    theta = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
    theta = weight * theta
    if bias is not None:
        theta = theta + bias
    
    mul = torch.sin(theta) / norm_v
    r = torch.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k

def qpu_linear_multi_bias(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1]
    in_channels = in_channels // 4
    out_channels = weight.shape[0]

    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)

    r, i, j, k = quaternion_power_multi_bias(r, i, j, k, weight, bias)
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = quaternion_chained_prod(r, i, j, k, -1)
    return torch.cat([r, i, j, k], dim=-1)

class QPU_MultiBias(QPU):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU_MultiBias, self).__init__(in_features, out_features, bias)
        self.bias = Parameter(torch.Tensor(self.out_features,self.in_features))

    def forward(self, input):
        output = qpu_linear_multi_bias(input, self.weight, self.bias)
        return quaternion_normalize(output, dim=-1)

def quaternion_power_bias_last(r, i, j, k, weight, bias):
    """
    r, i, j, k: (*, 1, C_in)
    weight: (C_out, C_in)
    bias: (C_out)
    return: [cos(w * (acos(r) + bias)), sin(w * (acos(r) + bias)) v / |v|]
    """
    # Compute new theta
    norm_v = torch.sqrt(i**2 + j**2 + k**2 + 1e-12)
    theta = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
    theta = weight * theta
    if bias is not None:
        theta = theta + bias.unsqueeze(-1)
    
    mul = torch.sin(theta) / norm_v
    r = torch.cos(theta)
    i = i * mul
    j = j * mul
    k = k * mul
    return r, i, j, k

def qpu_linear_bias_last(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1]
    in_channels = in_channels // 4
    out_channels = weight.shape[0]

    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)

    r, i, j, k = quaternion_power_bias_last(r, i, j, k, weight, bias)
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = quaternion_chained_prod(r, i, j, k, -1)
    return torch.cat([r, i, j, k], dim=-1)

class QPU_BiasLast(QPU):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU_BiasLast, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        output = qpu_linear_bias_last(input, self.weight, self.bias)
        return quaternion_normalize(output, dim=-1)



def qpu_linear_sum_reduce(input, weight, bias):
    """
    input: (*, C_in * 4) -> (*, C_out * 4)
    """
    in_channels = input.shape[-1]
    in_channels = in_channels // 4
    out_channels = weight.shape[0]

    r, i, j, k = input.unsqueeze(-2).split(in_channels, dim=-1)

    r, i, j, k = quaternion_power_bias(r, i, j, k, weight, bias)
    r, i, j, k = QuaternionRemoveZeros.apply(r, i, j, k)
    r, i, j, k = r.sum(-1), i.sum(-1), j.sum(-1), k.sum(-1)
    norm_l2 = torch.sqrt(r**2 + i**2 + j**2 + k**2 + 1e-12)
    r, i, j, k = r/norm_l2, i/norm_l2, j/norm_l2, k/norm_l2
    return torch.cat([r,i,j,k], dim=-1)

class QPU_SumReduce(QPU):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU_SumReduce, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        output = qpu_linear_sum_reduce(input, self.weight, self.bias)
        return quaternion_normalize(output, dim=-1)