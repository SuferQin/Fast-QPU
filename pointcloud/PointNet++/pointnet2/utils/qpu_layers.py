import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
from pointnet2.utils.qpu_ops import *

class QPU(nn.Module):
    """Quaternion product units. Apply weights on scalar part. Then perform chained Hamilton product.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QPU, self).__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight)
            a = math.sqrt(6 / (fan_in + fan_out))
            init.uniform_(self.bias, -a, a)

    def forward(self, input):
        output = qpu_linear(input, self.weight, self.bias)
        return quaternion_normalize(output, dim=-1)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'


class AngleAxisMap(nn.Module):
    """
    change the scalar part of a quaternion
    """
    def __init__(self, dim=-1, rinv=False):
        super(AngleAxisMap, self).__init__()
        self.dim = dim
        self.rinv = rinv

    def forward(self, input):
        r, i, j, k  = input.split(input.shape[self.dim] // 4, self.dim)
        r = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
        if self.rinv:
            return r
        sinTheta = torch.sin(r)
        i/=sinTheta
        j/=sinTheta
        k/=sinTheta
        return torch.cat((r, i, j, k),dim=self.dim)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'dim=' + str(self.dim) \
            + 'rinv=' + str(self.rinv) + ')'

class QuaterPostProcess(nn.Module):
    TYPE_LEN = {'r':1,'theta':1,'im':3,'axis':3,'angle':2,'inner':1}
    TYPE_MAP = {
        'im':'i,j,k',
        'angle':'alpha,beta',
        'axis':'x,y,z',
    }
    def __init__(self, dim=-1, out_type=['theta']):
        super(QuaterPostProcess, self).__init__()
        self.dim = dim
        self.out_type = out_type
        self.cmd = 'torch.cat([{}],-1)'.format(','.join([self.TYPE_MAP.get(ot,ot) for ot in self.out_type]))

    def outfeat(self,infeat_real):
        self.in_feat = infeat_real
        self.out_feat = sum([self.TYPE_LEN[ot] for ot in self.out_type])*(infeat_real//4) - int('inner' in self.out_type)
        return self.out_feat

    def forward(self,inputs):
        r,i,j,k = inputs.split(inputs.shape[self.dim] // 4, self.dim)
        if self.out_type == ['r'] or self.out_type == ['im'] or self.out_type == ['r','im']:
            return eval(self.cmd)
        theta = torch.acos(torch.clamp(r, min=-1+1e-6, max=1-1e-6))
        sinTheta = torch.sin(theta)
        x=i/sinTheta
        y=j/sinTheta
        z=k/sinTheta
        if 'angle' in self.out_type:
            alpha = torch.acos(torch.clamp(z, min=-1+1e-6, max=1-1e-6))
            beta = torch.acos(torch.clamp(x/torch.sin(alpha), min=-1+1e-6, max=1-1e-6))
        if 'inner' in self.out_type:
            inner = x[...,:-1]*x[...,1:]+y[...,:-1]*y[...,1:]+z[...,:-1]*z[...,1:]
            inner = torch.acos(inner.clamp(min=-1+1e-6, max=1-1e-6))
        return eval(self.cmd)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_feat) \
            + ', out_features=' + str(self.out_feat) + ')'