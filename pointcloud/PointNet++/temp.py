import torch
from collections import OrderedDict
a=torch.load('checkpoints/ptClsMixRI/fcAngle_M_axis_best.pth.tar')
k=list(a['model_state'].keys())
kn = k.copy()
kn[2]='SA_modules.0.mlps.0.1.coef_net.0.weight'
kn[3]='SA_modules.0.mlps.0.1.coef_net.0.bias'
new = OrderedDict()
for s,n in zip(k[:-2],kn[:-2]):
    new[n]=a['model_state'][s]
a['model_state'] = new
torch.save(a,'checkpoints/ptClsMixRI/fcAngle_M_axis_best.pth.tar')
