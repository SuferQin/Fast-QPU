import torch
import torch.nn as nn
from models.mlp_lstm import QLSTM
from qpu_layers import AngleAxisMap
from qpu_layers_ablation import QPU_BiasLast,QPU_SumReduce,QPU_MultiBias,QPU_MultiBiasFst

class QLSTM_MBF(QLSTM):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM_MBF, self).__init__(in_channels, num_joints, num_frames, num_cls, config)
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.Sequential(
                QPU_MultiBiasFst(self.num_joints * 4, self.feat_dim),
                QPU_MultiBiasFst(self.feat_dim, self.feat_dim*4),
                AngleAxisMap(dim=-1,rinv=True)
            )
        else:
            self.mlp = nn.Sequential(
                QPU_MultiBiasFst(self.num_joints * 4, self.feat_dim),
                QPU_MultiBiasFst(self.feat_dim, self.feat_dim),
                AngleAxisMap(dim=-1,rinv=False)
            )

class QLSTM_MB(QLSTM):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM_MB, self).__init__(in_channels, num_joints, num_frames, num_cls, config)
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.Sequential(
                QPU_MultiBias(self.num_joints * 4, self.feat_dim),
                QPU_MultiBias(self.feat_dim, self.feat_dim*4),
                AngleAxisMap(dim=-1,rinv=True)
            )
        else:
            self.mlp = nn.Sequential(
                QPU_MultiBias(self.num_joints * 4, self.feat_dim),
                QPU_MultiBias(self.feat_dim, self.feat_dim),
                AngleAxisMap(dim=-1,rinv=False)
            )

class QLSTM_BL(QLSTM):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM_BL, self).__init__(in_channels, num_joints, num_frames, num_cls, config)
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.Sequential(
                QPU_BiasLast(self.num_joints * 4, self.feat_dim),
                QPU_BiasLast(self.feat_dim, self.feat_dim*4),
                AngleAxisMap(dim=-1,rinv=True)
            )
        else:
            self.mlp = nn.Sequential(
                QPU_BiasLast(self.num_joints * 4, self.feat_dim),
                QPU_BiasLast(self.feat_dim, self.feat_dim),
                AngleAxisMap(dim=-1,rinv=False)
            )

class QLSTM_SR(QLSTM):
    def __init__(self, in_channels, num_joints, num_frames, num_cls, config):
        super(QLSTM_SR, self).__init__(in_channels, num_joints, num_frames, num_cls, config)
        if 'rinv' in config and config['rinv']:
            self.mlp = nn.Sequential(
                QPU_SumReduce(self.num_joints * 4, self.feat_dim),
                QPU_SumReduce(self.feat_dim, self.feat_dim*4),
                AngleAxisMap(dim=-1,rinv=True)
            )
        else:
            self.mlp = nn.Sequential(
                QPU_SumReduce(self.num_joints * 4, self.feat_dim),
                QPU_SumReduce(self.feat_dim, self.feat_dim),
                AngleAxisMap(dim=-1,rinv=False)
            )