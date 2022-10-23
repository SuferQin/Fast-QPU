from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import etw_pytorch_utils as pt_utils
from collections import namedtuple

from pointnet2.utils.pointnet2_modules import PointnetSAModule, PointnetSAModuleQPU
from pointnet2.utils import qpu_layers

def model_fn_decorator(criterion):
    ModelReturn = namedtuple("ModelReturn", ["preds", "loss", "acc"])

    def model_fn(model, data, epoch=0, eval=False):
        with torch.set_grad_enabled(not eval):
            inputs, labels = data
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            preds = model(inputs)
            labels = labels.view(-1)
            loss = criterion(preds, labels)

            _, classes = torch.max(preds, -1)
            acc = (classes == labels).float().sum() / labels.numel()

            return ModelReturn(preds, loss, {"acc": acc.item(), "loss": loss.item()})

    return model_fn


class Pointnet2SSG(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True, type_r=None):
        super(Pointnet2SSG, self).__init__()

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=32,
                mlp=[input_channels, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz)
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256, bn=True)
            .dropout(0.5)
            .fc(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2SSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))


class Pointnet2SSGQPU(nn.Module):
    r"""
        PointNet2 with single-scale grouping
        Classification network

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier
        input_channels: int = 3
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, input_channels=3, use_xyz=True, type_r=['theta']):
        super(Pointnet2SSGQPU, self).__init__()

        post_process = qpu_layers.QuaterPostProcess(dim=-1,out_type=type_r)
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleQPU(
                npoint=256,
                radius=0.4,
                nsample=32,
                mlp=nn.Sequential(
                    qpu_layers.QPU(4*8, 64*4),
                    post_process,
                    nn.Linear(post_process.outfeat(64*4), 128)
                ),
                use_xyz=use_xyz
            )
        )


        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=use_xyz)
        )

        self.FC_layer = (
            pt_utils.Seq(1024)
            .fc(512, bn=True)
            .dropout(0.5)
            .fc(256, bn=True)
            .dropout(0.5)
            .fc(num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        xyz -= xyz.reshape(-1,3).mean(0)
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        # type: (Pointnet2MSG, torch.cuda.FloatTensor) -> pt_utils.Seq
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.FC_layer(features.squeeze(-1))

if __name__ == "__main__":
    pass