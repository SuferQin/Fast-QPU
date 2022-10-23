import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
import pytorch_utils as pt_utils
from typing import Tuple
import math
from _ext import pointnet2


class RandomDropout(nn.Module):

    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(
            X, theta, self.train, self.inplace
        )


class FurthestPointSampling(Function):

    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()

        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
        pointnet2.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output
        )
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor

        idx : torch.Tensor
            (B, npoint) tensor of the features to gather

        Returns
        -------
        torch.Tensor
            (B, C, npoint) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()

        output = torch.cuda.FloatTensor(B, C, npoint)

        pointnet2.gather_points_wrapper(
            B, C, N, npoint, features, idx, output
        )

        ctx.for_backwards = (idx, C, N)

        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())
        grad_out_data = grad_out.data.contiguous()
        pointnet2.gather_points_grad_wrapper(
            B, C, N, npoint, grad_out_data, idx, grad_features.data
        )

        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):

    @staticmethod
    def forward(ctx, unknown: torch.Tensor,
                known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of known features
        known : torch.Tensor
            (B, m, 3) tensor of unknown features

        Returns
        -------
        dist : torch.Tensor
            (B, n, 3) l2 distance to the three nearest neighbors
        idx : torch.Tensor
            (B, n, 3) index of 3 nearest neighbors
        """
        assert unknown.is_contiguous()
        assert known.is_contiguous()

        B, N, _ = unknown.size()
        m = known.size(1)
        dist2 = torch.cuda.FloatTensor(B, N, 3)
        idx = torch.cuda.IntTensor(B, N, 3)

        pointnet2.three_nn_wrapper(B, N, m, unknown, known, dist2, idx)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):

    @staticmethod
    def forward(
            ctx, features: torch.Tensor, idx: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            (B, c, m) Features descriptors to be interpolated from
        idx : torch.Tensor
            (B, n, 3) three nearest neighbors of the target features in features
        weight : torch.Tensor
            (B, n, 3) weights

        Returns
        -------
        torch.Tensor
            (B, c, n) tensor of the interpolated features
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()
        assert weight.is_contiguous()

        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        output = torch.cuda.FloatTensor(B, c, n)

        pointnet2.three_interpolate_wrapper(
            B, c, m, n, features, idx, weight, output
        )

        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, c, n) tensor with gradients of ouputs

        Returns
        -------
        grad_features : torch.Tensor
            (B, c, m) tensor with gradients of features

        None

        None
        """
        idx, weight, m = ctx.three_interpolate_for_backward
        B, c, n = grad_out.size()

        grad_features = Variable(torch.cuda.FloatTensor(B, c, m).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.three_interpolate_grad_wrapper(
            B, c, n, m, grad_out_data, idx, weight, grad_features.data
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of points to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of points to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample)

        pointnet2.group_points_wrapper(
            B, C, N, nfeatures, nsample, features, idx, output
        )

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx,
                 grad_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, npoint, nsample) tensor of the gradients of the output from forward

        Returns
        -------
        torch.Tensor
            (B, C, N) gradient of the features
        None
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = Variable(torch.cuda.FloatTensor(B, C, N).zero_())

        grad_out_data = grad_out.data.contiguous()
        pointnet2.group_points_grad_wrapper(
            B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data
        )

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):

    @staticmethod
    def forward(
            ctx, radius: float, nsample: int, xyz: torch.Tensor,
            new_xyz: torch.Tensor, fps_idx: torch.IntTensor
    ) -> torch.Tensor:
        r"""

        Parameters
        ----------
        radius : float
            radius of the balls
        nsample : int
            maximum number of features in the balls
        xyz : torch.Tensor
            (B, N, 3) xyz coordinates of the features
        new_xyz : torch.Tensor
            (B, npoint, 3) centers of the ball query

        Returns
        -------
        torch.Tensor
            (B, npoint, nsample) tensor with the indicies of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)
        idx = torch.cuda.IntTensor(B, npoint, nsample).zero_()

        pointnet2.ball_query_wrapper(
            B, N, npoint, radius, nsample, new_xyz, xyz, fps_idx, idx
        )
        
        return torch.cat([fps_idx.unsqueeze(2), idx], dim = 2)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of points to gather in the ball
    """

    def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
        super().__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None,
            fps_idx: torch.IntTensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz, fps_idx)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(
            xyz_trans, idx
        )  # (B, 3, npoint, nsample)
        raw_grouped_xyz = grouped_xyz
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([raw_grouped_xyz, grouped_xyz, grouped_features],
                                         dim=1)  # (B, C + 3 + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = torch.cat([raw_grouped_xyz, grouped_xyz], dim = 1)

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 3, 1, N) tensor
        """

        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features

### Code for quaternion representation

def normalize(v, dim):
    n = torch.sqrt(torch.sum(v*v, dim=dim, keepdim=True))
    return v / (n + 1e-6)


def project(p, v, dim):
    """
    project v to the plan orthognal to p
    p is normalized
    """
    p = normalize(p, dim)
    vert = torch.sum(p * v, dim=dim, keepdim=True) * p
    return v - vert
    

def project_one(p, dim):
    """Get one reference vector in the orthogonal projection plan.
       [1,0,0] - <[1,0,0],p>*p = [1-px*px, -px*py, -px*pz] or
       [0,1,0] - <[0,1,0],p>*p = [-py*px, 1-py*py, -py*pz] if p colinear with [1,0,0]
       Then normalize
    """
    p = normalize(p, dim)
    ref = torch.zeros_like(p)
    px = p.select(dim=dim, index=0)
    py = p.select(dim=dim, index=1)
    pz = p.select(dim=dim, index=2)
    colinear_x = torch.abs(px) > (1 - 1e-3)
    ref.select(dim, 0)[~colinear_x] = 1 - px[~colinear_x] * px[~colinear_x]
    ref.select(dim, 1)[~colinear_x] = - px[~colinear_x] * py[~colinear_x]
    ref.select(dim, 2)[~colinear_x] = - px[~colinear_x] * pz[~colinear_x]
    # if colinear
    ref.select(dim, 0)[colinear_x] = - py[colinear_x] * px[colinear_x]
    ref.select(dim, 1)[colinear_x] = 1 - py[colinear_x] * py[colinear_x]
    ref.select(dim, 2)[colinear_x] = - py[colinear_x] * pz[colinear_x]
    return normalize(ref, dim)
    

def rot_sort(p, pts, coord_dim, sample_dim, ref=None):
    """
    sort pts according to their orthogonal projection of p, 
    clockwise w.r.t one reference vector.
    """
    p = normalize(p, dim=coord_dim)
    
    if ref is None:
        ref = project_one(p, coord_dim)
    ref = ref.expand_as(pts)
    
    projs = normalize(project(p, pts, coord_dim), coord_dim)

    # Compute angles from ref to projs 
    sinus = torch.sum(torch.cross(ref, projs, coord_dim) * p, dim=coord_dim, keepdim=True)
    cosinus = torch.sum(ref * projs, dim=coord_dim, keepdim=True)
    angles = torch.atan2(sinus, cosinus)

    # If projection is too small, we randomly give an angle 
    # (because ref is not rotation-invariant)
    close_ind = torch.sum(projs * projs, dim=coord_dim, keepdim=True) < 1e-12
    angles[close_ind] = ((torch.rand(close_ind.sum()) - 0.5) * math.pi * 2).cuda()  # something error here [TODO]

    # Sort according to angles
    ind = angles.sort(dim=sample_dim)[1]
    pts = pts.gather(index=ind.expand_as(pts), dim=sample_dim)
    
    return pts


def to_quat(xyz, radius):
    """
    xyz: B, 3, N, S -> B, 4, N, S
    """
    dist = torch.sqrt(torch.sum(xyz * xyz, dim=1, keepdim=True))

    ori = xyz / (dist + 1e-6)
    theta = dist / radius * math.pi / 2.
    s = torch.cos(theta)
    v = torch.sin(theta) * ori
    q = torch.cat([s, v], dim=1)

    return q

def calc_invariance(center,relative_vector,dim):
    norm2 = torch.sqrt(torch.sum(relative_vector**2, dim=dim, keepdim=True)+1e-12)
    center = normalize(center,dim=dim)
    cross = (center*relative_vector).sum(dim, keepdim=True)/norm2
    angle = torch.acos(torch.clamp(cross, min=-1+1e-6, max=1-1e-6))
    return torch.cat([norm2,angle],dim=dim)


class QueryAndGroupQuat(nn.Module):
    r"""
    Groups with a ball query of radius, then convert neighbor points to quaternion, sorted by distance

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroupQuat, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.M = 8

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None,
            fps_idx: torch.IntTensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor : 
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 2 + C, npoint, nsample+1) tensor
        -------
        N : num of all points
        npoint: num of furthest sample
        nsample: num of nearest neighbor
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz, fps_idx)

        if self.use_xyz:
            xyz_trans = xyz.transpose(1, 2).contiguous()
            grouped_xyz = grouping_operation(xyz_trans, idx)            # (B, 3, npoint, nsample+1)
            new_xyz = new_xyz.transpose(1, 2).unsqueeze(-1)             # (B, 3, npoint, 1)
            if features is None:                                        # first: (B, 3, npoint, nsample) ordered
                grouped_xyz = rot_sort(p=new_xyz, pts=grouped_xyz[...,1:] - new_xyz, coord_dim=1, sample_dim=-1) 
            invariance_feat = calc_invariance(new_xyz,grouped_xyz,dim=1) # (B, 2, npoint, nsample[+1])

        if features is not None:
            new_features = grouping_operation(features, idx)  # (B, C, npoint, nsample+1)
            if self.use_xyz:
                new_features = torch.cat([invariance_feat, new_features], dim=1)  # (B, 2 + C , npoint, nsample+1)
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            B, _, npoint, nsample = grouped_xyz.shape
            grouped_quat = to_quat(grouped_xyz, self.radius).unsqueeze(2)  # B, 4, 1, npoint, nsample
            quat_features = [grouped_quat]
            # Cyclic permute and concat
            index = torch.tensor([x for x in range(1, nsample)] + [0]).cuda()
            for _ in range(self.M - 1):
                quat_features.append(quat_features[-1].index_select(dim=-1, index=index))
            quat_features = torch.cat(quat_features, dim=2)                         # B, 4, M, npoint, nsample
            quat_features = quat_features.reshape(B, 4 * self.M, npoint, nsample)   # B, 4*M, npoint, nsample
            new_features = torch.cat([invariance_feat, quat_features], dim = 1)
        return new_features

class GroupAllQuat(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz: bool = True):
        super().__init__()
        self.use_xyz = use_xyz

    def forward(
            self,
            xyz: torch.Tensor,
            new_xyz: torch.Tensor,
            features: torch.Tensor = None
    ) -> Tuple[torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor, is None
            Ignored
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, C + 2, 1, N) tensor
        """        
        # new_features = calc_invariance(xyz.mean(1,keepdim=True),xyz,2).transpose(1, 2).unsqueeze(2)
        new_features = (xyz*xyz).sum(2,keepdim=True).sqrt().transpose(1, 2).unsqueeze(2)
        if features is not None:
            grouped_features = features.unsqueeze(2)
            if self.use_xyz:
                new_features = torch.cat([new_features, grouped_features], dim=1)  # (B, 2 + C, 1, N)
            else:
                new_features = grouped_features

        return new_features