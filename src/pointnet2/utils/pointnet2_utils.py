from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
from torch.autograd import Function
import torch.nn as nn
from . import etw_pytorch_utils as pt_utils
import sys
import numpy as np


import builtins


try:
    import pointnet2._ext as _ext
except ImportError:
    if not getattr(builtins, "__POINTNET2_SETUP__", False):
        raise ImportError(
            "Could not import _ext module.\n"
            "Please see the setup instructions in the README: "
            "https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/README.rst"
        )

if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class RandomDropout(nn.Module):
    def __init__(self, p=0.5, inplace=False):
        super(RandomDropout, self).__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, X):
        theta = torch.Tensor(1).uniform_(0, self.p)[0]
        return pt_utils.feature_dropout_no_scaling(X, theta, self.train, self.inplace)


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
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
        return _ext.furthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
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

        _, C, N = features.size()

        ctx.for_backwards = (idx, C, N)

        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards

        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None


gather_operation = GatherOperation.apply


class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
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
        dist2, idx = _ext.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None


three_nn = ThreeNN.apply


class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
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
        B, c, m = features.size()
        n = idx.size(1)

        ctx.three_interpolate_for_backward = (idx, weight, m)

        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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

        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )

        return grad_features, None, None


three_interpolate = ThreeInterpolate.apply


class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        # type: (Any, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""

        Parameters
        ----------
        features : torch.Tensor
            (B, C, N) tensor of features to group
        idx : torch.Tensor
            (B, npoint, nsample) tensor containing the indicies of features to group with

        Returns
        -------
        torch.Tensor
            (B, C, npoint, nsample) tensor
        """
        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()

        ctx.for_backwards = (idx, N)

        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.tensor) -> Tuple[torch.Tensor, torch.Tensor]
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

        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)

        return grad_features, None


grouping_operation = GroupingOperation.apply


class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
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
        return _ext.ball_query(new_xyz, xyz, radius, nsample)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery.apply



class QueryAndGroupRRI(nn.Module):
    r"""
    Groups with a ball query of radius for rigious rotation invariant

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample):
        # type: (QueryAndGroupRRI, float, int, bool) -> None
        super(QueryAndGroupRRI, self).__init__()
        self.radius, self.nsample = radius, nsample

    def forward(self, xyz, new_xyz, features=None):
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        new_xyz_reshaped = new_xyz.transpose(1, 2).unsqueeze(-1)
        #grouped_xyz = grouped_xyz - new_xyz_reshaped
        diff_grouped2grouped = grouped_xyz[:, :, :, :, None] - grouped_xyz[:, :, :, None, :]
        dis_grouped2grouped = torch.sqrt(torch.sum(diff_grouped2grouped * diff_grouped2grouped, dim=1))
        mean_dis_grouped2grouped = torch.mean(dis_grouped2grouped, dim=3)
        tip_idxs = torch.argmax(mean_dis_grouped2grouped, dim=2)
        tip_idxs_viewed = tip_idxs[:, None, :].repeat(1, grouped_xyz.shape[1], 1).view(grouped_xyz.shape[0]*grouped_xyz.shape[1]*grouped_xyz.shape[2])
        tip_pts = grouped_xyz.view(grouped_xyz.shape[0]*grouped_xyz.shape[1]*grouped_xyz.shape[2], grouped_xyz.shape[3])[range(tip_idxs_viewed.shape[0]), tip_idxs_viewed]
        tip_pts = tip_pts.view(grouped_xyz.shape[0], grouped_xyz.shape[1], grouped_xyz.shape[2])

        grouped_proj_vec = torch.cross(torch.cross(new_xyz_reshaped.repeat(1, 1, 1, grouped_xyz.shape[3]), grouped_xyz, dim=1),
                                       new_xyz_reshaped.repeat(1, 1, 1, grouped_xyz.shape[3]), dim=1)
        grouped_proj_vec = grouped_proj_vec / torch.norm(grouped_proj_vec, dim=1)[:, None, :, :]
        new_r = torch.sqrt(torch.sum(new_xyz_reshaped * new_xyz_reshaped, dim=1))
        new_norm = new_xyz_reshaped / (new_r[:, None, :, :] + 1e-8)

        tip_proj_vec = torch.cross(torch.cross(new_xyz_reshaped.squeeze(dim=3), tip_pts, dim=1), new_xyz_reshaped.squeeze(dim=3), dim=1)
        tip_proj_vec = tip_proj_vec / torch.norm(tip_proj_vec, dim=1)[:, None, :]
        grouped_cross_vec = torch.cross(grouped_proj_vec, tip_proj_vec[:, :, :, None].repeat(1, 1, 1, grouped_proj_vec.shape[3]), dim=1)
        grouped_proj_vec_sin = torch.sum(grouped_cross_vec * new_norm, dim=1)


        dis_grouped2grouped = torch.transpose(dis_grouped2grouped, 1, 2)
        dis_grouped2grouped_sort, _ = torch.sort(dis_grouped2grouped, dim=1)

        dis_grouped2grouped_sort = dis_grouped2grouped_sort * grouped_proj_vec_sin[:, None, :, :]

        grouped_r = torch.sqrt(torch.sum(grouped_xyz * grouped_xyz, dim=1))

        grouped_rri = torch.cat((dis_grouped2grouped_sort, grouped_r[:, None, :, :]), dim=1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            new_features = torch.cat(
                [grouped_rri, grouped_features], dim=1
            )  # (B, C + 4, npoint, nsample)
        else:
            new_features = grouped_rri

        return new_features


    '''
    backup
        def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroupRRI, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
      
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        new_xyz_reshaped = new_xyz.transpose(1, 2).unsqueeze(-1)

        grouped_r = torch.sqrt(torch.sum(grouped_xyz * grouped_xyz, dim=1))
        new_r = torch.sqrt(torch.sum(new_xyz_reshaped * new_xyz_reshaped, dim=1))
        grouped_norm = grouped_xyz / (grouped_r[:, None, :, :] + 1e-8)
        new_norm = new_xyz_reshaped / (new_r[:, None, :, :] + 1e-8)

        thetas = torch.acos(torch.clamp(torch.sum(grouped_norm * new_norm, dim=1), -1+1e-6, 1-1e-6))
        diff_grouped2grouped = grouped_xyz[:, :, :, :, None] - grouped_xyz[:, :, :, None, :]
        dis_grouped2grouped = torch.sqrt(torch.sum(diff_grouped2grouped * diff_grouped2grouped, dim=1))
        mean_dis_grouped2grouped = torch.mean(dis_grouped2grouped, dim=3)
        max_dis_grouped2grouped, _ = torch.max(dis_grouped2grouped, dim=3)
        std_dis_grouped2grouped = torch.std(dis_grouped2grouped, dim=3)




        
        grouped_proj_vec = torch.cross(torch.cross(new_xyz_reshaped.repeat(1, 1, 1, grouped_xyz.shape[3]), grouped_xyz, dim=1), new_xyz_reshaped.repeat(1, 1, 1, grouped_xyz.shape[3]), dim=1)
        grouped_proj_vec = grouped_proj_vec / torch.norm(grouped_proj_vec, dim=1)[:, None, :, :]
        grouped_proj_vec_sin = torch.cross(grouped_proj_vec[:, :, :, :, None].repeat(1, 1, 1, 1, grouped_proj_vec.shape[3]), grouped_proj_vec[:, :, :, None, :].repeat(1, 1, 1, grouped_proj_vec.shape[3], 1), dim=1)
        grouped_proj_vec_sin = torch.sum(grouped_proj_vec_sin * new_norm[:, :, :, :, None], dim=1)

        grouped_proj_vec_cos = torch.sum(grouped_proj_vec[:, :, :, :, None] * grouped_proj_vec[:, :, :, None, :], dim=1)
        grouped_proj_atan = torch.atan2(grouped_proj_vec_sin, grouped_proj_vec_cos)
        grouped_proj_atan[grouped_proj_atan < 0] = grouped_proj_atan[grouped_proj_atan < 0] + np.pi*2
       # grouped_proj_atan = torch.fmod(grouped_proj_atan + np.pi, np.pi)
        grouped_proj_atan, _ = torch.sort(grouped_proj_atan, dim=3)
        grouped_proj_atan = grouped_proj_atan[:, :, :, 1]

        grouped_rri = torch.cat((grouped_r[:, None, :, :], new_r.repeat(1, 1, grouped_r.shape[2])[:, None, :, :], thetas[:, None, :, :], grouped_proj_atan[:, None, :, :]), dim=1)
      




        if features is not None:
            grouped_features = grouping_operation(features, idx)
            new_features = torch.cat(
                [grouped_rri, grouped_features], dim=1
            )  # (B, C + 4, npoint, nsample)
        else:
            new_features = grouped_rri

        return new_features
    '''



class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(self, radius, nsample, use_xyz=True):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
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

        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert (
                self.use_xyz
            ), "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        return new_features


class GroupAll(nn.Module):
    r"""
    Groups all features

    Parameters
    ---------
    """

    def __init__(self, use_xyz=True):
        # type: (GroupAll, bool) -> None
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        # type: (GroupAll, torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor]
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
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
