import torch
import numpy as np


def knn_with_batch(p1, p2, k, is_max=False):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :param is_max: k-nearest neighbors or k-farthest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    top_dist, k_nn = torch.topk(dist, k, dim=2, largest=is_max)

    return k_nn


def distance_map_with_batch(p1, p2):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :return: for each point in p1, returns the distances to all the points in p2; size[B,N,M]
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    return dist


def closest_distance_with_batch(p1, p2, is_sum=True):
    '''
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice


def point2sphere_distance_with_batch(p1, p2):
    '''
    :param p1: size[B,N,3]
    :param p2: size[B,M,4]
    :return: the distances from p1 to the closest spheres in p2
    '''
    assert p1.size(0) == p2.size(0) and p1.size(2) == 3 and p2.size(2) == 4

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1 = p1.transpose(1, 2)
    p2 = p2.repeat(1, p1.size(1), 1, 1)
    p2_xyzr = p2
    p2 = p2_xyzr[:, :, :, 0:3]
    p2_r = p2_xyzr[:, :, :, 3]

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    min_indice = torch.unsqueeze(min_indice, 2)
    min_dist = torch.unsqueeze(min_dist, 2)

    p2_min_r = torch.gather(p2_r, 2, min_indice)
    min_dist = min_dist - p2_min_r
    min_dist = torch.norm(min_dist, 2, dim=2)

    dist_scalar = torch.sum(min_dist)

    return dist_scalar


def sphere2point_distance_with_batch(p1, p2):
    '''
    :param p1: size[B,N,4]
    :param p2: size[B,M,3]
    :return: the distances from sphere p1 to the closest points in p2
    '''

    assert p1.size(0) == p2.size(0) and p1.size(2) == 4 and p2.size(2) == 3

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1_r = p1[:, :, :, 3]
    p1 = p1.repeat(1, p2.size(2), 1, 1)
    p1_r = p1_r.transpose(1, 2)

    p1 = p1.transpose(1, 2)
    p1_xyzr = p1
    p1 = p1_xyzr[:, :, :, 0:3]

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=3)

    min_dist, min_indice = torch.min(dist, dim=2)
    min_dist = torch.unsqueeze(min_dist, 2)
    min_dist = min_dist - p1_r
    min_dist = torch.norm(min_dist, 2, dim=2)

    dist_scalar = torch.sum(min_dist)

    return dist_scalar


def closest_distance_np(p1, p2, is_sum=True):
    '''
    :param p1: size[N, D], numpy array
    :param p2: size[M, D], numpy array
    :param is_sum: whehter to return the summed scalar or the separate distances with indices
    :return: the distances from p1 to the closest points in p2
    '''

    p1 = torch.from_numpy(p1[None, :, :]).double()
    p2 = torch.from_numpy(p2[None, :, :]).double()

    assert p1.size(0) == 1 and p2.size(0) == 1
    assert p1.size(2) == p2.size(2)

    p1 = p1.repeat(p2.size(1), 1, 1)
    p1 = p1.transpose(0, 1)

    p2 = p2.repeat(p1.size(0), 1, 1)

    dist = torch.add(p1, torch.neg(p2))
    dist = torch.norm(dist, 2, dim=2)
    min_dist, min_indice = torch.min(dist, dim=1)
    dist_scalar = torch.sum(min_dist)

    if is_sum:
        return dist_scalar
    else:
        return min_dist, min_indice
