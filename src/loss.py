import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd.variable import Variable
from torch.nn import MSELoss

from src.utils import chamfer_distance, chamfer_distance_one_side


def spline_reconstruction_loss_one_sided(nu, nv, output, points, config, side=1):
    """
    Spline reconsutruction loss defined using chamfer distance, but one
    sided either gt surface can cover the prediction or otherwise, which
    is defined by the network. side=1 means prediction can cover gt.
    :param nu: spline basis function in u direction.
    :param nv: spline basis function in v direction.
    :param points: points sampled over the spline.
    :param config: object of configuration class for extra parameters.
    """
    reconst_points = []
    batch_size = output.shape[0]
    c_size_u = output.shape[1]
    c_size_v = output.shape[2]
    grid_size_u = nu.shape[0]
    grid_size_v = nv.shape[0]

    output = output.view(config.batch_size, config.grid_size, config.grid_size, 3)
    points = points.permute(0, 2, 1)
    for b in range(config.batch_size):
        point = []
        for i in range(3):
            point.append(torch.matmul(torch.matmul(nu, output[b, :, :, i]), torch.transpose(nv, 1, 0)))
        reconst_points.append(torch.stack(point, 2))

    reconst_points = torch.stack(reconst_points, 0)
    reconst_points = reconst_points.view(config.batch_size, grid_size_u * grid_size_v, 3)
    dist = chamfer_distance_one_side(reconst_points, points, side)
    return dist, reconst_points
