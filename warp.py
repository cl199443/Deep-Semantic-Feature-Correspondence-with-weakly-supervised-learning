from __future__ import print_function, division
from scipy.optimize import fsolve,leastsq

import os
from os.path import exists, join, basename
from collections import OrderedDict
import numpy as np
import numpy.random
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from lib.dataloader import DataLoader # modified dataloader
from lib.model import ImMatchNet
from lib.im_pair_dataset import ImagePairDataset
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint, str_to_bool
from lib.torch_util import BatchTensorToVars, str_to_bool

def toSolveFullMatrix(point_des, point_src, npoints): # right, left
    ## solve the matrix from des to src (right to left)
    point_des, point_src = point_des.cpu().numpy(), point_src.cpu().numpy()
    x_des = point_des[0, :].T  # col
    y_des = point_des[1, :].T  # row
    x_src = point_src[0, :].T
    y_src = point_src[1, :].T

    A = np.zeros((npoints * 2, 8))  # 2N * 8
    B = np.ones((npoints * 2, 1))

    for i in range(npoints):
        A[2 * i][0] = x_des[i]
        A[2 * i][1] = y_des[i]
        A[2 * i][2] = 1
        A[2 * i][6] = -x_des[i] * x_src[i]
        A[2 * i][7] = -y_des[i] * x_src[i]
        B[2 * i][0] = x_src[i]

        A[2 * i + 1][3] = x_des[i]
        A[2 * i + 1][4] = y_des[i]
        A[2 * i + 1][5] = 1
        A[2 * i + 1][6] = -x_des[i] * y_src[i]
        A[2 * i + 1][7] = -y_des[i] * y_src[i]
        B[2 * i + 1][0] = y_src[i]

    flag = np.linalg.inv(np.dot(A.T, A))
    flag1 = np.dot(flag, A.T)
    h = np.dot(flag1, B)
    ans = np.ones((3, 3))
    ans[0][0] = h[0][0]
    ans[0][1] = h[1][0]
    ans[0][2] = h[2][0]
    ans[1][0] = h[3][0]
    ans[1][1] = h[4][0]
    ans[1][2] = h[5][0]
    ans[2][0] = h[6][0]
    ans[2][1] = h[7][0]
    ans[2][2] = 1

    return ans