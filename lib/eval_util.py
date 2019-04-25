import torch
import torch.nn
import numpy as np
import os
from skimage import draw
import torch.nn.functional as F
from torch.autograd import Variable
from lib.pf_dataset import PFPascalDataset
from lib.point_tnf import PointsToUnitCoords, PointsToPixelCoords, bilinearInterpPointTnf
from lib.point_tnf import *
import skimage
from skimage import io
from PIL import Image
from numpy import *
from numba import jit

def singleToThree(a):
    image = np.expand_dims(a, axis=2)
    image = np.concatenate((image, image, image), axis=-1)
    return image

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    flagImg = Image.fromarray(np.uint8(img))
    flagImg = flagImg.resize((224, 224))
    flagImg = np.asarray(flagImg)
    resized_img = flagImg / 255.0

    if len(resized_img.shape) == 2:
        resized_img = singleToThree(resized_img)
        img = singleToThree(img)  # transfer single Channel to three channels

    return img
def appendImage(imLeft, imRight):
    # select the image with the fewest rows and fill in enough empty rows
    # rowsNew = max(imLeft.shape[0], imRight.shape[0])
    # colNew = max(imLeft.shape[1], imRight.shape[1])

    rows1 = imLeft.shape[0]
    rows2 = imRight.shape[0]

    if len(imLeft.shape) == 3:
        assert imLeft.shape[2] == imRight.shape[2]
        depth = imLeft.shape[2]
        if rows1 < rows2:
            pad = ones((rows2-rows1, imLeft.shape[1], depth))*255
            imLeft = concatenate([imLeft, pad], axis=0).astype(uint8)  # 0为行扩展
        elif rows1 > rows2:
            pad = ones((rows1 - rows2, imRight.shape[1], depth))*255
            imRight = concatenate([imRight, pad], axis=0).astype(uint8)  # 数据类型转换要用arrayName.astype(dtype)
    elif len(imLeft.shape) == 2:
        if rows1 < rows2:
            pad = ones((rows2 - rows1, imLeft.shape[1])) * 255
            imLeft = concatenate([imLeft, pad], axis=0).astype(uint8)  # 0为行扩展
        elif rows1 > rows2:
            pad = ones((rows1 - rows2, imRight.shape[1])) * 255
            imRight = concatenate([imRight, pad], axis=0).astype(uint8)  # 数据类型转换要用arrayName.astype(dtype)

    imFinal = concatenate((imLeft, imRight), axis=1).astype(uint8)  # 列扩展
    # plt.imshow(imFinal)
    # plt.axis('off')
    # plt.show()
    return imFinal

def toSolveShape(sheet1, idx, path, mask = 0):
    src = sheet1.cell_value(idx, 0)
    dst = sheet1.cell_value(idx, 1)
    XA = sheet1.cell_value(idx, 3).split(';')
    YA = sheet1.cell_value(idx, 4).split(';')
    XB = sheet1.cell_value(idx, 5).split(';')
    YB = sheet1.cell_value(idx, 6).split(';')

    num = len(XA)
    matcher_A = np.ones((3, num))
    matcher_B = np.ones((3, len(XB)))

    for number in range(0, num):
        matcher_A[0][number] = float(XA[number])  # col-y
        matcher_A[1][number] = float(YA[number])  # row-x
    for number in range(len(XB)):
        matcher_B[0][number] = float(XB[number])
        matcher_B[1][number] = float(YB[number])

    img1Source = load_image(path + src)
    img2Source = load_image(path + dst)
    imgTotal = appendImage(img2Source, img1Source)

    if mask == 1:
        maskA = maskBool(matcher_A[1], matcher_A[0], img1Source)
        maskB = maskBool(matcher_B[1], matcher_B[0], img2Source)
        return img1Source, img2Source, imgTotal, maskA, maskB

    return img1Source, img2Source, imgTotal

def maskBool(rowCoor, colCool, img1Source):
    fill_row_coords, fill_col_coords = draw.polygon(rowCoor, colCool, (img1Source.shape[0], img1Source.shape[1]))  # (row, col)
    mask = np.zeros((img1Source.shape[0], img1Source.shape[1]), dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask + 0

def th_sampling_grid_to_np_flow(source_grid,h_src,w_src):
    # remove batch dimension
    source_grid = source_grid.squeeze(0)
    # get mask
    in_bound_mask=(source_grid.data[:,:,0]>-1) & (source_grid.data[:,:,0]<1) & (source_grid.data[:,:,1]>-1) & (source_grid.data[:,:,1]<1)
    in_bound_mask=in_bound_mask.cpu().numpy()
    # convert coords
    h_tgt,w_tgt=source_grid.size(0),source_grid.size(1)
    source_x_norm=source_grid[:,:,0]
    source_y_norm=source_grid[:,:,1]
    source_x=unnormalize_axis(source_x_norm,w_src)
    source_y=unnormalize_axis(source_y_norm,h_src)
    source_x=source_x.data.cpu().numpy()
    source_y=source_y.data.cpu().numpy()
    grid_x,grid_y = np.meshgrid(range(1,w_tgt+1),range(1,h_tgt+1))
    disp_x=source_x-grid_x
    disp_y=source_y-grid_y
    # apply mask
    disp_x = disp_x*in_bound_mask+1e10*(1-in_bound_mask)
    disp_y = disp_y*in_bound_mask+1e10*(1-in_bound_mask)
    flow = np.concatenate((np.expand_dims(disp_x,2),np.expand_dims(disp_y,2)),2)
    return flow

def pck(source_points,warped_points,L_pck,alpha=0.1):
    # compute precentage of correct keypoints
    batch_size=source_points.size(0)
    pck=torch.zeros((batch_size))
    for i in range(batch_size):
        p_src = source_points[i,:]
        p_wrp = warped_points[i,:]
        N_pts = torch.sum(torch.ne(p_src[0,:],-1)*torch.ne(p_src[1,:],-1))
        point_distance = torch.pow(torch.sum(torch.pow(p_src[:,:N_pts]-p_wrp[:,:N_pts],2),0),0.5)
        L_pck_mat = L_pck[i].expand_as(point_distance)
        correct_points = torch.le(point_distance,L_pck_mat*alpha)
        pck[i]=torch.mean(correct_points.float())
    return pck


def pck_metric(batch,batch_start_idx,matches,stats,args,use_cuda=True, alpha=0.1):
       
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)

    # compute points stage 1 only
    warped_points_norm = bilinearInterpPointTnf(matches,target_points_norm)  # B中特征点warp到A中的coordnate位置i(-1~1)
    warped_points = PointsToPixelCoords(warped_points_norm,source_im_size)  # warp后的坐标反标准化
    
    L_pck = batch['L_pck'].data  # for scNet rule, the L_pck is 224
    
    current_batch_size=batch['source_im_size'].size(0)
    indices = range(batch_start_idx,batch_start_idx+current_batch_size)

    # compute PCK
    pck_batch = pck(source_points.data, warped_points.data, L_pck, alpha)
    stats['point_tnf']['pck'][indices] = pck_batch.unsqueeze(1).cpu().numpy()
        
    return stats

def intersectionOverUnion(wapredMask, mask):
    andMask = wapredMask.copy().astype('uint8') & mask
    intersection = np.sum(andMask)
    union = np.sum(wapredMask.copy().astype('uint8') | mask)
    IoU = intersection / union
    return IoU

def label_transfer_accuracy(wapredMask, mask):
    labelMask = (wapredMask == mask) + 0
    LT_ACC = np.mean(labelMask)
    return LT_ACC

@jit
def localization_error(wapredMask, mask, matcher_A, matcher_B, matrix):
    XoA, YoA = int(matcher_A[0].min()), int(matcher_A[1].min())
    XoB, YoB = int(matcher_B[0].min()), int(matcher_B[1].min())
    wA, hA = int(matcher_A[0].max() - matcher_A[0].min() + 1), int(matcher_A[1].max() - matcher_A[1].min() + 1)
    wB, hB = int(matcher_B[0].max() - matcher_B[0].min() + 1), int(matcher_B[1].max() - matcher_B[1].min() + 1)

    V = 0  # the total points in edge of source image
    error = 0

    for i in range(XoA, XoA + wA):  # col
        for j in range(YoA, YoA + hA):  # row
            iWarp = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2]
            jWarp = matrix[1][0] * i + matrix[1][1] * j + matrix[1][2]
            delWarp = matrix[2][0] * i + matrix[2][1] * j + matrix[2][2]
            iWarp = int(iWarp / delWarp)
            jWarp = int(jWarp / delWarp)
            if iWarp < 0 or jWarp < 0 or iWarp > mask.shape[1] or jWarp > mask.shape[0]:
                continue
            if iWarp >= XoB and iWarp <= matcher_B[0].max() and jWarp >= YoB and jWarp <= matcher_B[1].max():
                V += 1
                xA, yA = (i - XoA) / wA, (j - YoA) / hA  # the ratio in imageA
                xWarp, yWarp = (iWarp - XoB) / wB, (jWarp - YoB) / hB
                error += (abs(xA - xWarp) + abs(yA - yWarp))

    LOC_ERR = error / V if V else -1  # filter out the outliers
    return LOC_ERR