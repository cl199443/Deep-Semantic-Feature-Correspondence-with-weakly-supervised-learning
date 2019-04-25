from __future__ import print_function, division
import os
from os.path import exists, join, basename
import numpy as np
import scipy as sc
import scipy.misc
from collections import OrderedDict
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch
import torch.nn as nn
import cv2
from TPS import *
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset

from lib.model import ImMatchNet,MutualMatching
from lib.normalization import NormalizeImageDict
from lib.torch_util import save_checkpoint, str_to_bool
import xlrd, xlwt
from lib.point_tnf import corr_to_matches
from lib.eval_util import *
from lib.point_tnf import normalize_axis, unnormalize_axis, bilinearInterpPointTnf
from lib.plot import plot_image
from warp import *
# CUDA
use_cuda = torch.cuda.is_available()

###################################Load pretrained model###################################
name = 1
checkpoint = 'ourModel/250205LOSSbest_checkpoint_adam.pth.tar' if name == 1 else 'trained_models/ncnet_pfpascal.pth.tar'
model = ImMatchNet(use_cuda=use_cuda,
                   checkpoint=checkpoint)
# print(model) # show the net structure
model.FeatureExtraction.eval()
for param in model.NeighConsensus.parameters():
    param.requires_grad = False
print('Done')

##############################################Dataset#########################################
from lib.pf_dataset import PFPascalDataset
Dataset = PFPascalDataset

eval_dataset_path = 'datasets/pf-pascal/'
csv_file = 'image_pairs/test_pairs.csv'
xlsx_file = eval_dataset_path + 'image_pairs/test_pairs.xlsx'
image_size = 250  # 400
feature_size = int(image_size*0.0625)

dataset = Dataset(csv_file=os.path.join(eval_dataset_path, csv_file),
                  dataset_path=eval_dataset_path,
                  transform=NormalizeImageDict(['source_image', 'target_image']),
                  output_size=(image_size, image_size))


xlfile = xlrd.open_workbook(xlsx_file)
sheet_name = xlfile.sheet_names()[0]
sheet1 = xlfile.sheet_by_name(sheet_name)
rows, cols = sheet1.nrows, sheet1.ncols
#######################################Evaluate model#####################################
# draw sample
while 1:
    idx = np.random.randint(len(dataset)) # 184
    print(idx)
    sample = dataset[idx]
    src = sample['source_image']  # (3, image_size, image_size)
    tgt = sample['target_image']
    tgt_pts = Variable(sample['target_points'], requires_grad=False)  # (row:2, col:20)
    src_pts = Variable(sample['source_points'], requires_grad=False)  # (row:2, col:20)

    imgSrc, imgDst, imgTotal = toSolveShape(sheet1, idx+1, eval_dataset_path)  # notice the index in excel !!!!!!!
    print(imgDst.shape)

    # ground truth annotations
    valid_pts = tgt_pts[0, :] != -1  # 标记出存储有标记特征点的列
    tgt_pts = torch.cat((tgt_pts[0, :][valid_pts].view(1, 1, -1), tgt_pts[1, :][valid_pts].view(1, 1, -1)), 1)  # (1, 2, annotations number)
    tgt_pts[0, 0, :] = normalize_axis(tgt_pts[0, 0, :], sample['target_im_size'][1])  # col
    tgt_pts[0, 1, :] = normalize_axis(tgt_pts[0, 1, :], sample['target_im_size'][0])  # row
    # normalize the ground truth feature points
    valid_pts = src_pts[0, :] != -1
    src_pts = torch.cat((src_pts[0, :][valid_pts].view(1, 1, -1), src_pts[1, :][valid_pts].view(1, 1, -1)), 1)
    src_pts[0, 0, :] = normalize_axis(src_pts[0, 0, :], sample['source_im_size'][1])
    src_pts[0, 1, :] = normalize_axis(src_pts[0, 1, :], sample['source_im_size'][0])

    # evaluate model on image pair
    batch_tnf = {'source_image': Variable(src.unsqueeze(0)), 'target_image': Variable(tgt.unsqueeze(0))}

    if use_cuda:
        src_pts = src_pts.cuda()
        tgt_pts = tgt_pts.cuda()
        batch_tnf['source_image'] = batch_tnf['source_image'].cuda()  # torch.size([1, 3, image_size, image_size])
        batch_tnf['target_image'] = batch_tnf['target_image'].cuda()

    corr4d = model(batch_tnf)  # torch.size([1, 1, 16, 16, 16, 16])

    # compute matches from output
    c2m = corr_to_matches
    xA, yA, xB, yB, sB = c2m(corr4d, do_softmax=True)
    warped_points = bilinearInterpPointTnf((xA, yA, xB, yB), tgt_pts)  # B中特征点warp到A中的coordinate位置(-1~1)

    ###############################################MatchingGridCellPoints##############################################
    colA = unnormalize_axis(xA, sample['source_im_size'][1])
    rowA = unnormalize_axis(yA, sample['source_im_size'][0])
    colB = unnormalize_axis(xB, sample['target_im_size'][1])
    rowB = unnormalize_axis(yB, sample['target_im_size'][0])
    plt.imshow(imgTotal)  # tar + src
    beginScore, endScore = torch.min(sB), torch.max(sB)
    flag = endScore - 0.0001  # - 0.005 * (endScore-beginScore)
    for i in range(colA.shape[1]):
        a = sB[0][i].data > flag.data
        if int(a.cpu()):
            xa = float(colB[0][i])
            ya = float(rowB[0][i])
            xb = float(colA[0][i]) + imgDst.shape[1]
            yb = float(rowA[0][i])
            c = np.random.rand(3)
            # plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            # plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            # plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5)

    # plt.axis('off')
    # plt.title('gridCellMatching')
    # plt.savefig('cell.png', dpi=300)
    # plt.show()

    ###############################################MatchingGridCellPoints##############################################


    # display result
    im = plot_image(torch.cat((batch_tnf['target_image'], batch_tnf['source_image']), 3), 0, return_im=True)
    tgt_pts[0, 0, :] = unnormalize_axis(tgt_pts[0, 0, :], tgt.shape[2])  # B points
    tgt_pts[0, 1, :] = unnormalize_axis(tgt_pts[0, 1, :], tgt.shape[1])
    src_pts[0, 0, :] = unnormalize_axis(src_pts[0, 0, :], src.shape[2])  # A points
    src_pts[0, 1, :] = unnormalize_axis(src_pts[0, 1, :], src.shape[1])
    warped_points[0, 0, :] = unnormalize_axis(warped_points[0, 0, :], src.shape[2])
    warped_points[0, 1, :] = unnormalize_axis(warped_points[0, 1, :], src.shape[1])  # 变形后的点坐标反归一化
    # cat image(B+A:tar+src)  # tranform B points to A
    # plt.imshow(im)
    # for i in range(tgt_pts.shape[2]):
    #     xa = float(tgt_pts[0, 0, i])
    #     ya = float(tgt_pts[0, 1, i])
    #     xb = float(warped_points[0, 0, i]) + tgt.shape[2]
    #     yb = float(warped_points[0, 1, i])
    #     c = np.random.rand(3)
    #     plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))  # 5
    #     plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))  # 5
    #     plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5)
    # plt.axis('on')
    # plt.show()

##########################################From250ToSourceSize#################################
    assert tgt_pts.shape[2] == warped_points.shape[2]
    for index in range(tgt_pts.shape[2]):
        # print(tgt_pts[0, 0, index], tgt_pts[0, 0, index] * imgDst.shape[1] / image_size)
        tgt_pts[0, 0, index] = tgt_pts[0, 0, index] * imgDst.shape[1] / image_size
        tgt_pts[0, 1, index] = tgt_pts[0, 1, index] * imgDst.shape[0] / image_size
        src_pts[0, 0, index] = src_pts[0, 0, index] * imgSrc.shape[1] / image_size
        src_pts[0, 1, index] = src_pts[0, 1, index] * imgSrc.shape[0] / image_size
        warped_points[0, 0, index] = warped_points[0, 0, index] * imgSrc.shape[1] / image_size
        warped_points[0, 1, index] = warped_points[0, 1, index] * imgSrc.shape[0] / image_size
##########################################From250ToSourceSize#################################



######################################showMatching####################################
    plt.imshow(imgTotal)
    for i in range(tgt_pts.shape[2]):
        xa = float(tgt_pts[0, 0, i])  # col
        ya = float(tgt_pts[0, 1, i])  # row
        xb = float(warped_points[0, 0, i]) + imgDst.shape[1]  # tgt.shape[2]
        yb = float(warped_points[0, 1, i])
        c = np.random.rand(3)
        plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))  # 5
        plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))  # 5
        plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5)
    plt.axis('off')
    # plt.savefig('real'+ str(idx) +'.png', dpi=300)
    plt.show()

    # plt.imshow(imgTotal)
    # for i in range(tgt_pts.shape[2]):
    #     xa = float(tgt_pts[0, 0, i])  # col
    #     ya = float(tgt_pts[0, 1, i])  # row
    #     xb = float(src_pts[0, 0, i]) + imgDst.shape[1]  # tgt.shape[2]
    #     yb = float(src_pts[0, 1, i])
    #     c = np.random.rand(3)
    #     plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))  # 5
    #     plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))  # 5
    #     plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.5)
    # plt.axis('off')
    # plt.title('groundTruth')
    # plt.savefig('truth.png', dpi=300)
    # plt.show()

    c_src, c_dst, c_dstWarped = src_pts.data.cpu().numpy(), tgt_pts.data.cpu().numpy(), warped_points.data.cpu().numpy()
    c_src, c_dst, c_dstWarped = np.transpose(c_src.squeeze()), np.transpose(c_dst.squeeze()), np.transpose(c_dstWarped.squeeze())
    c_dstWarped[:, 0], c_dstWarped[:, 1] = c_dstWarped[:, 0]/imgSrc.shape[1], c_dstWarped[:, 1]/imgSrc.shape[0]
    c_src[:, 0], c_src[:, 1] = c_src[:, 0] / imgSrc.shape[1], c_src[:, 1] / imgSrc.shape[0]
    c_dst[:, 0], c_dst[:, 1] = c_dst[:, 0] / imgDst.shape[1], c_dst[:, 1] / imgDst.shape[0]
    # warpedSD = warp_image_cv(gridCellLine(imgSrc, 16, (0, 1, 0)), c_dstWarped, c_dst, dshape=(imgDst.shape[0], imgDst.shape[1]))  # dshape : (row, col)
    # warpedDS = warp_image_cv(gridCellLine(imgDst, 16, (0, 1, 0)), c_dst, c_dstWarped, dshape=(imgSrc.shape[0], imgSrc.shape[1]))  # dshape : (row, col)
    warpedSD = warp_image_cv(imgSrc, c_dstWarped, c_dst,
                             dshape=(imgDst.shape[0], imgDst.shape[1]))  # dshape : (row, col)
    warpedDS = warp_image_cv(imgDst, c_dst, c_dstWarped,
                             dshape=(imgSrc.shape[0], imgSrc.shape[1]))  # dshape : (row, col)
    plt.figure(), plt.imshow(warpedSD), plt.axis('off'), plt.title('TPS')#, plt.savefig('TPS1_' + str(idx)+'.png', dpi=300)
    plt.figure(), plt.imshow(warpedDS), plt.axis('off'), plt.title('TPS')#, plt.savefig('TPS2_' + str(idx)+'.png', dpi=300)
    plt.show()
######################################showMatching####################################



    ########################################## WARP ################################################
    assert tgt_pts.size(2) == warped_points.size(2)
    matcher_A, matcher_B = torch.ones((3, tgt_pts.size(2))), torch.ones((3, tgt_pts.size(2)))
    matcher_A = matcher_A.type(torch.cuda.FloatTensor)
    matcher_B = matcher_B.type(torch.cuda.FloatTensor)
    for item in range(tgt_pts.size(2)):
        matcher_A[0][item] = tgt_pts.data[0][0][item]  # col
        matcher_A[1][item] = tgt_pts.data[0][1][item]  # row
        matcher_B[0][item] = warped_points.data[0][0][item]
        matcher_B[1][item] = warped_points.data[0][1][item]

    rightMatrix = toSolveFullMatrix(matcher_A, matcher_B, tgt_pts.size(2))
    # srcImage = plot_image(batch_tnf['source_image'], 0, return_im=True)
    # dstImage = plot_image(batch_tnf['target_image'], 0, return_im=True)
    warpedAffine0 = cv2.warpPerspective(imgDst, rightMatrix, (imgSrc.shape[1], imgSrc.shape[0]))
    warpedAffine1 = cv2.warpPerspective(imgSrc, np.linalg.inv(rightMatrix), (imgDst.shape[1], imgDst.shape[0]))  # newWarpAffine(img1, img2, Hom)
    # plt.figure(), plt.axis('off'), plt.imshow(warpedAffine0)
    # plt.figure(), plt.axis('off'), plt.imshow(warpedAffine1)
    # # plt.savefig('plot123_2.png', dpi=300)
    # plt.show()
    ########################################## WARP ################################################