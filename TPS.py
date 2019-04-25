import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import skimage
from skimage import io
import cv2
from PIL import ImageDraw
import thinplate as tps
def show_warped(img, warped):
    plt.figure(), plt.imshow(img)
    plt.figure(), plt.imshow(warped)
    # fig, axs = plt.subplots(1, 2, figsize=(16,8))
    # axs[0].axis('off')
    # axs[1].axis('off')
    #axs[0].imshow(img[...,::-1], origin='upper')
    # axs[0].scatter(c_src[:, 0]*img.shape[1], c_src[:, 1]*img.shape[0], marker='+', color='black')
    #axs[1].imshow(warped[...,::-1], origin='upper')
    # axs[1].scatter(c_dst[:, 0]*warped.shape[1], c_dst[:, 1]*warped.shape[0], marker='+', color='black')
    plt.show()


def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

def gridCellLine(image, cellNumber, boolRGB):
    r = 255 if boolRGB[0] else 0
    g = 255 if boolRGB[1] else 0
    b = 255 if boolRGB[2] else 0
    img = image.copy()
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    # img = Image.open('image.png')
    img_d = ImageDraw.Draw(img)
    x_len, y_len = img.size
    x_len_cell, y_len_cell = (x_len-1)/cellNumber, (y_len-1)/cellNumber
    for x in range(cellNumber+1):
        img_d.line(((x * x_len_cell, 0), (x * x_len_cell, y_len)), (r, g, b), width=1)
    for y in range(cellNumber+1):
        img_d.line(((0, y * y_len_cell), (x_len, y * y_len_cell)), (r, g, b), width=1)
    return np.array(img)

# img = skimage.io.imread('image.png')
# img = gridCellLine(img)
# plt.imshow(img)
# plt.show()
#
# c_src = np.array([
#     [0.0, 0.5],  # (col, row)
#     [0.5, 0],
#     [0.8, 1],
#     [0, 1],
#     [0.3, 0.3],
#     [0.7, 0.7],
# ])
#
# c_dst = np.array([
#     [0., 0.5],
#     [0.5, 0],
#     [0.9, 1],
#     [0.3, 1],
#     [0.4, 0.4],
#     [0.6, 0.6],
# ])
#
# warped = warp_image_cv(img, c_src, c_dst, dshape=(300, 512))
# show_warped(img, warped)