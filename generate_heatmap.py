import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
# import os
from os.path import join
from os.path import split

# -------------------
# SETTING
saliencymap_path = '/Users/july/workspace/ViNet/Result/kjp/images/*.png' # Saliency Mapのパス
image_path = '/Users/july/workspace/ViNet/Dataset/kjp/images/images/' # 画像のパス
color_alpha = 0.3 # overlayするカラー画像の透明度
# output_name = './test2.png'
output_path = '/Users/july/workspace/ViNet/Overlay/kjp/'
# -------------------

def color_saliencymap(saliencymap):
    """
    Saliency Mapに色をつけて可視化する。1を赤、0を青にする。 

    Parameters
    ----------------
    saliencymap : ndarray, np.uint8, (h, w) or (h, w, rgb)

    Returns
    ----------------
    saliencymap_colored : ndarray, np.uint8, (h, w, rgb)
    """
    assert saliencymap.dtype==np.uint8
    assert (saliencymap.ndim == 2) or (saliencymap.ndim == 3)

    saliencymap_colored = cv2.applyColorMap(saliencymap, cv2.COLORMAP_JET)[:, :, ::-1]

    return saliencymap_colored

def overlay_saliencymap_and_image(saliencymap_color, image):
    """
    Saliency Mapと画像を重ねる。

    Parameters
    ----------------
    saliencymap_color : ndarray, (h, w, rgb), np.uint8
    image : ndarray, (h, w, rgb), np.uint8

    Returns
    ----------------
    overlaid_image : ndarray(h, w, rgb)
    """
    assert saliencymap_color.ndim==3
    assert saliencymap_color.dtype==np.uint8
    assert image.ndim==3
    assert image.dtype==np.uint8
    im_size = (image.shape[1], image.shape[0])
    saliencymap_color = cv2.resize(saliencymap_color, im_size, interpolation=cv2.INTER_CUBIC)
    overlaid_image = cv2.addWeighted(src1=image, alpha=1, src2=saliencymap_color, beta=color_alpha, gamma=0)
    return overlaid_image

files = glob(saliencymap_path)

for file_name in files:
    image_name = split(file_name)[1]
    saliencymap = cv2.imread(file_name)[:, :, ::-1] # ndarray, (h, w, rgb), np.uint8 (0-255)
    saliencymap = saliencymap[:, :, 0] # ndarray, (h, w), np.uint8 (0-255)
    image = cv2.imread(join(image_path, image_name))[:, :, ::-1] # ndarray, (h, w, rgb), np.uint8 (0-255)

    saliencymap_colored = color_saliencymap(saliencymap) # ndarray, (h, w, rgb), np.uint8
    overlaid_image = overlay_saliencymap_and_image(saliencymap_colored, image) # ndarray, (h, w, rgb), np.uint8

    # 画像出力
    """
    ndarrayは(h,w,c)だが、そのままだとBGRの並びになっている。
    [:, :, ::-1]を末尾に入れてBGR>RGB反転をしてから出力する。
    """
    cv2.imwrite(join(output_path,image_name), overlaid_image[:, :, ::-1])