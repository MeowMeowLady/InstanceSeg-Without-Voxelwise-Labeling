# -*- coding: utf-8 -*-
"""
Created on 19-3-7 下午10:27
IDE PyCharm 

@author: Meng Dong

this script is used to binarize the nuclei detection results combining the
PRM results, which is finished with an improved Otsu algorithm.
"""

from otsu import otsu_py, otsu_py_2d, otsu_py_2d_fast
from skimage import io, morphology
import os
import numpy as np
from glob import glob
import utils.boxes_3d as box_utils_3d
from libtiff import TIFF
from scipy import ndimage
from cc3d import connected_components

prm_path = '' # the path for PRM results
save_path = '' # the path to save the final binarization results
if not os.path.exists(save_path):
    os.makedirs(save_path)
test_list = open('../test.txt', 'r').readlines() # the test image list is saved in a txt file
test_list = [item.rstrip() for item in test_list]
nms_thresh = 0.15
norm_side = 200
folder = '01' # select one track to process once
if folder == '01': # both tracks are saved in one list-txt
    test_list = test_list[:70]
elif folder == '02':
    test_list = test_list[70:]

for img_path in test_list:

    img = io.imread(img_path)

    track = img_path.split('/')[-2]
    img_name = track + '_' +img_path.split('/')[-1][:-4]

    img = ndimage.gaussian_filter(img, sigma=1)
    img = ndimage.median_filter(img, size=3)

    print('{}'.format(img_name))
    instance_idex = np.empty((0, 5), dtype=int)
    slices, height, width = img.shape
    overlap = 100
    patch_size = 200
    widx = list(range(0, width - patch_size, patch_size - overlap)) + [width - patch_size]
    hidx = list(range(0, height - patch_size, patch_size - overlap)) + [height - patch_size]
    len_h = len(hidx)
    len_w = len(widx)
    # read detection results and apply nms to them
    dets = np.empty((0, 7), dtype=np.float32)
    for ih, h in enumerate(hidx):
        for iw, w in enumerate(widx):
            num = ih * len_w + iw
            instance_path = os.path.join(prm_path, img_name, 'instances','{}'.format(num))
            n = len(glob(os.path.join(instance_path, '*.tif')))
            if n == 0:
                continue
            off_set = np.array([w, h, 0, w, h, 0, 0])
            cur_dets = np.load(os.path.join(instance_path, 'dets.npy'))

            assert cur_dets.shape[0] == n, 'the prm tiff number does not match the dets number'

            dets = np.concatenate((dets, off_set+cur_dets), axis=0)
            for i in range(n):
                instance_idex = np.concatenate((instance_idex, np.array([[num, i, w, h, 0]], dtype=int)), axis=0)

    # remove broken boxes at edges
    condition1 = (dets[:, 0] > 10) & (dets[:, 3] < width-10) & ((dets[:, 3] - dets[:, 0] + 1) < 32)
    condition2 = (dets[:, 1] > 10) & (dets[:, 4] < width - 10) & ((dets[:, 4] - dets[:, 1] + 1) < 32)
    keep = condition1 | condition2
    keep = ~keep
    dets = dets[keep, :].astype(np.float32)
    instance_idex = instance_idex[keep, :]
    # apply nms
    keep = box_utils_3d.nms_3d_volume(dets, nms_thresh)
    dets = dets[keep, :].copy()
    instance_idex = instance_idex[keep, :].copy()
    # keep score bigger than 0.4
    keep = dets[:, -1]>0.4
    dets = dets[keep, :].copy()
    instance_idex = instance_idex[keep, :].copy()

    id_det = np.zeros((0, 8), dtype=np.float32)
    seg = np.zeros(img.shape, dtype=np.uint16)
    mask_id = 0
    for d, det in enumerate(dets):
        mask_id += 1
        num, i, w, h, s = instance_idex[d, :]
        prm = io.imread(os.path.join(prm_path, img_name, 'instances', '{}/{}.tif'.format(num, i)))
        off_set = np.array([w, h, s, w, h, s, 0])
        det_prm = det - off_set
        x1, y1, z1, x2, y2, z2 = det_prm[:6].astype(int) # the locations at prm or copped image
        x1 = np.maximum(0, x1)
        y1 = np.maximum(0, y1)
        z1 = np.maximum(0, z1)
        x2 = np.minimum(norm_side-1, x2)
        y2 = np.minimum(norm_side-1, y2)
        z2 = np.minimum(slices-1, z2)
        crop_img = img[:, h: h+norm_side, w: w+norm_side].copy()
        crop_img_bi = seg[:, h: h+norm_side, w: w+norm_side].copy()
        box_img = crop_img[z1: z2+1, y1: y2+1, x1: x2+1].copy()
        box_prm = prm[z1: z2+1, y1: y2+1, x1: x2+1].copy()

        # normalize gray image and prm image into similar intensity range
        box_prm = box_prm.astype(float)
        gray_min = np.min(box_img)
        gray_max = np.max(box_img)
        gray_range = gray_max - gray_min + 1
        if gray_range < 400:
            box_img = box_img.astype(float)
            box_img = (box_img/gray_max*400).astype(np.uint16) + gray_min
        gray_min = np.min(box_img)
        gray_max = np.max(box_img)
        box_prm = np.round((box_prm-np.min(box_prm)) / (np.max(box_prm) - np.min(box_prm)) * (gray_max - gray_min) + gray_min).astype(
            np.uint16)

        # binarization using our improved otsu algorithm
        box_bi, k, b = otsu_py_2d_fast(box_img, box_prm)
        # find the largest connected component
        labels_out = connected_components(box_bi)
        segids = [x for x in np.unique(labels_out) if x!= 0]
        vol = [np.sum(labels_out==segid) for segid in segids]
        largestCC = labels_out==segids[np.argmax(vol)]
        box_bi = largestCC.copy()

        box_bi_not = (1-box_bi).astype(bool)
        labels_out = connected_components(box_bi_not)
        segids = [x for x in np.unique(labels_out) if x != 0]
        vol = [np.sum(labels_out == segid) for segid in segids]
        largestCC = labels_out == segids[np.argmax(vol)]
        largestCC = (1-largestCC).astype(bool)

        largestCC = morphology.binary_closing(largestCC)

        box_bi = largestCC.astype(np.uint16)*mask_id

        crop_img_bi_zeros = crop_img_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1] == 0
        crop_img_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1][crop_img_bi_zeros] = box_bi[crop_img_bi_zeros]
        seg[:, h: h+norm_side, w: w+norm_side] = crop_img_bi.copy()
        x1, y1, z1, x2, y2, z2 = [x1, y1, z1, x2, y2, z2] + off_set[:6]
        if mask_id in np.unique(crop_img_bi):
            id_det = np.concatenate((id_det, np.array([mask_id, x1, y1, z1, x2, y2, z2, det[-1]])[np.newaxis, :]), axis=0)
    # save both detection and segmentation results
    np.save(os.path.join(save_path, '{}.npy'.format(img_name)), id_det)
    image3D = TIFF.open(os.path.join(save_path, '{}.tif'.format(img_name)), mode='w')
    for k in range(seg.shape[0]):
        image3D.write_image(seg[k, :], compression='lzw', write_rgb=True)
    image3D.close()
