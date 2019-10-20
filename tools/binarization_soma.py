# -*- coding: utf-8 -*-
"""
Created on 18-12-17 下午3:06
IDE PyCharm 

@author: Meng Dong

this script is used to binarize the soma detection results combining the
PRM results, which is finished with an improved Otsu algorithm.
"""

import numpy as np
from skimage import io
import os
from libtiff import TIFF
from skimage.measure import label
import utils.boxes_3d as box_utils_3d
from glob import glob
from otsu import otsu_py, otsu_py_2d_fast

prm_path = '' # the path for PRM results
save_path = ''# path for the path to save the final binarization results

if not os.path.exists(save_path):
    os.makedirs(save_path)
imglist = open('../test.txt', 'r').readlines()# the test image list is saved in a txt file
data_path = ''# the path for soma images
imglist = [x.rstrip() for x in imglist]
nms_thresh = 0.23
for im_name in imglist:
    print('img', im_name)
    img = io.imread(os.path.join(data_path, im_name, im_name+'.tif'))
    seg = np.zeros(img.shape, dtype=np.uint16)

    mask_id = 0
    dets = np.empty((0, 7), dtype=np.float32)
    instance_idex = np.empty((0, 5), dtype=int)
    # collect all the detected boxes and apply nms to them
    for s in range(3):
        for h in range(2):
            for w in range(2):
                num = s*4+h*2+w
                sub_root_path = os.path.join(prm_path, im_name, 'instances', '{}'.format(num))
                n = len(glob(os.path.join(sub_root_path, '*.tif')))
                if n == 0:
                    tmp_seg = None
                    continue
                ws = w*96
                hs = h*96
                ss = s*32
                off_set = np.array([ws, hs, ss, ws, hs, ss, 0], dtype=np.float32)
                dets = np.concatenate((dets, np.load(os.path.join(sub_root_path, 'dets.npy'))+off_set), axis=0).astype(np.float32)
                for i in range(n):
                    instance_idex = np.concatenate((instance_idex, np.array([[num, i, ws, hs, ss]], dtype=int)), axis=0)

    # apply nms
    keep = box_utils_3d.nms_3d(dets, nms_thresh)
    dets = dets[keep, :].copy()
    instance_idex = instance_idex[keep, :].copy()
    sort_idex = dets[:, -1].argsort()[::-1]
    dets = dets[sort_idex, :]
    instance_idex = instance_idex[sort_idex, :]

    scores = np.zeros((0, 2), dtype=np.float32)

    for d, (det, instance_id) in enumerate(zip(dets, instance_idex)):
        mask_id += 1
        sub_root_path = os.path.join(prm_path, im_name, 'instances', '{}'.format(instance_id[0]))
        prm_instance_im = io.imread(os.path.join(sub_root_path, '{}.tif'.format(instance_id[1])))
        ws, hs, ss = instance_id[2:]
        img_crop = img[ss: ss + 64, hs: hs + 160, ws: ws + 160].copy()
        img_crop_bi = seg[ss: ss + 64, hs: hs + 160, ws: ws + 160].copy()

        pos_voxel = prm_instance_im[prm_instance_im > 0]
        if len(pos_voxel) == 0:
            continue

        [x1, y1, z1, x2, y2, z2] = (det[:6] - np.array([ws, hs, ss, ws, hs, ss])).astype(int)

        # crop box-part image and prm
        box_prm = prm_instance_im[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1]
        box_img = img_crop[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1]

        # normalize gray image and prm image into similar intensity range
        gray_max = np.max(box_img)
        box_img_float = box_img.astype(float)
        box_img_float = np.clip(box_img_float/gray_max*300, 0, 300)+30
        box_img = box_img_float.astype(np.uint16)

        box_prm = box_prm.astype(float)
        box_prm = np.round(box_prm / np.max(box_prm) * 300 + 30).astype(np.uint16)

        # binarization using our improved otsu algorithm
        box_bi, _, _ = otsu_py_2d_fast(box_img, box_prm)

        # keep the biggest connected component
        labels = label(box_bi)
        largestCC = labels == (np.argsort(np.bincount(labels.flat)[1:])[-1] + 1)
        cc = largestCC.astype(np.uint16) * mask_id
        box_img_crop_bi_zeros = img_crop_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1] == 0
        img_crop_bi[z1: z2 + 1, y1: y2 + 1, x1: x2 + 1][box_img_crop_bi_zeros] = cc[box_img_crop_bi_zeros]
        seg[ss: ss + 64, hs: hs + 160, ws: ws + 160] = img_crop_bi.copy()
        if mask_id in np.unique(seg):
            scores = np.concatenate((scores, np.array([mask_id, det[-1]])[np.newaxis, :]), axis=0)
    # save detection and segmentation results
    np.save(os.path.join(save_path, '{}.npy'.format(im_name)), scores)
    image3D = TIFF.open(os.path.join(save_path, '{}.tif'.format(im_name)), mode='w')
    for k in range(seg.shape[0]):
        image3D.write_image(seg[k, :], compression='lzw', write_rgb=True)
    image3D.close()

