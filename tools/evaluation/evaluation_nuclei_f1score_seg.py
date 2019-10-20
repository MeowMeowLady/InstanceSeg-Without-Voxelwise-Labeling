# -*- coding: utf-8 -*-
"""
Created on 19-3-8 上午11:57
IDE PyCharm 

@author: Meng Dong
this script is used to evaluate the instance segmentation results
as the metric used in https://arxiv.org/abs/1806.11137
Deep Learning Based Instance Segmentation in 3D Biomedical Images Using Weak Annotation
"""
import numpy as np
from skimage import io
import os
from six.moves import cPickle as pickle


def load_gt_bbox(label_file):
    label_fid = open(label_file, 'r')
    label = label_fid.readlines()
    label_fid.close()
    gt_boxes = np.empty((0, 6), dtype=np.float32)
    markers = np.empty(0, dtype=np.uint16)
    for obj in label[1:]:
        parts = obj.rstrip().split(' ')
        x1 = int(parts[1])
        y1 = int(parts[2])
        z1 = int(parts[3])
        x2 = x1 + int(parts[4]) - 1
        y2 = y1 + int(parts[5]) - 1
        z2 = z1 + int(parts[6]) - 1
        marker = int(parts[7])
        markers = np.append(markers, marker)
        box = np.array((x1, y1, z1, x2, y2, z2), dtype=np.float32)[np.newaxis, :]
        gt_boxes = np.append(gt_boxes, box, axis=0)
    return gt_boxes, markers

res_path = '../binarization_2dotsu'# path for instance segmentation results
src_path = '../cell-tracking-challenge/Fluo-N3DH-SIM+_Train' # path for image data
test_txt = '../cell-tracking-challenge/Fluo-N3DH-SIM+_Train/test.txt'# path for test-list txt

ovthresh = 0.4

# get the test set list
list_f = open(test_txt, 'r')
img_paths = list_f.readlines()
img_paths = [x.rstrip() for x in img_paths]
list_f.close()
npos = 0
class_recs = {}
print('start evaluating detection...')
# evaluate detection
# load ground truth and save as class_recs

gt_pixel = 0
pre_pixel = 0
tp_pixel = 0
fp_pixel = 0
save_as_pkl = False
folder = '02'
if folder == '01':
    img_paths = img_paths[:70]
elif folder == '02':
    img_paths = img_paths[70:]

for img_path in img_paths:
    # read label
    track = img_path.split('/')[-2]
    img_name = track + '_' + img_path.split('/')[-1][:-4]
    print(img_name)
    label_file = os.path.join(src_path, track+'_GT', 'BBOX', 'bbox_' +img_name[-3:]+ '.txt')
    gt_bbox, gt_mask_ids = load_gt_bbox(label_file)
    gt_bbox = gt_bbox.astype(float)
    detected = np.zeros(gt_bbox.shape[0], dtype=bool)
    # read detection results
    if save_as_pkl:
        with open(os.path.join(res_path,  img_name+'.pkl'), 'rb') as f_pkl:
            info = pickle.load(f_pkl)
        dets_bbox = info['all_boxes'][1]
    else:
        dets = np.load(os.path.join(res_path, img_name+'.npy'))
        dets_bbox = dets[:, 1:7].astype(float)
    tp = np.zeros(dets_bbox.shape[0])
    fp = np.zeros(dets_bbox.shape[0])

    # read gt_mask and pred_mask
    gt_mask = io.imread(os.path.join(src_path, track+'_GT', 'SEG', 'man_seg'+img_name[-3:]+'.tif'))
    pred_mask = io.imread(os.path.join(res_path, img_name+'.tif'))
    gt_mask_bool = gt_mask > 0
    pred_mask_bool = pred_mask > 0
    #
    gt_pixel += np.sum(gt_mask_bool)
    keep_pred_mask = np.zeros(pred_mask.shape, dtype=bool)
    pre_pixel += np.sum(pred_mask_bool)


    if gt_bbox.shape[0] > 0:
        for ib, bbox in enumerate(dets_bbox):
            # calculate IoU
            # intersection
            ixmin = np.maximum(gt_bbox[:, 0], bbox[0])
            iymin = np.maximum(gt_bbox[:, 1], bbox[1])
            izmin = np.maximum(gt_bbox[:, 2], bbox[2])
            ixmax = np.minimum(gt_bbox[:, 3], bbox[3])
            iymax = np.minimum(gt_bbox[:, 4], bbox[4])
            izmax = np.minimum(gt_bbox[:, 5], bbox[5])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            iss = np.maximum(izmax - izmin + 1., 0.)
            inters = iw * ih * iss

            # union
            uni = ((bbox[3] - bbox[0] + 1.) * (bbox[4] - bbox[1] + 1.) * (bbox[5] - bbox[2] + 1.) +
                   (gt_bbox[:, 3] - gt_bbox[:, 0] + 1.) *
                   (gt_bbox[:, 4] - gt_bbox[:, 1] + 1.) *
                   (gt_bbox[:, 5] - gt_bbox[:, 2] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

            # find the max IoU
            if ovmax > ovthresh:  # if bigger than the threshold
                if not detected[jmax]:  # is not detected
                    tp[ib] = 1.
                    detected[jmax] = 1  # note as detected
                    x1, y1, z1, x2, y2, z2 = bbox.astype(int)[:]
                    keep_pred_mask[z1: z2+1, y1: y2+1, x1: x2+1] = pred_mask_bool[z1: z2+1, y1: y2+1, x1: x2+1].copy()
                else:
                    fp[ib] = 1.
            else:
                fp[ib] = 1.

        tp_pixel += np.sum(keep_pred_mask&gt_mask_bool)
        ppre = np.sum(keep_pred_mask&gt_mask_bool)/np.sum(pred_mask_bool)
        rrec = np.sum(keep_pred_mask&gt_mask_bool)/np.sum(gt_mask_bool)
        print('F1: {:.5f} precision: {:.5f} recall: {:.5f} '.format((2*ppre*rrec)/(ppre+rrec), ppre, rrec))


recall = tp_pixel/gt_pixel
precision = tp_pixel/pre_pixel
f1score_det = 2 * (recall * precision)/(recall + precision)
print('done, instance segmentation f1 score is {:.5f}, precision is {:.5f}, recall is {:.5f}'.format(f1score_det, precision, recall))


