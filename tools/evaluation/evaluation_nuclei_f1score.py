# -*- coding: utf-8 -*-
"""
Created on 18-12-21 上午9:48
IDE PyCharm 

@author: Meng Dong
this script is used to evaluate the detection and segmentation
results on cell tracking challenge dataset, the metric is F1 score.
according to the miccai2018 3d instance segmentation paper.
"""

import numpy as np
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

# calculate the iou between ground truth and predicted segmentation
def segm_ovlp(mask, mask_ref, markers, kept):
    iou = np.zeros(len(markers))
    tp = np.zeros(len(markers))
    fp = np.zeros(len(markers))
    fn = np.zeros(len(markers))
    # pos_num = np.sum(mask)
    for id, marker in enumerate(markers):
        if kept[id]:
            pos_ref = mask_ref == marker
            inner_set = mask & pos_ref
            union_set = mask | pos_ref
            inner = np.sum(inner_set)
            union = np.sum(union_set)
            tp[id] = inner
            fp[id] = np.sum(mask ^ inner_set)
            fn[id] = np.sum(pos_ref ^ inner_set)
            iou[id] = np.float(inner)/np.float(union)
      #  else:
      #      fp[id] = pos_num
    return iou, tp, fp, fn

res_path = '../dets'# path for detection results
src_path = '../cell-tracking-challenge/Fluo-N3DH-SIM+_Train' # path for image data
test_txt = '../cell-tracking-challenge/Fluo-N3DH-SIM+_Train/test.txt'# path for test-list txt
ovthresh = 0.4
track = '02'
save_as_pkl = True

# get the test set list
list_f = open(test_txt, 'r')
img_paths = list_f.readlines()
img_paths = [x.rstrip() for x in img_paths]
list_f.close()
npos = 0
class_recs = {}
print('start evaluating detection...')
if track == '01':
    img_paths = img_paths[:70]
elif track == '02':
    img_paths = img_paths[70:]
# evaluate detection
#--------------------------------------------------------------------------------------------
# load ground truth and save as class_recs
for img_path in img_paths:
    # read label
    # bbox
    track = img_path.split('/')[-2]
    img_name = track + '_' + img_path.split('/')[-1][:-4]
    label_file = os.path.join(src_path, track+'_GT', 'BBOX', 'bbox_' +img_name[-3:]+ '.txt')
    class_recs[img_name] = {}
    class_recs[img_name]['bbox'], class_recs[img_name]['markers'] = load_gt_bbox(label_file)
    class_recs[img_name]['det'] = np.zeros(class_recs[img_name]['bbox'].shape[0], dtype=bool)
    npos += class_recs[img_name]['bbox'].shape[0]

#---------------------------------------------------------------------------------------------
# load bbox detection results
image_ids = []
confidence = np.empty((0, 1), dtype=np.float32)
BB = np.empty((0, 6), dtype=np.float32)
#class_dets = {}

for img_path in img_paths:
    img_name =  img_path.split('/')[-2] + '_' + img_path.split('/')[-1][:-4]
    if save_as_pkl:
        with open(os.path.join(res_path, img_name+'.pkl'), 'rb') as f_pkl:
            info = pickle.load(f_pkl)
        res = info['all_boxes'][1]
    else:
        res = np.load(os.path.join(res_path, img_name+'.npy'))

    # keep those whose score bigger than 0.4
    keep = res[:, -1] > 0.4
    res = res[keep, :]

    confidence = np.append(confidence, res[:, -1])
    if res.shape[1] == 7:
        BB = np.append(BB, res[:, :6], axis=0)
    elif res.shape[1] == 8:
        BB = np.append(BB, res[:, 1:7], axis=0)#
    for i in range(res.shape[0]):
        image_ids.append(img_name)


# traverse the predicted bbox and calculate the TPs and FPs
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)

for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)  # ground truth

    if BBGT.size > 0:
        # calculate IoU
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        izmin = np.maximum(BBGT[:, 2], bb[2])
        ixmax = np.minimum(BBGT[:, 3], bb[3])
        iymax = np.minimum(BBGT[:, 4], bb[4])
        izmax = np.minimum(BBGT[:, 5], bb[5])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        iss = np.maximum(izmax - izmin + 1., 0.)
        inters = iw * ih * iss

        # union
        uni = ((bb[3] - bb[0] + 1.) * (bb[4] - bb[1] + 1.) * (bb[5] - bb[2] + 1.) +
               (BBGT[:, 3] - BBGT[:, 0] + 1.) *
               (BBGT[:, 4] - BBGT[:, 1] + 1.) *
               (BBGT[:, 5] - BBGT[:, 2] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    # find the max IoU
    if ovmax > ovthresh:  # if bigger than the threshold
        if not R['det'][jmax]:    # is not detected
            tp[d] = 1.
            R['det'][jmax] = 1    # note as detected
        else:
            fp[d] = 1.
    else:
        fp[d] = 1.

recall_det = np.sum(tp)/npos
precision_det = np.sum(tp)/nd
f1score_det = 2 * (recall_det * precision_det)/(recall_det + precision_det)
print('done, detection f1 score is {}, precision is {}, recall is {}'.format(f1score_det, precision_det, recall_det))
