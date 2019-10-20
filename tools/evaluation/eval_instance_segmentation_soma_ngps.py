# -*- coding: utf-8 -*-
"""
Created on 19-3-3 下午5:36
IDE PyCharm 

@author: Meng Dong
"""

import numpy as np

from mask_iou import mask_iou, mask_iou_fast
from skimage import io, measure
import os

def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:  # use method from 07 year
        # 11 points
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # interpolate
            mpre = np.concatenate((mpre, [p / 11.]))
            ap = ap + p / 11.
    else:  # new method from 2010, consider all the data points
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision curve value (also use interpolate)
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return mrec, mpre, ap


def eval_instance_segmentation_soma(
        flag, pred_mask_path, gt_mask_path, img_names, iou_thresh, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.
    This function evaluates predicted masks obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in `FCIS`_.
    .. _`FCIS`: https://arxiv.org/abs/1611.07709
    Args:
        pred_masks (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of masks. Its index corresponds to an index for the base
            dataset. Each element of :obj:`pred_masks` is an object mask
            and is an array whose shape is :math:`(R, H, W)`,
            where :math:`R` corresponds
            to the number of masks, which may vary among images.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_masks`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted masks. Similar to :obj:`pred_masks`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_masks (iterable of numpy.ndarray): An iterable of ground truth
            masks whose length is :math:`N`. An element of :obj:`gt_masks` is
            an object mask whose shape is :math:`(R, H, W)`. Note that the
            number of masks :math:`R` in each image does not need to be
            same as the number of corresponding predicted masks.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_masks`. Its
            length is :math:`N`.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.
    Returns:
        dict:
        The keys, value-types and the description of the values are listed
        below.
        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.
    """

    prec, rec = calc_instance_segmentation_voc_prec_rec(
        flag, pred_mask_path, gt_mask_path, img_names, iou_thresh)

    _, _, ap = voc_ap(rec, prec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_instance_segmentation_voc_prec_rec(
        flag, pred_mask_path, gt_mask_path, img_names, iou_thresh):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.
    This function calculates precision and recall of
    predicted masks obtained from a dataset which has :math:`N` images.
    The code is based on the evaluation code used in `FCIS`_.
    .. _`FCIS`: https://arxiv.org/abs/1611.07709
    Args:
        pred_masks (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of masks. Its index corresponds to an index for the base
            dataset. Each element of :obj:`pred_masks` is an object mask
            and is an array whose shape is :math:`(R, H, W)`,
            where :math:`R` corresponds
            to the number of masks, which may vary among images.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_masks`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted masks. Similar to :obj:`pred_masks`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_masks (iterable of numpy.ndarray): An iterable of ground truth
            masks whose length is :math:`N`. An element of :obj:`gt_masks` is
            an object mask whose shape is :math:`(R, H, W)`. Note that the
            number of masks :math:`R` in each image does not need to be
            same as the number of corresponding predicted masks.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_masks`. Its
            length is :math:`N`.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.
        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.
    """

    n_pos = 0
    #score = []
    match = []
    save = False # True if save evaluation results
    if save:
        fid = open('../eval_result.txt', 'w+')

    for img_name in img_names:
        match_single = []
        print('img {}'.format(img_name))
        # the pred_mask and gt_mask are uint16 3d array, each instance has an id
        gt_mask = io.imread(os.path.join(gt_mask_path, img_name, img_name + '.tif'))
        if flag == 'NGPS':
            pred_mask = np.zeros(gt_mask.shape, dtype=np.uint16)
            s, h, w = pred_mask.shape[:]
            swc_file = open(os.path.join(pred_mask_path, img_name + '.swc'), 'r')
            lines = swc_file.read().rstrip().split('\n')
            swc_file.close()
            mask_id = 0
            for line in lines:
                mask_id += 1
                lineParts = line.rstrip().split(' ')
                x = int(np.float(lineParts[2]))
                y = int(np.float(lineParts[3]))
                z = int(np.float(lineParts[4]))
                r = int(np.float(lineParts[5]))
                x_start = max(1, x - r)
                x_end = min(w, x + r + 1)
                y_start = max(1, y - r)
                y_end = min(h, y + r + 1)
                z_start = max(1, z - r)
                z_end = min(s, z + r + 1)
                for ix in range(x_start, x_end):
                    for iy in range(y_start, y_end):
                        for iz in range(z_start, z_end):
                            if ((ix - x) ** 2 + (iy - y) ** 2 + (iz - z) ** 2 <= r ** 2) and r >= 6:
                                pred_mask[iz, iy, ix] = mask_id
        elif flag == 'DSN':
            pred_mask = io.imread(os.path.join(pred_mask_path, img_name + '.tif'))
            pred_mask = measure.label(pred_mask) # connected components as instances
            s, h, w = pred_mask.shape[:]
        # get all the ids of instance and remove background id 0
        pred_mask_ids = np.unique(pred_mask).tolist()
        for m_id in pred_mask_ids:
            if np.sum(pred_mask==m_id) < 300:
                pred_mask_ids.remove(m_id)
        pred_mask_ids.remove(0)
        gt_mask_ids = np.unique(gt_mask).tolist()
        gt_mask_ids.remove(0)
        # update n_pos
        n_pos += len(gt_mask_ids)
        # get pred masks for each instance and save as bool array
        pred_masks = np.zeros((len(pred_mask_ids), s, h, w), dtype=np.bool)
        for i in range(len(pred_mask_ids)):
            pred_masks[i, :] = pred_mask==pred_mask_ids[i]
        if len(pred_mask_ids)==0:
            continue
        if len(gt_mask_ids) == 0:
            match.extend([0,]*len(pred_mask_ids))
            match_single.extend([0, ] * len(pred_mask_ids))
        # get gt masks
        gt_masks = np.zeros((len(gt_mask_ids), s, h, w), dtype=np.bool)
        for i, gt_mask_id in enumerate(gt_mask_ids):
            gt_masks[i, :] = gt_mask==gt_mask_id

        iou = mask_iou_fast(pred_masks, gt_masks)
        gt_index = iou.argmax(axis=1)
        # set -1 if there is no matching ground truth
        gt_index[iou.max(axis=1) < iou_thresh] = -1
        del iou

        selec = np.zeros(len(gt_mask_ids), dtype=bool)
        for gt_idx in gt_index:
            if gt_idx >= 0:
                if not selec[gt_idx]:
                    match.append(1)
                    match_single.append(1)
                else:
                    match.append(0)
                    match_single.append(0)
                selec[gt_idx] = True
            else:
                match.append(0)
        match_single = np.array(match_single, dtype=np.int8)
        tp_single = np.cumsum(match_single == 1)
        fp_single = np.cumsum(match_single == 0)

        # If an element of fp + tp is 0,
        # the prec is nan.
        prec_single = tp_single / (fp_single + tp_single)
        # If n_pos is 0, rec is None.
        if len(gt_mask_ids) > 0:
            rec_single = tp_single / len(gt_mask_ids)
        _, _, ap_single = voc_ap(rec_single, prec_single, use_07_metric=False)
        if save:
            fid.write('{}: {:.4f}\n'.format(img_name, ap_single))
    if save:
        fid.close()

    match = np.array(match, dtype=np.int8)

    tp = np.cumsum(match == 1)
    fp = np.cumsum(match == 0)

    # If an element of fp + tp is 0,
    # the prec is nan.
    prec = tp / (fp + tp)
    # If n_pos is 0, rec is None.
    if n_pos > 0:
        rec = tp / n_pos

    return prec, rec


if __name__ == "__main__":
    flag = 'NGPS'#'NGPS' or 'DSN
    if flag == 'DSN':
        pred_mask_path = ''# DSN's segmentation results
    elif flag == 'NGPS':
        pred_mask_path = ''# NeuroGPS segmentation results
    gt_mask_path = '' # ground truth
    img_names = os.listdir(gt_mask_path)
    res = eval_instance_segmentation_soma(flag, pred_mask_path, gt_mask_path, img_names, iou_thresh=0.3)
    print('ap: {}'.format(res['ap']))




