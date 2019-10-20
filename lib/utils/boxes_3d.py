# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Box manipulation functions. The internal Detectron box format is
[x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2)
specify the bottom-right box corner. Boxes from external sources, e.g.,
datasets, may be in other formats (such as [x, y, w, h]) and require conversion.

This module uses a convention that may seem strange at first: the width of a box
is computed as x2 - x1 + 1 (likewise for height). The "+ 1" dates back to old
object detection days when the coordinates were integer pixel indices, rather
than floating point coordinates in a subpixel coordinate frame. A box with x2 =
x1 and y2 = y1 was taken to include a single pixel, having a width of 1, and
hence requiring the "+ 1". Now, most datasets will likely provide boxes with
floating point coordinates and the width should be more reasonably computed as
x2 - x1.

In practice, as long as a model is trained and tested with a consistent
convention either decision seems to be ok (at least in our experience on COCO).
Since we have a long history of training models with the "+ 1" convention, we
are reluctant to change it even if our modern tastes prefer not to use it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import warnings
import numpy as np

from core.config import cfg
import utils.cython_bbox_3d as cython_bbox_3d
import utils.cython_nms_3d as cython_nms_3d

bbox_overlaps_3d = cython_bbox_3d.bbox_overlaps_3d
DEBUG = False

def boxes_volume(boxes):
    """Compute the volumes of an array of boxes."""
    w = (boxes[:, 3] - boxes[:, 0] + 1)
    h = (boxes[:, 4] - boxes[:, 1] + 1)
    s = (boxes[:, 5] - boxes[:, 2] + 1)
    volumes = w * h * s

    neg_area_idx = np.where(volumes < 0)[0]
    if neg_area_idx.size:
        warnings.warn("Negative areas founds: %d" % neg_area_idx.size, RuntimeWarning)
    #TODO proper warm up and learning rate may reduce the prob of assertion fail
    # assert np.all(areas >= 0), 'Negative areas founds'
    return volumes, neg_area_idx


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xyzwhs_to_xyzxyz(xyzwhs):
    """Convert [x1 y1 z1 w h s] box format to [x1 y1 z1 x2 y2 z2] format."""
    if isinstance(xyzwhs, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyzwhs) == 6
        x1, y1, z1 = xyzwhs[0], xyzwhs[1], xyzwhs[2]
        x2 = x1 + np.maximum(0., xyzwhs[3] - 1.)
        y2 = y1 + np.maximum(0., xyzwhs[4] - 1.)
        z2 = z1 + np.maximum(0., xyzwhs[5] - 1.)
        return (x1, y1, z1, x2, y2, z2)
    elif isinstance(xyzwhs, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xyzwhs[:, 0:3], xyzwhs[:, 0:3] + np.maximum(0, xyzwhs[:, 3:6] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')



def xyzxyz_to_xyzwhs(xyzxyz):
    """Convert [x1 y1 z1 x2 y2 z2] box format to [x1 y1 z1 w h s] format."""
    if isinstance(xyzxyz, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyzxyz) == 6
        x1, y1, z1 = xyzxyz[0], xyzxyz[1], xyzxyz[2]
        w = xyzxyz[3] - x1 + 1
        h = xyzxyz[4] - y1 + 1
        s = xyzxyz[5] - z1 + 1
        return (x1, y1, z1, w, h, s)
    elif isinstance(xyzxyz, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyzxyz[:, 0:3], xyzxyz[:, 3:6] - xyzxyz[:, 0:3] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w > min_size) & (h > min_size))[0]
    return keep


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def clip_xyzxyz_to_image(x1, y1, z1, x2, y2, z2, slices, height, width):
    """Clip coordinates to an image with the given slices, height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    z1 = np.minimum(slices - 1., np.maximum(0, z1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    z2 = np.minimum(slices - 1., np.maximum(0., z2))
    return x1, y1, z1, x2, y2, z2


def clip_tiled_boxes_3d(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [slices, height, width] and boxes
    has shape (N, 6 * num_tiled_boxes)."""
    assert boxes.shape[1] % 6== 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 6.'.format(
        boxes.shape[1]
    )
    # x1 >= 0
    boxes[:, 0::6] = np.maximum(np.minimum(boxes[:, 0::6], im_shape[2] - 1), 0)
    # y1 >= 0
    boxes[:, 1::6] = np.maximum(np.minimum(boxes[:, 1::6], im_shape[1] - 1), 0)
    # z1 >= 0
    boxes[:, 2::6] = np.maximum(np.minimum(boxes[:, 2::6], im_shape[0] - 1), 0)
    # x2 < im_shape[2]
    boxes[:, 3::6] = np.maximum(np.minimum(boxes[:, 3::6], im_shape[2] - 1), 0)
    # y2 < im_shape[1]
    boxes[:, 4::6] = np.maximum(np.minimum(boxes[:, 4::6], im_shape[1] - 1), 0)
    # z2 < im_shape[0]
    boxes[:, 5::6] = np.maximum(np.minimum(boxes[:, 5::6], im_shape[0] - 1), 0)
    return boxes


# $add by Meng$
def bbox_transform_3d(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 3] - boxes[:, 0] + 1.0
    heights = boxes[:, 4] - boxes[:, 1] + 1.0
    slices = boxes[:, 5] - boxes[:, 2] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    ctr_z = boxes[:, 2] + 0.5 * slices

    wx, wy, wz, ww, wh, ws = weights
    dx = deltas[:, 0::6] / wx
    dy = deltas[:, 1::6] / wy
    dz = deltas[:, 2::6] / wz
    dw = deltas[:, 3::6] / ww
    dh = deltas[:, 4::6] / wh
    ds = deltas[:, 5::6] / ws

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, cfg.BBOX_XFORM_CLIP)
    dh = np.minimum(dh, cfg.BBOX_XFORM_CLIP)
    ds = np.minimum(ds, cfg.BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_z = dz * slices[:, np.newaxis] + ctr_z[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_s = np.exp(ds) * slices[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    if DEBUG:
        print('width shape: {}, dx shape: {}'.format(widths.shape, dx.shape))
        print('ctr_x: {}, ctr_y: {}, ctr_z: {}'.format(ctr_x, ctr_y, ctr_z))
        print('pred_s shape: {}, pred_ctr_x shape:{}, pred_boxes shape:{}'.format(pred_s.shape, pred_ctr_x.shape, pred_boxes.shape))
        print('dx: {}, dy: {}, dz: {}, ds: {}'.format(dx[0, :], dy[0, :], dz[0, :], ds[0, :]))
        print('pred_ctr_x: {}, pred_ctr_y: {}, pred_ctr_z: {}, pred_s: {}'.format(pred_ctr_x[0,:],
                                                                              pred_ctr_y[0,:], pred_ctr_z[0,:],pred_s[0,:]))
    # x1
    pred_boxes[:, 0::6] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::6] = pred_ctr_y - 0.5 * pred_h
    # z1
    pred_boxes[:, 2::6] = pred_ctr_z - 0.5 * pred_s
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::6] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 4::6] = pred_ctr_y + 0.5 * pred_h - 1
    # z2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 5::6] = pred_ctr_z + 0.5 * pred_s - 1

    return pred_boxes


def bbox_transform_inv_3d(boxes, gt_boxes, weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 3] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 4] - boxes[:, 1] + 1.0
    ex_slices = boxes[:, 5] - boxes[:, 2] + 1.0
    # ex_sides = np.maximum(ex_widths, ex_heights, ex_slices)

    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights
    ex_ctr_z = boxes[:, 2] + 0.5 * ex_slices

    gt_width = gt_boxes[:, 3] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 4] - gt_boxes[:, 1] + 1.0
    gt_slices = gt_boxes[:, 5] - gt_boxes[:, 2] + 1.0

    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights
    gt_ctr_z = gt_boxes[:, 2] + 0.5 * gt_slices

    wx, wy, wz, ww, wh, ws = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dz = wz * (gt_ctr_z - ex_ctr_z)  / ex_slices
    targets_dw = ww * np.log(gt_width / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)
    targets_ds = ws * np.log(gt_slices / ex_slices)

    targets = np.vstack((targets_dx, targets_dy, targets_dz,
                         targets_dw, targets_dh, targets_ds)).transpose()
    return targets


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 3] - boxes[:, 0]) * .5
    h_half = (boxes[:, 4] - boxes[:, 1]) * .5
    s_half = (boxes[:, 5] - boxes[:, 2]) * .5
    x_c = (boxes[:, 3] + boxes[:, 0]) * .5
    y_c = (boxes[:, 4] + boxes[:, 1]) * .5
    z_c = (boxes[:, 5] + boxes[:, 2]) * .5

    w_half *= scale
    h_half *= scale
    s_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 3] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 4] = y_c + h_half
    boxes_exp[:, 2] = z_c - s_half
    boxes_exp[:, 5] = z_c + s_half

    return boxes_exp


def flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def aspect_ratio(boxes, aspect_ratio):
    """Perform width-relative aspect ratio transformation."""
    boxes_ar = boxes.copy()
    boxes_ar[:, 0::6] = aspect_ratio * boxes[:, 0::6]
    boxes_ar[:, 3::6] = aspect_ratio * boxes[:, 3::6]

    return boxes_ar


def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 7] each row is [x1 y1 z1 x2 y2 z2, sore]
    # all_dets is [N, 7] each row is [x1 y1 z1 x2 y2 z2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :6]
    all_boxes = all_dets[:, :6]
    all_scores = all_dets[:, 6]
    top_to_all_overlaps = bbox_overlaps_3d(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 6] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 6] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 6] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws**beta)**(1.0 / beta)
            top_dets_out[k, 6] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 6] = ws.sum() / float(len(ws))**beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out


def nms_3d(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms_3d.nms_3d(dets, thresh)

def nms_3d_volume(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms_3d.nms_3d_volume(dets, thresh)

def soft_nms(
    dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if dets.shape[0] == 0:
        return dets, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets, keep = cython_nms_3d.soft_nms_3d(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method])
    )
    return dets, keep

