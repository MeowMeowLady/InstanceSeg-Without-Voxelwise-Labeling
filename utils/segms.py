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
"""Functions for interacting with segmentation masks in the COCO format.

The following terms are used in this module
    mask: a binary mask encoded as a 2D numpy array
    segm: a segmentation mask in one of the two COCO formats (polygon or RLE)
    polygon: COCO's polygon format
    RLE: COCO's run length encoding format
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from core.config import cfg
import numpy as np
from skimage import transform
import pycocotools.mask as mask_util
from utils.cython_mask_3d import binary_mask_to_rle, rle_to_binary_mask

def flip_segms(segms, height, width):
  """Left/right flip each mask in a list of masks."""

  def _flip_poly(poly, width):
    flipped_poly = np.array(poly)
    flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
    return flipped_poly.tolist()

  def _flip_rle(rle, height, width):
    if 'counts' in rle and type(rle['counts']) == list:
      # Magic RLE format handling painfully discovered by looking at the
      # COCO API showAnns function.
      rle = mask_util.frPyObjects([rle], height, width)
    mask = mask_util.decode(rle)
    mask = mask[:, ::-1, :]
    rle = mask_util.encode(np.array(mask, order='F', dtype=np.uint8))
    return rle

  flipped_segms = []
  for segm in segms:
    if type(segm) == list:
      # Polygon format
      flipped_segms.append([_flip_poly(poly, width) for poly in segm])
    else:
      # RLE format
      assert type(segm) == dict
      flipped_segms.append(_flip_rle(segm, height, width))
  return flipped_segms


def polys_to_mask(polygons, height, width):
  """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed inside a height x width image. The resulting
    mask is therefore of shape (height, width).
    """
  rle = mask_util.frPyObjects(polygons, height, width)
  mask = np.array(mask_util.decode(rle), dtype=np.float32)
  # Flatten in case polygons was a list
  mask = np.sum(mask, axis=2)
  mask = np.array(mask > 0, dtype=np.float32)
  return mask


def mask_to_bbox(mask):
  """Compute the tight bounding box of a binary mask."""
  xs = np.where(np.sum(mask, axis=0) > 0)[0]
  ys = np.where(np.sum(mask, axis=1) > 0)[0]

  if len(xs) == 0 or len(ys) == 0:
    return None

  x0 = xs[0]
  x1 = xs[-1]
  y0 = ys[0]
  y1 = ys[-1]
  return np.array((x0, y0, x1, y1), dtype=np.float32)


def polys_to_mask_wrt_box(polygons, box, M):
  """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
  w = box[2] - box[0]
  h = box[3] - box[1]

  w = np.maximum(w, 1)
  h = np.maximum(h, 1)

  polygons_norm = []
  for poly in polygons:
    p = np.array(poly, dtype=np.float32)
    p[0::2] = (p[0::2] - box[0]) * M / w
    p[1::2] = (p[1::2] - box[1]) * M / h
    polygons_norm.append(p)

  rle = mask_util.frPyObjects(polygons_norm, M, M)
  mask = np.array(mask_util.decode(rle), dtype=np.float32)
  # Flatten in case polygons was a list
  mask = np.sum(mask, axis=2)
  mask = np.array(mask > 0, dtype=np.float32)
  return mask


def spot_to_mask_wrt_box(spot, box, M):
  """Convert from the Imaris spots segmentation format to a binary mask
    encoded as a 3D array of data type numpy.float32. The spots segmentation
    is understood to be enclosed in the given box and rasterized to an M x M x M
    mask. The resulting mask is therefore of shape (M, M, M).
    """
  w = box[3] - box[0] # should take the brink cropping case into consideration
  h = box[4] - box[1]
  s = box[5] - box[2]

  w = np.maximum(w, 2)
  h = np.maximum(h, 2)
  s = np.maximum(s, 2)
  #print('w: {}, h: {}, s: {}'.format(w, h, s))

  sp = np.array(spot, dtype=np.float32)
  x = sp[0] - box[0]
  y = sp[1] - box[1]
  z = sp[2] - box[2]
  r = sp[3]
  #print('x: {}, y: {}, z: {}, r: {}'.format(x, y, z, r))
  mask = np.zeros((int(s), int(h), int(w)), dtype=np.float32)
  for i in range(int(s)):
    for j in range(int(h)):
      for k in range(int(w)):
        if (i-z)**2 + (j-y)**2 + (k-x)**2 < r**2:
          mask[i, j, k] = 1

  mask = transform.resize(mask, (M, M, M), mode='reflect', anti_aliasing=True)
  mask = np.array(mask > 0, dtype=np.float32)
  return mask

def rle_to_mask_wrt_box(rle, box_gt, box, M):
  """Convert from the rle format segmentation format to a binary mask
      encoded as a 3D array of data type numpy.float32. The rles segmentation
      is understood to be enclosed in the given box and rasterized to an M x M x M
      mask. The resulting mask is therefore of shape (M, M, M).
  """
  box_int = box.astype(np.int)
  box_gt_int = box_gt.astype(np.int)
  w = box_int[3] - box_int[0]  # should take the brink cropping case into consideration
  h = box_int[4] - box_int[1]
  s = box_int[5] - box_int[2]

  w = np.maximum(w, 2)
  h = np.maximum(h, 2)
  s = np.maximum(s, 2)

  mask_gt = rle_to_binary_mask(rle)
  box_in = np.zeros(6, dtype=np.int)

  box_in[:3] = np.maximum(box_int[:3], box_gt_int[:3])
  box_in[3:] = np.minimum(box_int[3:], box_gt_int[3:])
  box_in_by_box = box_in - np.append(box_int[:3], box_int[:3])
  x1 = box_in_by_box[0]
  y1 = box_in_by_box[1]
  z1 = box_in_by_box[2]
  x2 = box_in_by_box[3]
  y2 = box_in_by_box[4]
  z2 = box_in_by_box[5]
  box_in_by_box_gt = box_in - np.append(box_gt_int[:3], box_gt_int[:3])
  x1_gt = box_in_by_box_gt[0]
  y1_gt = box_in_by_box_gt[1]
  z1_gt = box_in_by_box_gt[2]
  x2_gt = box_in_by_box_gt[3]
  y2_gt = box_in_by_box_gt[4]
  z2_gt = box_in_by_box_gt[5]

  mask = np.zeros((int(s), int(h), int(w)), dtype=np.float32)
  mask[z1: z2, y1: y2, x1: x2] = mask_gt[z1_gt: z2_gt, y1_gt: y2_gt, x1_gt: x2_gt]

  mask = transform.resize(mask, (M, M, M), mode='reflect', anti_aliasing=True)
  mask = np.array(mask > 0, dtype=np.float32)
  return mask

def polys_to_boxes(polys):
  """Convert a list of polygons into an array of tight bounding boxes."""
  boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
  for i in range(len(polys)):
    poly = polys[i]
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    boxes_from_polys[i, :] = [x0, y0, x1, y1]

  return boxes_from_polys


def spots_to_boxes(spots):
  """
  Convert a list of spots into an array of tight bounding boxes.
  """
  boxes_from_spots = np.zeros((len(spots), 6), dtype=np.float32)
  slices, height, width = cfg.TRAIN.IN_SIZE[:]
  for i in range(len(spots)):
    spot = spots[i]
    x0 = max(0, spot[0] - spot[3] + 1)
    x1 = min(width - 1, spot[0] + spot[3] - 1)
    y0 = max(0, spot[1] - spot[3] + 1)
    y1 = min(height - 1, spot[1] + spot[3] - 1)
    z0 = max(0, spot[2] - spot[3] + 1)
    z1 = min(slices - 1, spot[2] + spot[3] - 1)
    boxes_from_spots[i, :] = [x0, y0, z0, x1, y1, z1]

  return boxes_from_spots


def mask_to_bbox_3d(mask, marker=None):
  """Compute the tight bounding box of a mask with multiple markers in 3d fashion."""
  if marker is None:
    pos = np.where(mask)
  else:
    pos = np.where(mask == marker)
  min_x = np.min(pos[2])
  max_x = np.max(pos[2])
  min_y = np.min(pos[1])
  max_y = np.max(pos[1])
  min_z = np.min(pos[0])
  max_z = np.max(pos[0])
  bbox = np.array([[min_x, min_y, min_z, max_x, max_y, max_z], ], dtype=np.float32)
  return bbox

def rles_to_bbox_3d(rles):
  boxes_from_rles = np.zeros((len(rles), 6), dtype=np.float32)
  for i in range(len(rles)):
    rle = rles[i]
    mask = rle_to_binary_mask(rle)
    bbox = mask_to_bbox_3d(mask)
    boxes_from_rles[i, :] = bbox
  return boxes_from_rles



def rle_mask_voting(top_masks,
                    all_masks,
                    all_dets,
                    iou_thresh,
                    binarize_thresh,
                    method='AVG'):
  """Returns new masks (in correspondence with `top_masks`) by combining
    multiple overlapping masks coming from the pool of `all_masks`. Two methods
    for combining masks are supported: 'AVG' uses a weighted average of
    overlapping mask pixels; 'UNION' takes the union of all mask pixels.
    """
  if len(top_masks) == 0:
    return

  all_not_crowd = [False] * len(all_masks)
  top_to_all_overlaps = mask_util.iou(top_masks, all_masks, all_not_crowd)
  decoded_all_masks = [
      np.array(mask_util.decode(rle), dtype=np.float32) for rle in all_masks
  ]
  decoded_top_masks = [
      np.array(mask_util.decode(rle), dtype=np.float32) for rle in top_masks
  ]
  all_boxes = all_dets[:, :4].astype(np.int32)
  all_scores = all_dets[:, 4]

  # Fill box support with weights
  mask_shape = decoded_all_masks[0].shape
  mask_weights = np.zeros((len(all_masks), mask_shape[0], mask_shape[1]))
  for k in range(len(all_masks)):
    ref_box = all_boxes[k]
    x_0 = max(ref_box[0], 0)
    x_1 = min(ref_box[2] + 1, mask_shape[1])
    y_0 = max(ref_box[1], 0)
    y_1 = min(ref_box[3] + 1, mask_shape[0])
    mask_weights[k, y_0:y_1, x_0:x_1] = all_scores[k]
  mask_weights = np.maximum(mask_weights, 1e-5)

  top_segms_out = []
  for k in range(len(top_masks)):
    # Corner case of empty mask
    if decoded_top_masks[k].sum() == 0:
      top_segms_out.append(top_masks[k])
      continue

    inds_to_vote = np.where(top_to_all_overlaps[k] >= iou_thresh)[0]
    # Only matches itself
    if len(inds_to_vote) == 1:
      top_segms_out.append(top_masks[k])
      continue

    masks_to_vote = [decoded_all_masks[i] for i in inds_to_vote]
    if method == 'AVG':
      ws = mask_weights[inds_to_vote]
      soft_mask = np.average(masks_to_vote, axis=0, weights=ws)
      mask = np.array(soft_mask > binarize_thresh, dtype=np.uint8)
    elif method == 'UNION':
      # Any pixel that's on joins the mask
      soft_mask = np.sum(masks_to_vote, axis=0)
      mask = np.array(soft_mask > 1e-5, dtype=np.uint8)
    else:
      raise NotImplementedError('Method {} is unknown'.format(method))
    rle = mask_util.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    top_segms_out.append(rle)

  return top_segms_out


def rle_mask_nms(masks, dets, thresh, mode='IOU'):
  """Performs greedy non-maximum suppression based on an overlap measurement
    between masks. The type of measurement is determined by `mode` and can be
    either 'IOU' (standard intersection over union) or 'IOMA' (intersection over
    mininum area).
    """
  if len(masks) == 0:
    return []
  if len(masks) == 1:
    return [0]

  if mode == 'IOU':
    # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(union(m1, m2))
    all_not_crowds = [False] * len(masks)
    ious = mask_util.iou(masks, masks, all_not_crowds)
  elif mode == 'IOMA':
    # Computes ious[m1, m2] = area(intersect(m1, m2)) / min(area(m1), area(m2))
    all_crowds = [True] * len(masks)
    # ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
    ious = mask_util.iou(masks, masks, all_crowds)
    # ... = max(area(intersect(m1, m2)) / area(m2),
    #           area(intersect(m2, m1)) / area(m1))
    ious = np.maximum(ious, ious.transpose())
  elif mode == 'CONTAINMENT':
    # Computes ious[m1, m2] = area(intersect(m1, m2)) / area(m2)
    # Which measures how much m2 is contained inside m1
    all_crowds = [True] * len(masks)
    ious = mask_util.iou(masks, masks, all_crowds)
  else:
    raise NotImplementedError('Mode {} is unknown'.format(mode))

  scores = dets[:, 4]
  order = np.argsort(-scores)

  keep = []
  while order.size > 0:
    i = order[0]
    keep.append(i)
    ovr = ious[i, order[1:]]
    inds_to_keep = np.where(ovr <= thresh)[0]
    order = order[inds_to_keep + 1]

  return keep


def rle_masks_to_boxes(masks):
  """Computes the bounding box of each mask in a list of RLE encoded masks."""
  if len(masks) == 0:
    return []

  decoded_masks = [
      np.array(mask_util.decode(rle), dtype=np.float32) for rle in masks
  ]

  def get_bounds(flat_mask):
    inds = np.where(flat_mask > 0)[0]
    return inds.min(), inds.max()

  boxes = np.zeros((len(decoded_masks), 4))
  keep = [True] * len(decoded_masks)
  for i, mask in enumerate(decoded_masks):
    if mask.sum() == 0:
      keep[i] = False
      continue
    flat_mask = mask.sum(axis=0)
    x0, x1 = get_bounds(flat_mask)
    flat_mask = mask.sum(axis=1)
    y0, y1 = get_bounds(flat_mask)
    boxes[i, :] = (x0, y0, x1, y1)

  return boxes, np.where(keep)[0]

if __name__ == '__main__':
  '''polygons = [[[0, 2, 0, 3, 1, 4, 2, 4, 3, 3, 3, 2, 2, 0, 1, 0]]]
  box = polys_to_boxes(polygons)
  print(box[0])
  m =  polys_to_mask_wrt_box(polygons[0], box[0], 7)
  print(m)
  spot = [0, 20, 26, 12]
  box = spots_to_boxes([spot])[0]
  print(box)
  m = spot_to_mask_wrt_box(spot, box, 24)
  print(m[:, :, 0])'''
  a = np.ones((1,14,14), dtype=np.uint16)
  b = transform.resize(a, (14,14,14), mode='reflect',anti_aliasing=True)
