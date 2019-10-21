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
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
"""blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import cPickle as pickle
import numpy as np
import cv2
import math
import numpy.random as npr
import utils.segms as segm_utils
import utils.boxes_3d as box_utils_3d

from core.config import cfg


def get_image_blob(im):
    """Convert an image into a network input.

    Arguments:
        im (ndarray): a gray scale image

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale (float): image scale (target size) / (original size)
        im_info (ndarray)
    """
    processed_im, im_scale = prep_im_for_blob(im, entry = None, phase = 'test')
    blob = im_list_to_blob(processed_im)

    slices, height, width = blob.shape[2], blob.shape[3], blob.shape[4]
    im_info = np.hstack((slices, height, width, im_scale))[np.newaxis, :]
    return blob, im_scale, im_info.astype(np.float32)


def im_list_to_blob(ims):
    """Convert a list of images into a network input. Assumes images were
    prepared using prep_im_for_blob or equivalent
    Output is a 5D HCSHW tensor of the images concatenated along axis 0 with
    shape.
    """
    if not isinstance(ims, list):
        ims = [ims]
    max_shape = get_max_shape([im.shape[:3] for im in ims]) # np array [max_s, max_h, max_w]

    num_images = len(ims)
    blob = np.zeros(
        (num_images, max_shape[0], max_shape[1], max_shape[2], 1), dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], 0:im.shape[2], 0] = im
    # Move channels (axis 4) to axis 1
    # Axis order will become: (batch elem, channel, slices, height, width)
    channel_swap = (0, 4, 1, 2, 3)
    blob = blob.transpose(channel_swap)
    return blob


def get_max_shape(im_shapes):
    """Calculate max spatial size (s, h, w) for batching given a list of image shapes
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 3
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
        max_shape[2] = int(np.ceil(max_shape[2] / stride) * stride)
    return max_shape


def crop_data_3d(im, entry):

    #random select the cropping start index and crop with half-overlap
    #select the cropped block containing most positive voxels because of the sparsity

    data_slices, data_height, data_width = map(int, im.shape[:])
    boxes = entry['boxes'].copy()
    segms = entry['segms'].copy()
    ss = np.array(cfg.TRAIN.IN_SIZE, dtype=np.int16)
    x_min = math.floor(np.min(boxes[:, 0]))
    y_min = math.floor(np.min(boxes[:, 1]))
    z_min = math.floor(np.min(boxes[:, 2]))
    x_s_min = 0
    x_s_max = min(x_min, data_width - ss[2])
    y_s_min = 0
    y_s_max = min(y_min, data_height - ss[1])
    z_s_min = 0
    z_s_max = min(z_min, data_slices - ss[0])
    x_s = x_s_min if x_s_min == x_s_max else \
        npr.choice(range(x_s_min, x_s_max + 1))
    y_s = y_s_min if y_s_min == y_s_max else \
        npr.choice(range(y_s_min, y_s_max + 1))
    z_s = z_s_min if z_s_min == z_s_max else \
        npr.choice(range(z_s_min, z_s_max + 1))
    s_list = list(range(z_s, data_slices - ss[0], int(ss[0] / 2)))
    h_list = list(range(y_s, data_height - ss[1], int(ss[1] / 2)))
    w_list = list(range(x_s, data_width - ss[2], int(ss[2] / 2)))

    s_list.append(data_slices - ss[0])
    h_list.append(data_height - ss[1])
    w_list.append(data_width - ss[2])

    max_pos_num = 0
    posit = []
    for z in s_list:
        for y in h_list:
            for x in w_list:
                boxes[:, 0::3] -= x
                boxes[:, 1::3] -= y
                boxes[:, 2::3] -= z
                np.clip(boxes[:, 0::3], 0, ss[2] - 1, out=boxes[:, 0::3])
                np.clip(boxes[:, 1::3], 0, ss[1] - 1, out=boxes[:, 1::3])
                np.clip(boxes[:, 2::3], 0, ss[0] - 1, out=boxes[:, 2::3])
                invalid = (boxes[:, 0] == boxes[:, 3]) | (boxes[:, 1] == boxes[:, 4]) | (boxes[:, 2] == boxes[:, 5])
                valid_inds = np.nonzero(~ invalid)[0]
                pos_box_volumes, _ = box_utils_3d.boxes_volume(boxes[valid_inds, :])
                tmp_pos_num = np.sum(pos_box_volumes)
                if tmp_pos_num > max_pos_num:
                    max_pos_num = tmp_pos_num
                    posit = [x, y, z]
                boxes = entry['boxes'].copy()
    x, y, z = posit[:]
    im = im[z: z+ss[0], y: y+ss[1], x: x+ss[2]]
    boxes[:, 0::3] -= x
    boxes[:, 1::3] -= y
    boxes[:, 2::3] -= z
    segms[:, 0] -= x
    segms[:, 1] -= y
    segms[:, 2] -= z
    np.clip(boxes[:, 0::3], 0, ss[2] - 1, out=boxes[:, 0::3])
    np.clip(boxes[:, 1::3], 0, ss[1] - 1, out=boxes[:, 1::3])
    np.clip(boxes[:, 2::3], 0, ss[0] - 1, out=boxes[:, 2::3])
    np.clip(segms[:, 0], 0, ss[2] - 1, out=segms[:, 0])
    np.clip(segms[:, 1], 0, ss[1] - 1, out=segms[:, 1])
    np.clip(segms[:, 2], 0, ss[0] - 1, out=segms[:, 2])
    entry['boxes'] = boxes
    entry['segms'] = segms
    entry['slices'] = ss[0]
    entry['height'] = ss[1]
    entry['width'] = ss[2]
    return im


def prep_im_for_blob(im, entry, phase):
    """Prepare an image for use as a network input blob. Specially:
      - Subtract per-channel pixel mean
      - Convert to float32
      - Rescale to each of the specified target size (capped at max_size)
      - crop if need
    Returns a list of transformed images, one for each target size. Also returns
    the scale factors that were used to compute each returned image.
    """
    im = im.astype(np.float32, copy=False)
    if cfg.PP_METHOD == 'norm1':
        mask = im > 0
        mean_val = np.mean(im[mask])
        std_val = np.std(im[mask])
        im = (im - mean_val) / std_val
    elif cfg.PP_METHOD == 'norm2':
        im = im/65535.

    if phase == 'train' and cfg.TRAIN.NEED_CROP:
        im = crop_data_3d(im, entry)
        # Check bounding box
        boxes = entry['boxes']
        invalid = (boxes[:, 0] == boxes[:, 3]) | (boxes[:, 1] == boxes[:, 4]) | (boxes[:, 2] == boxes[:, 5])
        valid_inds = np.nonzero(~ invalid)[0]
        if len(valid_inds) < len(boxes):
            for key in ['boxes', 'segms', 'gt_classes', 'seg_volumes', 'gt_overlaps', 'is_crowd',
                        'gt_keypoints', 'max_classes', 'max_overlaps', 'bbox_targets']:
                if key in entry:
                    entry[key] = entry[key][valid_inds]
            entry['box_to_gt_ind_map'] = np.array(list(range(len(valid_inds))))
    im_scales = [1.0]
    ims = [im]
    return ims, im_scales


def get_im_blob_sizes(im_shape, target_sizes, max_size):
    """Calculate im blob size for multiple target_sizes given original im shape
    """
    im_size_min = np.min(im_shape)
    im_size_max = np.max(im_shape)
    im_sizes = []
    for target_size in target_sizes:
        im_scale = get_target_scale(im_size_min, im_size_max, target_size, max_size)
        im_sizes.append(np.round(im_shape * im_scale))
    return np.array(im_sizes)


def get_target_scale(im_size_min, im_size_max, target_size, max_size):
    """Calculate target resize scale
    """
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def zeros(shape, int32=False):
    """Return a blob of all zeros of the given shape with the correct float or
    int data type.
    """
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Return a blob of all ones of the given shape with the correct float or
    int data type.
    """
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
