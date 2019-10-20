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
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

# array([[ -83.,  -39.,  100.,   56.],
#        [-175.,  -87.,  192.,  104.],
#        [-359., -183.,  376.,  200.],
#        [ -55.,  -55.,   72.,   72.],
#        [-119., -119.,  136.,  136.],
#        [-247., -247.,  264.,  264.],
#        [ -35.,  -79.,   52.,   96.],
#        [ -79., -167.,   96.,  184.],
#        [-167., -343.,  184.,  360.]])


def generate_anchors(
    stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)
):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )

# $add by Meng$
def generate_anchors_3d(
    stride=8, sizes=(12, 20, 30), aspect_ratios=np.array([[1., 1.], [1., 0.5]])
):
    """Generates a matrix of anchor boxes in (x1, y1, z1, x2, y2, z2) format. Anchors
    are centered on stride / 2, have (approximate) cubic volume of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_anchors_3d(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )
# $add by Meng$

def _generate_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors
# $add by Meng$
def _generate_anchors_3d(base_size, scales, aspect_ratios):
    """Generate anchor (reference) boxes by enumerating aspect ratios X
    scales wrt a reference (0, 0, 0, base_size - 1, base_size - 1, base_size - 1) box.
    """
    anchor = np.array([1, 1, 1, base_size, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum_3d(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum_3d(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors
# $add by Meng$

def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

# $add by Meng$
def _sctrs_3d(anchor):
    """Return width, height, slice, x center, and y center for an anchor (window)."""
    w = anchor[3] - anchor[0] + 1
    h = anchor[4] - anchor[1] + 1
    s = anchor[5] - anchor[2] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    z_ctr = anchor[2] + 0.5 * (s - 1)
    return w, h, s, x_ctr, y_ctr, z_ctr
# $add by Meng$

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors

# $add by Meng$
def _mkanchors_3d(w, h, s, x_ctr, y_ctr, z_ctr):
    """Given a vector of side(s) around a center
    (x_ctr, y_ctr, z_ctr), output a set of anchors (boxes).
    """
    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    s = s[:, np.newaxis]

    anchors = np.hstack(
        (
            x_ctr - 0.5 * (w - 1),
            y_ctr - 0.5 * (h - 1),
            z_ctr - 0.5 * (s - 1),
            x_ctr + 0.5 * (w - 1),
            y_ctr + 0.5 * (h - 1),
            z_ctr + 0.5 * (s - 1)
        )
    )
    return anchors
# $add by Meng$

def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# $add by Meng$
def _ratio_enum_3d(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, s, x_ctr, y_ctr, z_ctr = _sctrs_3d(anchor)
    size = w*h*s
    size_ratios = size / (ratios[:, 0]*ratios[:, 1]) # only support ratio=1.0
    ws = np.round(size_ratios**(1./3))
    hs = np.round(ws * ratios[:, 0])
    ss = np.round(ws * ratios[:, 1])
    anchors = _mkanchors_3d(ws, hs, ss, x_ctr, y_ctr, z_ctr)
    return anchors
# $add by Meng$

def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

# $add by Meng$
def _scale_enum_3d(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, s, x_ctr, y_ctr, z_ctr = _sctrs_3d(anchor)
    ws = w * scales
    hs = h * scales
    ss = s * scales
    anchors = _mkanchors_3d(ws, hs, ss, x_ctr, y_ctr, z_ctr)
    return anchors
# $add by Meng$

if __name__ == '__main__':
    tmp = generate_anchors_3d()
    print(tmp)