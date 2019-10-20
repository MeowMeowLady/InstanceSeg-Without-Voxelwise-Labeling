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

"""Representation of the standard COCO json dataset format.

When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
from six.moves import cPickle as pickle
import logging
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
import utils.env as envu
envu.set_up_matplotlib()
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

import utils.boxes_3d as box_utils_3d
from core.config import cfg
from utils.timer import Timer
from .dataset_catalog import ANN_DIR
from .dataset_catalog import DATASETS
from .dataset_catalog import IM_DIR
from .dataset_catalog import IM_PREFIX

logger = logging.getLogger(__name__)
Pi = 3.14159

class SomaDataset(object):
    """A class representing my soma dataset."""

    def __init__(self, name):

        assert name in DATASETS.keys(), \
            'Unknown dataset name: {}'.format(name)
        assert os.path.exists(DATASETS[name][IM_DIR]), \
            'Image directory \'{}\' not found'.format(DATASETS[name][IM_DIR])
        assert os.path.exists(DATASETS[name][ANN_DIR]), \
            'Annotation directory \'{}\' not found'.format(DATASETS[name][ANN_DIR])
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.label_directory = DATASETS[name][ANN_DIR]
        self.image_prefix = (
            '' if IM_PREFIX not in DATASETS[name] else DATASETS[name][IM_PREFIX]
        )
        self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = [1]
        categories = ['soma']
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self._init_keypoints()


    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def valid_cached_keys(self):
        """ Can load following key-ed values from the cached roidb file

        'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
        so we don't need to overwrite it again.
        """
        keys = ['boxes', 'segms', 'gt_classes', 'seg_volumes', 'gt_overlaps',
                'is_crowd', 'box_to_gt_ind_map', 'slices', 'width', 'height']
        if self.keypoints is not None:
            keys += ['gt_keypoints', 'has_visible_keypoints']
        return keys

    def get_roidb(
            self,
            gt=False,
            proposal_file=None,
            min_proposal_size=2,
            proposal_limit=-1,
            crowd_filter_thresh=0,
            phase = 'train'
        ):

        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'

        # roidb is a list and each item is a dictionary with keys including such as width, height, id, filename
        if phase == 'train':
            fid_list = open(os.path.join(self.image_directory[:-5], 'train.txt'), 'r')
        elif phase == 'test':
            fid_list = open(os.path.join(self.image_directory[:-5], 'test.txt'), 'r')
        elif phase == 'valid':
            fid_list = open(os.path.join(self.image_directory[:-5], 'valid.txt'), 'r')
        img_list = fid_list.readlines()
        fid_list.close()
        roidb = []
        for t in img_list:
            roidb.append({'file_name': t.rstrip()})

        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            cache_filepath = os.path.join(self.cache_path, self.name+'_gt_roidb.pkl')
            if os.path.exists(cache_filepath) and not cfg.DEBUG:
                self.debug_timer.tic()
                self._add_gt_from_cache(roidb, cache_filepath)
                logger.debug(
                    '_add_gt_from_cache took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
            else:
                self.debug_timer.tic()
                for entry in roidb:
                    self._add_gt_annotations(entry)
                logger.debug(
                    '_add_gt_annotations took {:.3f}s'.
                    format(self.debug_timer.toc(average=False))
                )
                if not cfg.DEBUG:
                    with open(cache_filepath, 'wb') as fp:
                        pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
                    logger.info('Cache ground truth roidb to %s', cache_filepath)
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh
            )
            logger.debug(
                '_add_proposals_from_file took {:.3f}s'.
                format(self.debug_timer.toc(average=False))
            )
        _add_class_assignments(roidb)

        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        im_path = os.path.join(
            self.image_directory, entry['file_name'], entry['file_name'] + '.tif'
        )
        assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
        entry['image'] = im_path
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 6), dtype=np.float32)
        entry['segms'] = np.empty((0, 4), dtype=np.float32)
        entry['gt_classes'] = np.empty(0, dtype=np.int32)
        entry['seg_volumes'] = np.empty(0, dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(
            np.empty((0, self.num_classes), dtype=np.float32)
        )
        entry['need_crop'] = cfg.TRAIN.NEED_CROP
        entry['is_crowd'] = np.empty(0, dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty(0, dtype=np.int32)
        entry['slices'] = np.empty(0, dtype=np.float32)
        entry['height'] = np.empty(0, dtype=np.float32)
        entry['width'] = np.empty(0, dtype=np.float32)


    def _load_ann_objs(self, file_name):
        ann_file_path = os.path.join(self.label_directory, file_name+'.txt')
        fid_ann = open(ann_file_path, 'r')
        ann = fid_ann.readlines()[1:]
        objs = []
        slices, height, width = cfg.TRAIN.IM_SIZE[:]
        for a in ann:
            part_a = a.rstrip().split(' ')
            pos_x = np.minimum(width - 1,  int(part_a[0]))
            pos_y = np.minimum(height - 1, int(part_a[1]))
            pos_z = np.minimum(slices - 1, int(part_a[2]))
            radius = int(part_a[3])
            obj = {}
            expand_dimeter = radius*2*(1. + cfg.TRAIN.RADIUS_EXP_RATIO)

            x1 = int(np.maximum(pos_x - expand_dimeter/2.0, 0.0))
            y1 = int(np.maximum(pos_y - expand_dimeter/2.0, 0.0))
            z1 = int(np.maximum(pos_z - expand_dimeter/2.0, 0.0))
            w = int(np.minimum(pos_x + expand_dimeter/2.0, width-1)) - x1 + 1
            h = int(np.minimum(pos_y + expand_dimeter/2.0, height-1)) - y1 + 1
            s = int(np.minimum(pos_z + expand_dimeter/2.0, slices-1)) - z1 + 1
            obj['bbox'] = [x1, y1, z1, w, h, s]
            obj['category_id'] = self.category_to_id_map['soma']
            obj['segmentation'] = [pos_x, pos_y, pos_z, radius]
            obj['volume'] = 4./3.*Pi*radius**3
            obj['iscrowd'] = 0
            obj['ignore'] = 0
            objs.append(obj)
        return  objs


    def _add_gt_annotations(self, entry):
        """Add ground truth annotation metadata to an roidb entry."""
        objs = self._load_ann_objs(entry['file_name'])
        # Sanitize bboxes -- some are invalid
        valid_objs = []

        width = entry['width'] = cfg.TRAIN.IM_SIZE[2]
        height = entry['height'] = cfg.TRAIN.IM_SIZE[1]
        slices = entry['slices'] = cfg.TRAIN.IM_SIZE[0]
        for obj in objs:
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form (x1, y1, z1, w, h, s) to (x1, y1, z1, x2, y2, z2)
            x1, y1, z1, x2, y2, z2 = box_utils_3d.xyzwhs_to_xyzxyz(obj['bbox'])
            x1, y1, z1, x2, y2, z2 = box_utils_3d.clip_xyzxyz_to_image(
                x1, y1, z1, x2, y2, z2, slices, height, width
            )
            # Require non-zero seg volume and more than 1x1 box size
            if obj['volume'] > 0 and x2 > x1 and y2 > y1 and z2 > z1:
                obj['clean_bbox'] = [x1, y1, z1, x2, y2, z2]
                valid_objs.append(obj)
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 6), dtype=entry['boxes'].dtype)
        segms = np.zeros((num_valid_objs, 4), dtype=entry['segms'].dtype)
        gt_classes = np.zeros(num_valid_objs, dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype
        )
        seg_volumes = np.zeros(num_valid_objs, dtype=entry['seg_volumes'].dtype)
        is_crowd = np.zeros(num_valid_objs, dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            num_valid_objs, dtype=entry['box_to_gt_ind_map'].dtype
        )

        for ix, obj in enumerate(valid_objs):
            cls = obj['category_id']
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_volumes[ix] = obj['volume']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            segms[ix, :] = obj['segmentation']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['segms'] = np.append(entry['segms'], segms, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_volumes'] = np.append(entry['seg_volumes'], seg_volumes)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map
        )


    def _add_gt_from_cache(self, roidb, cache_filepath):
        """Add ground truth annotation metadata from cached file."""
        logger.info('Loading cached gt_roidb from %s', cache_filepath)
        with open(cache_filepath, 'rb') as fp:
            cached_roidb = pickle.load(fp)

        assert len(roidb) == len(cached_roidb)

        for entry, cached_entry in zip(roidb, cached_roidb):
            values = [cached_entry[key] for key in self.valid_cached_keys]
            boxes, segms, gt_classes, seg_volumes, gt_overlaps, is_crowd, box_to_gt_ind_map, \
            slices, height, width = values[:10]
            entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
            entry['segms'] = np.append(entry['segms'], segms, axis=0)
            entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
            entry['seg_volumes'] = np.append(entry['seg_volumes'], seg_volumes)
            entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
            entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
            entry['box_to_gt_ind_map'] = np.append(
                entry['box_to_gt_ind_map'], box_to_gt_ind_map
            )
            entry['slices'] = np.append(entry['slices'], slices)
            entry['height'] = np.append(entry['height'], height)
            entry['width'] = np.append(entry['width'], width)


    def _add_proposals_from_file(
        self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh
    ):
        """Add proposals from a proposals file to an roidb."""
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils_3d.clip_boxes_to_image(
                boxes, entry['height'], entry['width']
            )
            keep = box_utils_3d.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils_3d.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self):
        """Initialize COCO keypoint information."""
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if cfg.KRCNN.NUM_KEYPOINTS != -1:
                assert cfg.KRCNN.NUM_KEYPOINTS == self.num_keypoints, \
                    "number of keypoints should equal when using multiple datasets"
            else:
                cfg.KRCNN.NUM_KEYPOINTS = self.num_keypoints
            self.keypoint_flip_map = {
                'left_eye': 'right_eye',
                'left_ear': 'right_ear',
                'left_shoulder': 'right_shoulder',
                'left_elbow': 'right_elbow',
                'left_wrist': 'right_wrist',
                'left_hip': 'right_hip',
                'left_knee': 'right_knee',
                'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        """Return ground truth keypoints."""
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps


def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        box_to_gt_ind_map = -np.ones(
            num_boxes, dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils_3d.bbox_overlaps_3d(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros(num_boxes, dtype=entry['gt_classes'].dtype)
        )
        entry['seg_volumes'] = np.append(
            entry['seg_volumes'],
            np.zeros(num_boxes, dtype=entry['seg_volumes'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros(num_boxes, dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils_3d.xyzxyz_to_xyzwhs(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils_3d.xyzxyz_to_xyzwhs(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    """Sort proposals by the specified id field."""
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]
