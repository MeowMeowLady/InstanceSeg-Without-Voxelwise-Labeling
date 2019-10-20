from types import MethodType

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.misc import imresize
from prm.peak_backprop_3d import pr_conv3d, pr_deconv3d
from prm.peak_stimulation_3d import peak_stimulation_3d
from modeling.model_builder import Generalized_RCNN
from core.config import cfg
from core.test import box_results_with_nms_and_limit
from numpy import *
import utils.boxes_3d as box_utils_3d

class PeakResponseMapping_3d(Generalized_RCNN):

    def __init__(self, *args, **kargs):
        # super(PeakResponseMapping_3d, self).__init__(*args)
        super().__init__()

        self.inferencing = True
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', False)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, s, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, s * h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1, 1)
    
    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, s, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, s * h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1, 1)
    
    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, s, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, s * h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv3d, module)
            if isinstance(module, nn.ConvTranspose3d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_deconv3d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward
            if isinstance(module, nn.ConvTranspose3d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def instance_nms(self, instance_list, threshold=0.3, merge_peak_response=True):
        pass

    def instance_seg(self, class_response_maps, peak_list, peak_response_maps, retrieval_cfg):        
        pass
    
    def forward(self, data, im_info, im_scale = 1.0, roidb=None, peak_threshold=0.1, retrieval_cfg=None):
        assert data.dim() == 5, 'PeakResponseMapping layer only supports batch mode.'
        if self.inferencing:
            data.requires_grad_()

        # detection network forwarding
        # return_dict = super(PeakResponseMapping_3d, self).forward(data, im_info)
        # rpn_ret = return_dict['rpn_ret']

        blob_conv = self.Conv_Body(data)
        rpn_ret = self.RPN(blob_conv, im_info, roidb)
        scores_keep_idx = rpn_ret['scores_keep_idx']  # 1, S, H, W, A
        class_response_maps = rpn_ret['class_response_maps']

        if cfg.MODEL.RPN_ONLY:
            boxes = rpn_ret['rpn_rois'][:, 1:7]/ im_scale
        else:
            box_feat = self.Box_Head(blob_conv, rpn_ret)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
            rois = rpn_ret['rpn_rois']
            # unscale back to raw image space
            boxes = rois[:, 1:7] / im_scale
            # cls prob (activations after softmax)
            scores = cls_score.data.cpu().numpy().squeeze()
            # In case there is 1 proposal
            if size(scores) == 1:
                scores = np.reshape(scores, (1, 1))
            else:
                scores = scores.reshape([-1, scores.shape[-1]])

            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data.cpu().numpy().squeeze()
            # In case there is 1 proposal
            box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
            input_shape = im_info[0][:3]

            pred_boxes = box_utils_3d.bbox_transform_3d(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
            pred_boxes = box_utils_3d.clip_tiled_boxes_3d(pred_boxes, input_shape)

            scores, boxes, _, scores_keep_idx = box_results_with_nms_and_limit(scores, pred_boxes, scores_keep_idx)
            scores_keep_idx = scores_keep_idx[1]

        B, A, S, H, W = class_response_maps.shape

        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.upsample(class_response_maps, scale_factor=self.sub_pixel_locating_factor, mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation_3d(class_response_maps, win_size=self.win_size, peak_filter=self.peak_filter)
        else:
            peak_list = np.zeros((len(scores_keep_idx), 5), dtype=np.int)
            for i, idx in enumerate(scores_keep_idx):
                peak_list[i, :] = np.unravel_index(idx, (B, S, H, W, A))
            peak_list = torch.from_numpy(np.hstack((peak_list[:, 0, np.newaxis], peak_list[:, 4, np.newaxis], peak_list[:, 1:4])))
            aggregation = None

        if self.inferencing:
            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps
            
            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation_3d(class_response_maps, return_aggregation=False, win_size=self.win_size, peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            dets = np.zeros((0, 7))
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            for idx in range(len(scores_keep_idx)):
                if cfg.MODEL.RPN_ONLY:
                    peak_val = class_response_maps[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3], peak_list[idx, 4]].detach().cpu().numpy()
                else:
                    peak_val = scores[idx]
                if peak_val > peak_threshold:
                    dets = np.append(dets, np.append(boxes[idx, :], peak_val)[np.newaxis, :], axis=0)
                    grad_output.zero_()
                    # starting from the peak
                    grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3], peak_list[idx, 4]] = 1.
                    if data.grad is not None:
                        data.grad.zero_()
                    class_response_maps.backward(grad_output, retain_graph=True)
                    prm = data.grad.detach().sum(1).clone().clamp(min=0)
                    peak_response_maps.append(prm / prm.sum())
                    valid_peak_list.append(peak_list[idx, :])
            
            # return results
            class_response_maps = class_response_maps.detach()
            if not aggregation is None:
                aggregation = aggregation.detach()

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list).cuda()
                peak_response_maps = torch.cat(peak_response_maps, 0).cuda()
                dets = torch.from_numpy(dets).cuda()
                if retrieval_cfg is None:
                    # classification confidence scores, class-aware and instance-aware visual cues
                    return aggregation, class_response_maps, valid_peak_list, peak_response_maps, dets
                else:
                    # instance segmentation using build-in proposal retriever
                    return self.instance_seg(class_response_maps, valid_peak_list, peak_response_maps, retrieval_cfg)
            else:
                return None, None, None, None, None
        else:
            # classification confidence scores
            return aggregation

    def train(self, mode=True):
        super(PeakResponseMapping_3d, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping_3d, self).train(False)
        self._patch()
        self.inferencing = True
        return self
