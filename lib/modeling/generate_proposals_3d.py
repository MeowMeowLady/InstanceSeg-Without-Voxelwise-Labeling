import logging
import numpy as np

from torch import nn

from core.config import cfg
# import utils.boxes as box_utils
import utils.boxes_3d as box_utils_3d
logger = logging.getLogger(__name__)


class GenerateProposalsOp_3d(nn.Module):
    def __init__(self, anchors, spatial_scale):
        super().__init__()
        self._anchors = anchors
        self._num_anchors = self._anchors.shape[0]
        self._feat_stride = 1. / spatial_scale

    def forward(self, rpn_cls_prob, rpn_bbox_pred, im_info):
        """Op for generating RPN porposals.

        blobs_in:
          - 'rpn_cls_probs': 5D tensor of shape (N, A, S, H, W), where N is the
            number of minibatch images, A is the number of anchors per
            locations, and (S, H, W) is the spatial size of the prediction grid.
            Each value represents a "probability of object" rating in [0, 1].
          - 'rpn_bbox_pred': 5D tensor of shape (N, 4 * A, S, H, W) of predicted
            deltas for transformation anchor boxes into RPN proposals.
          - 'im_info': 2D tensor of shape (N, 4) where the four columns encode
            the input image's [slices, height, width, scale]. Slices, height and
            width are for the input to the network, not the original image; scale
            is the scale factor used to scale the original image to the network
            input size.

        blobs_out:
          - 'rpn_rois': 2D tensor of shape (R, 7), for R RPN proposals where the
            five columns encode [batch ind, x1, y1, z1, x2, y2, z2]. The boxes are
            w.r.t. the network input, which is a *scaled* version of the
            original image; these proposals must be scaled by 1 / scale (where
            scale comes from im_info; see above) to transform it back to the
            original input image coordinate system.
          - 'rpn_roi_probs': 1D tensor of objectness probability scores
            (extracted from rpn_cls_probs; see above).
        """
        # 1. for each location i in a (S, H, W) grid:
        #      generate A anchor boxes centered on cell i
        #      apply predicted bbox deltas to each of the A anchors at cell i
        # 2. clip predicted boxes to image
        # 3. remove predicted boxes with either height or width or slices < threshold
        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take the top pre_nms_topN proposals before NMS
        # 6. apply NMS with a loose threshold (0.7) to the remaining proposals
        # 7. take after_nms_topN proposals after NMS
        # 8. return the top proposals
        
        """Type conversion"""
        # predicted probability of fg object for each RPN anchor
        scores = rpn_cls_prob.data.cpu().numpy()
        # predicted achors transformations
        bbox_deltas = rpn_bbox_pred.data.cpu().numpy()
        # input image (height, width, scale), in which scale is the scale factor
        # applied to the original dataset image to get the network input image
        im_info = im_info.data.cpu().numpy()

        # 1. Generate proposals from bbox deltas and shifted anchors
        slices, height, width = scores.shape[-3:]
        # Enumerate all shifted positions on the (S, H, W) grid
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_z = np.arange(0, slices) * self._feat_stride
        shift_z, shift_y, shift_x = np.meshgrid(shift_z, shift_y, shift_x, indexing= 'ij', copy=False)


        # Convert to (K, 6), K=S*H*W, where the columns are (dx, dy, dz, dx, dy, dz)
        # shift pointing to each grid location
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_z.ravel(), shift_x.ravel(),
                            shift_y.ravel(), shift_z.ravel())).transpose()

        # Broacast anchors over shifts to enumerate all anchors at all positions
        # in the (S, H, W) grid:
        #   - add A anchors of shape (1, A, 6) to
        #   - K shifts of shape (K, 1, 6) to get
        #   - all shifted anchors of shape (K, A, 6)
        #   - reshape to (K*A, 6) shifted anchors
        num_images = scores.shape[0]
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = self._anchors[np.newaxis, :, :] + shifts[:, np.newaxis, :]
        all_anchors = all_anchors.reshape((K * A, 6))
        # all_anchors = torch.from_numpy(all_anchors).type_as(scores)

        rois = np.empty((0, 7), dtype=np.float32)
        roi_probs = np.empty((0, 1), dtype=np.float32)
        for im_i in range(num_images):
            im_i_boxes, im_i_probs, scores_keep_idx = self.proposals_for_one_image(
                im_info[im_i, :], all_anchors, bbox_deltas[im_i, :, :, :],
                scores[im_i, :, :, :, :])
            batch_inds = im_i * np.ones(
                (im_i_boxes.shape[0], 1), dtype=np.float32)
            im_i_rois = np.hstack((batch_inds, im_i_boxes))
            rois = np.append(rois, im_i_rois, axis=0)
            roi_probs = np.append(roi_probs, im_i_probs, axis=0)

        return rois, roi_probs, scores_keep_idx # Note: ndarrays

    def proposals_for_one_image(self, im_info, all_anchors, bbox_deltas, scores):
        # Get mode-dependent configuration
        cfg_key = 'TRAIN' if self.training else 'TEST'
        pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
        nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
        min_size = cfg[cfg_key].RPN_MIN_SIZE
        # print('generate_proposals:', pre_nms_topN, post_nms_topN, nms_thresh, min_size)

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:
        #   - bbox deltas will be (6 * A, S, H, W) format from conv output
        #   - transpose to (S, H, W, 6 * A)
        #   - reshape to (S * H * W * A, 6) where rows are ordered by (S, H, W, A)
        #     in slowest to fastest order to match the enumerated anchors
        bbox_deltas = bbox_deltas.transpose((1, 2, 3, 0)).reshape((-1, 6))

        # Same story for the scores:
        #   - scores are (A, S, H, W) format from conv output
        #   - transpose to (S, H, W, A)
        #   - reshape to (S * H * W * A, 1) where rows are ordered by (S, H, W, A)
        #     to match the order of anchors and bbox_deltas
        # save the index for all the kept score in score map
        scores = scores.transpose((1, 2, 3, 0)).reshape((-1, 1))

        # print('pre_nms:', bbox_deltas.shape, scores.shape)

        # 4. sort all (proposal, score) pairs by score from highest to lowest
        # 5. take top pre_nms_topN (e.g. 6000)
        if pre_nms_topN <= 0 or pre_nms_topN >= len(scores):
            order = np.argsort(-scores.squeeze())
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(-scores.squeeze(),
                                   pre_nms_topN)[:pre_nms_topN]
            order = np.argsort(-scores[inds].squeeze())
            order = inds[order]
        bbox_deltas = bbox_deltas[order, :]
        all_anchors = all_anchors[order, :]
        scores = scores[order]

        # Transform anchors into proposals via bbox transformations
        proposals = box_utils_3d.bbox_transform_3d(all_anchors, bbox_deltas,
                                             (1.0, 1.0, 1.0, 1.0, 1.0, 1.0))

        # 2. clip proposals to image (may result in proposals with zero area
        # that will be removed in the next step)
        proposals = box_utils_3d.clip_tiled_boxes_3d(proposals, im_info[:3])

        # 3. remove predicted boxes with either height or width or slice < min_size
        keep = _filter_boxes_3d(proposals, min_size, im_info)
        proposals = proposals[keep, :]
        scores = scores[keep]
        order_kept = order[keep]

        # print('pre_nms:', proposals.shape, scores.shape)

        # 6. apply loose nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        if nms_thresh > 0:
            keep = box_utils_3d.nms_3d(np.hstack((proposals, scores)), nms_thresh)
            # print('nms keep:', keep.shape)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores = scores[keep]
            order_kept = order_kept[keep]
        scores_keep_idx = order_kept # the order of index is [1, S, H, W, A]
        # print('final proposals:', proposals.shape, scores.shape)
        return proposals, scores, scores_keep_idx


def _filter_boxes_3d(boxes, min_size, im_info):
    """Only keep boxes with all three sides >= min_size and center within the image.
  """
    # Scale min_size to match image scale
    min_size *= im_info[3]
    ss = boxes[:, 3] - boxes[:, 0] + 1
    #hs = boxes[:, 3] - boxes[:, 1] + 1
    x_ctr = boxes[:, 0] + ss / 2.
    y_ctr = boxes[:, 1] + ss / 2.
    z_ctr = boxes[:, 2] + ss / 2.
    keep = np.where((ss >= min_size) & (x_ctr < im_info[2]) &
                    (y_ctr < im_info[1]) & (z_ctr < im_info[0]))[0]
    return keep

