from ..functions.roi_align_3d import RoIAlignFunction_3d
from torch.nn.modules.module import Module
from torch.nn.functional import avg_pool3d, max_pool3d


class RoIAlign_3d(Module):
    def __init__(self, aligned_slices, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlign_3d, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.aligned_slices = int(aligned_slices)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        return RoIAlignFunction_3d(self.aligned_slices, self.aligned_height, self.aligned_width,
                                self.spatial_scale, self.sampling_ratio)(features, rois)

class RoIAlignAvg_3d(Module):
    def __init__(self, aligned_slices, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignAvg_3d, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.aligned_slices = int(aligned_slices)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x =  RoIAlignFunction_3d(self.aligned_slices+1, self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale, self.sampling_ratio)(features, rois)
        return avg_pool3d(x, kernel_size=2, stride=1)

class RoIAlignMax_3d(Module):
    def __init__(self, aligned_slices, aligned_height, aligned_width, spatial_scale, sampling_ratio):
        super(RoIAlignMax_3d, self).__init__()

        self.aligned_width = int(aligned_width)
        self.aligned_height = int(aligned_height)
        self.aligned_slices = int(aligned_slices)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)

    def forward(self, features, rois):
        x =  RoIAlignFunction_3d(self.aligned_slices+1, self.aligned_height+1, self.aligned_width+1,
                                self.spatial_scale, self.sampling_ratio)(features, rois)
        return max_pool3d(x, kernel_size=2, stride=1)

