import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage import io
from libtiff import TIFF
import os

DEBUG = False

class BilinearInterpolation2d(nn.Module):
    """Bilinear interpolation in space of scale.

    Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale

    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        if DEBUG:
            print(bil_filt)

        kernel = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(in_channels), range(out_channels), :, :] = bil_filt

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                         stride=self.up_scale, padding=self.padding)

        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)


class BilinearInterpolation3d(nn.Module):
    """Bilinear interpolation in space of scale.

    Takes input of NxKxSxHxW and outputs NxKx(sS)x(sH)x(sW), where s:= up_scale

    Adapted from the CVPR'15 FCN code.
    See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
    """
    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        assert in_channels == out_channels
        assert up_scale % 2 == 0, 'Scale should be even'
        self.in_channes = in_channels
        self.out_channels = out_channels
        self.up_scale = int(up_scale)
        self.padding = up_scale // 2

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor) *
                    (1 - abs(og[2] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype=np.float32
        )
        kernel[range(in_channels), range(out_channels), :, :, :] = bil_filt

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                         stride=self.up_scale, padding=self.padding)

        self.upconv.weight.data.copy_(torch.from_numpy(kernel))
        self.upconv.bias.data.fill_(0)
        self.upconv.weight.requires_grad = False
        self.upconv.bias.requires_grad = False

    def forward(self, x):
        return self.upconv(x)

if __name__ == '__main__':
    BI = BilinearInterpolation3d(1,1,2)
    # tmp_x = np.array([[[1, 2, 3], [4, 5, 6], [7,8,9]], [[1,1,1],[2,2,2],[3,3,3]]], dtype='float32')
    tmp_x = io.imread('/media/dongmeng/Hulk/dataset/total_d8/image/0006/0006.tif').astype('float32')
    tmp_x = tmp_x[np.newaxis, np.newaxis, :, :, :]
    in_x = torch.tensor(tmp_x)
    out_x = BI.forward(in_x)
    out_x = out_x.numpy()
    image3D = TIFF.open(os.path.join('/media/dongmeng/Hulk/dataset/total_d8/image/0006/1', 'up.tif'), mode='w')
    out_x = np.squeeze(out_x)
    for k in range(out_x.shape[0]):
        image3D.write_image(out_x[k, :].astype('uint16'), compression='lzw', write_rgb=False)
    image3D.close()
    #print(out_x)