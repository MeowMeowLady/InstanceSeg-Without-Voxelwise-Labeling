#ifndef _ROI_ALIGN_KERNEL_3D
#define _ROI_ALIGN_KERNEL_3D

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIAlignForward_3d(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int slices, const int height, const int width,
    const int channels, const int aligned_slices, const int aligned_height, const int aligned_width,
    const int sampling_ratio, const float* bottom_rois, float* top_data);

int ROIAlignForwardLaucher_3d(
    const float* bottom_data, const float spatial_scale, const int num_rois,
    const int slices, const int height, const int width, const int channels,
    const int aligned_slices, const int aligned_height, const int aligned_width,
    const int sampling_ratio, const float* bottom_rois,
    float* top_data, cudaStream_t stream);

__global__ void ROIAlignBackward_3d(const int nthreads, const float* top_diff,
    const float spatial_scale, const int slices, const int height, const int width, const int channels,
    const int aligned_slices, const int aligned_height, const int aligned_width, const int sampling_ratio,
    float* bottom_diff, const float* bottom_rois);

int ROIAlignBackwardLaucher_3d(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int slices, const int height, const int width, const int channels, const int aligned_slices, const int aligned_height,
    const int aligned_width,  const int sampling_ratio, const float* bottom_rois,
    float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

