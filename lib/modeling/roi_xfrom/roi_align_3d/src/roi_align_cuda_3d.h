int roi_align_forward_cuda_3d(int aligned_slices, int aligned_height, int aligned_width, float spatial_scale, int sampling_ratio,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int roi_align_backward_cuda_3d(int aligned_slices, int aligned_height, int aligned_width, float spatial_scale, int sampling_ratio,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad);
