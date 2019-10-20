#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "roi_align_kernel_3d.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

    /*** Forward ***/

    __device__ float trilinear_interpolate(const float* bottom_data, const int slices, const int height,
                           const int width, float z, float y, float x, const int index /* index for debug only*/) {
            // deal with cases that inverse elements are out of feature map boundary
            if (z < -1.0 || z > slices || y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                return 0;
            }
            if (z <= 0) {
                z = 0;
            }
            if (y <= 0) {
                y = 0;
            }
            if (x <= 0) {
                x = 0;
            }

            int z_low = (int)z;
            int y_low = (int)y;
            int x_low = (int)x;
            int z_high;
            int y_high;
            int x_high;

            if (z_low >= slices - 1) {
                z_high = z_low = slices - 1;
                z = (float)z_low;
            } else {
                z_high = z_low + 1;
            }
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (float)y_low;
            } else {
                y_high = y_low + 1;
            }
            
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (float)x_low;
            } else {
                x_high = x_low + 1;
            }

            float lz = z - z_low;
            float ly = y - y_low;
            float lx = x - x_low;
            float hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;
            float v1 = bottom_data[z_low * (width * height) + y_low * width + x_low];
            float v2 = bottom_data[z_low * (width * height) + y_low * width + x_high];
            float v3 = bottom_data[z_low * (width * height) + y_high * width + x_low];
            float v4 = bottom_data[z_low * (width * height) + y_high * width + x_high];
            float v5 = bottom_data[z_high * (width * height) + y_low * width + x_low];
            float v6 = bottom_data[z_high * (width * height) + y_low * width + x_high];
            float v7 = bottom_data[z_high * (width * height) + y_high * width + x_low];
            float v8 = bottom_data[z_high * (width * height) + y_high * width + x_high];

            float w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
            float w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;

            float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4 + w5 * v5 + w6 * v6 + w7 * v7 + w8 *v8);

            return val;
        }

    __global__ void ROIAlignForward_3d(const int nthreads, const float* bottom_data, const float spatial_scale, const int slices,
                                    const int height, const int width, const int channels, const int aligned_slices, const int aligned_height,
                                    const int aligned_width, const int sampling_ratio, const float* bottom_rois, float* top_data)
                                   {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ps, ph, pw) is an element in the aligned output
            int ps = index % aligned_slices;
            int pw = (index / aligned_slices) % aligned_width;
            int ph = (index / aligned_slices / aligned_width) % aligned_height;
            int c  = (index / aligned_slices / aligned_width / aligned_height) % channels;
            int n  = index / aligned_slices / aligned_width / aligned_height / channels;

            const float* offset_bottom_rois = bottom_rois + n * 7;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            float roi_start_w = offset_bottom_rois[1] * spatial_scale;
            float roi_start_h = offset_bottom_rois[2] * spatial_scale;
            float roi_start_s = offset_bottom_rois[3] * spatial_scale;
            float roi_end_w = offset_bottom_rois[4] * spatial_scale;
            float roi_end_h = offset_bottom_rois[5] * spatial_scale;
            float roi_end_s = offset_bottom_rois[6] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_slices = fmaxf(roi_end_s - roi_start_s, 1.f);
            float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
            float bin_size_s = roi_slices / aligned_slices;
            float bin_size_h = roi_height / aligned_height;
            float bin_size_w = roi_width / aligned_width;

            const float* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * slices * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_s = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_slices / aligned_slices);
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const float count = roi_bin_grid_s * roi_bin_grid_h * roi_bin_grid_w; // e.g. = 8
            float output_val = 0.;
            for (int iz = 0; iz < roi_bin_grid_s; iz++)
            {
                const float z = roi_start_s + ps * bin_size_s +
                           (iz + .5f) * bin_size_s / roi_bin_grid_s;
                for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
                {
                    const float y = roi_start_h + ph * bin_size_h +
                           (iy + .5f) * bin_size_h / roi_bin_grid_h;  // e.g., 0.5, 1.5
                    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                        const float x = roi_start_w + pw * bin_size_w +
                        (ix + .5f) * bin_size_w / roi_bin_grid_w;

                        float val = trilinear_interpolate(
                             offset_bottom_data, slices, height, width, z, y, x, index);
                        output_val += val;
                    }
                }
            }

            output_val /= count;

            top_data[index] = output_val;
        }
    }

    int ROIAlignForwardLaucher_3d(const float* bottom_data, const float spatial_scale, const int num_rois, const int slices,
                               const int height, const int width, const int channels, const int aligned_slices,
                               const int aligned_height,  const int aligned_width,  const int sampling_ratio,
                               const float* bottom_rois, float* top_data, cudaStream_t stream) {
        const int kThreadsPerBlock = 512;
        const int output_size = num_rois * aligned_slices * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignForward_3d<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, bottom_data, spatial_scale, slices, height, width, channels, aligned_slices,
          aligned_height, aligned_width, sampling_ratio, bottom_rois, top_data);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

    /*** Backward ***/
    inline __device__ float gpu_atomic_add(const float val, float* address);
    inline __device__ float gpu_atomic_add(const float val, float* address) {
        return atomicAdd(address, val);
    }

    __device__ void trilinear_interpolate_gradient(const int slices, const int height, const int width,
                                                  float z, float y, float x,
                                                  float& w1, float& w2, float& w3, float& w4,
                                                  float& w5, float& w6, float& w7, float& w8,
                                                  int& x_low, int& x_high, int& y_low, int& y_high,int& z_low, int& z_high,
                                                  const int index /* index for debug only*/) {
        // deal with cases that inverse elements are out of feature map boundary
        if (z < -0.1 || z > slices || y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            w1 = w2 = w3 = w4 = w5 = w6 = w7 = w8 = 0.;
            x_low = x_high = y_low = y_high = z_low = z_high = -1;
            return;
        }
        if (z <= 0) {
            z = 0;
        }
        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        z_low = (int)z;
        y_low = (int)y;
        x_low = (int)x;

        if (z_low >= slices - 1) {
            z_high = z_low = slices - 1;
            z = (float)z_low;
        } else {
            z_high = z_low + 1;
        }

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (float)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (float)x_low;
        } else {
            x_high = x_low + 1;
        }

        float lz = z - z_low;
        float ly = y - y_low;
        float lx = x - x_low;
        float hz = 1. - lz, hy = 1. - ly, hx = 1. - lx;

        w1 = hz * hy * hx, w2 = hz * hy * lx, w3 = hz * ly * hx, w4 = hz * ly * lx;
        w5 = lz * hy * hx, w6 = lz * hy * lx, w7 = lz * ly * hx, w8 = lz * ly * lx;
        return;
    }

    __global__ void ROIAlignBackward_3d(const int nthreads, const float* top_diff, const float spatial_scale,
                                     const int slices, const int height, const int width, const int channels,
                                     const int aligned_slices, const int aligned_height, const int aligned_width,
                                     const int sampling_ratio, float* bottom_diff, const float* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ps, ph, pw) is an element in the aligned output
            int ps = index % aligned_slices;
            int pw = (index / aligned_slices) % aligned_width;
            int ph = (index / aligned_slices / aligned_width) % aligned_height;
            int c  = (index / aligned_slices / aligned_width / aligned_height) % channels;
            int n  = index / aligned_slices / aligned_width / aligned_height / channels;

            const float* offset_bottom_rois = bottom_rois + n * 7;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            float roi_start_w = offset_bottom_rois[1] * spatial_scale;
            float roi_start_h = offset_bottom_rois[2] * spatial_scale;
            float roi_start_s = offset_bottom_rois[3] * spatial_scale;
            float roi_end_w = offset_bottom_rois[4] * spatial_scale;
            float roi_end_h = offset_bottom_rois[5] * spatial_scale;
            float roi_end_s = offset_bottom_rois[6] * spatial_scale;

            // Force malformed ROIs to be 1x1x1
            float roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
            float roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
            float roi_slices = fmaxf(roi_end_s - roi_start_s, 1.f);
            float bin_size_h = roi_height / aligned_height;
            float bin_size_w = roi_width / aligned_width;
            float bin_size_s = roi_slices / aligned_slices;

            float* offset_bottom_diff =
                bottom_diff + (roi_batch_ind * channels + c) * slices * height * width;

            int top_offset = (n * channels + c) * aligned_slices * aligned_height * aligned_width;
            const float* offset_top_diff = top_diff + top_offset;
            const float top_diff_this_bin = offset_top_diff[ps * aligned_height * aligned_width +
                                                                             ph * aligned_width + pw];

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_s = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_slices / aligned_slices);
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const float count = roi_bin_grid_s * roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

            for (int iz = 0; iz < roi_bin_grid_s; iz++)
            {
                const float z = roi_start_s + ps * bin_size_s +
                           (iz + .5f) * bin_size_s / roi_bin_grid_s;

                for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
                {
                    const float y = roi_start_h + ph * bin_size_h +
                        (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
                    for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                        const float x = roi_start_w + pw * bin_size_w +
                            (ix + .5f) * bin_size_w / roi_bin_grid_w;

                        float w1, w2, w3, w4;
                        float w5, w6, w7, w8;
                        int x_low, x_high, y_low, y_high, z_low, z_high;

                        trilinear_interpolate_gradient(
                            slices, height, width, z, y, x, w1, w2, w3, w4, w5, w6, w7, w8,
                            x_low, x_high, y_low, y_high, z_low, z_high, index);

                        float g1 = top_diff_this_bin * w1 / count;
                        float g2 = top_diff_this_bin * w2 / count;
                        float g3 = top_diff_this_bin * w3 / count;
                        float g4 = top_diff_this_bin * w4 / count;
                        float g5 = top_diff_this_bin * w5 / count;
                        float g6 = top_diff_this_bin * w6 / count;
                        float g7 = top_diff_this_bin * w7 / count;
                        float g8 = top_diff_this_bin * w8 / count;

                        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0 && z_low >= 0 && z_high >= 0) {
                        // atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                        // atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                        // atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                        // atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
                        gpu_atomic_add(g1, offset_bottom_diff + z_low * height * width + y_low * width + x_low);
                        gpu_atomic_add(g2, offset_bottom_diff + z_low * height * width + y_low * width + x_high);
                        gpu_atomic_add(g3, offset_bottom_diff + z_low * height * width + y_high * width + x_low);
                        gpu_atomic_add(g4, offset_bottom_diff + z_low * height * width + y_high * width + x_high);
                        gpu_atomic_add(g5, offset_bottom_diff + z_high * height * width + y_low * width + x_low);
                        gpu_atomic_add(g6, offset_bottom_diff + z_high * height * width + y_low * width + x_high);
                        gpu_atomic_add(g7, offset_bottom_diff + z_high * height * width + y_high * width + x_low);
                        gpu_atomic_add(g8, offset_bottom_diff + z_high * height * width + y_high * width + x_high);
                        } // if
                    } // ix
                } // iy
            } // iz
        } // CUDA_1D_KERNEL_LOOP
    } // RoIAlignBackward_3d

    int ROIAlignBackwardLaucher_3d(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
                                const int slices, const int height, const int width, const int channels,
                                const int aligned_slices, const int aligned_height, const int aligned_width,
                                const int sampling_ratio, const float* bottom_rois, float* bottom_diff,
                                cudaStream_t stream) {
        const int kThreadsPerBlock = 512;
        const int output_size = num_rois * aligned_slices * aligned_height * aligned_width * channels;
        cudaError_t err;

        ROIAlignBackward_3d<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
          output_size, top_diff, spatial_scale, slices, height, width, channels,
          aligned_slices, aligned_height, aligned_width,  sampling_ratio, bottom_diff, bottom_rois);

        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }


#ifdef __cplusplus
}
#endif
