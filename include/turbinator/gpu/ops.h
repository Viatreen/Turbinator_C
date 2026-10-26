#ifndef TURBINATOR_GPU_OPS_H
#define TURBINATOR_GPU_OPS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CUDA C tensor primitives. All tensors are contiguous FP32 device buffers.
 * Matrix tensors are row-major. The implementation owns its cuBLAS handle. */
typedef struct tb_gpu_ops tb_gpu_ops;

int tb_gpu_device_count(void);
tb_gpu_ops *tb_gpu_ops_create(int device_index);
void tb_gpu_ops_destroy(tb_gpu_ops *ops);
const char *tb_gpu_last_error(const tb_gpu_ops *ops);
int tb_gpu_device(const tb_gpu_ops *ops);
void *tb_gpu_stream(const tb_gpu_ops *ops);
int tb_gpu_sync(tb_gpu_ops *ops);
void *tb_gpu_malloc(tb_gpu_ops *ops, size_t bytes);
int tb_gpu_free(tb_gpu_ops *ops, void *device_pointer);
int tb_gpu_h2d(tb_gpu_ops *ops, void *device_destination, const void *host_source, size_t bytes);
int tb_gpu_d2h(tb_gpu_ops *ops, void *host_destination, const void *device_source, size_t bytes);
int tb_gpu_d2d(tb_gpu_ops *ops, void *device_destination, const void *device_source, size_t bytes);
int tb_gpu_zero(tb_gpu_ops *ops, float *destination, size_t count);
int tb_gpu_fill(tb_gpu_ops *ops, float *destination, float value, size_t count);
int tb_gpu_copy(tb_gpu_ops *ops, float *destination, const float *source, size_t count);
int tb_gpu_scale(tb_gpu_ops *ops, float *x, float scale, size_t count);
int tb_gpu_axpy(tb_gpu_ops *ops, float *y, const float *x, float alpha, size_t count);
int tb_gpu_l2_norm(tb_gpu_ops *ops, const float *x, size_t count, float *out_norm);

/* Computes C = alpha op(A) op(B) + beta C where logical C is m x n and
 * inner dimension k. Physical input shapes are contiguous row-major. */
int tb_gpu_gemm(tb_gpu_ops *ops,
                const float *a, const float *b, float *c,
                int m, int n, int k, int transpose_a, int transpose_b,
                float alpha, float beta);

/* Dense layer X[B,in] W[out,in] + bias[out] -> Y[B,out]. */
int tb_gpu_linear_fwd(tb_gpu_ops *ops, const float *x, const float *weight,
                      const float *bias, float *y, int batch, int in_dim, int out_dim);
/* dX, dW, dBias are accumulated when accumulate is nonzero, overwritten
 * otherwise. Any gradient output may be NULL. */
int tb_gpu_linear_bwd(tb_gpu_ops *ops, const float *x, const float *weight,
                      const float *dy, float *dx, float *dweight, float *dbias,
                      int batch, int in_dim, int out_dim, int accumulate);
int tb_gpu_bias_add(tb_gpu_ops *ops, float *y, const float *bias, int rows, int cols);
int tb_gpu_bias_grad(tb_gpu_ops *ops, const float *dy, float *dbias,
                     int rows, int cols, int accumulate);

/* mean and inv_std are optional length-rows device scratch arrays. If both
 * are supplied to forward, backward can reuse them. */
int tb_gpu_layernorm_fwd(tb_gpu_ops *ops, const float *x, const float *gamma,
                         const float *beta, float *y, float *mean, float *inv_std,
                         int rows, int cols, float epsilon);
int tb_gpu_layernorm_bwd(tb_gpu_ops *ops, const float *x, const float *gamma,
                         const float *dy, const float *mean, const float *inv_std,
                         float *dx, float *dgamma, float *dbeta,
                         int rows, int cols, int accumulate);
int tb_gpu_silu_fwd(tb_gpu_ops *ops, const float *x, float *y, size_t count);
int tb_gpu_silu_bwd(tb_gpu_ops *ops, const float *x, const float *dy,
                    float *dx, size_t count, int accumulate);
int tb_gpu_softmax_fwd(tb_gpu_ops *ops, const float *x, float *y,
                       int rows, int cols);
int tb_gpu_softmax_bwd(tb_gpu_ops *ops, const float *y, const float *dy,
                       float *dx, int rows, int cols, int accumulate);

/* NCHW convolution, stride 1.., symmetric zero padding. Output dimensions
 * are floor((height+2*pad_h-kernel_h)/stride_h)+1 and likewise for width.
 * Gradients are accumulated when accumulate is nonzero. */
int tb_gpu_conv2d_fwd(tb_gpu_ops *ops, const float *x, const float *weight,
                      const float *bias, float *y, int batch, int in_channels,
                      int out_channels, int height, int width, int kernel_h,
                      int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
int tb_gpu_conv2d_bwd(tb_gpu_ops *ops, const float *x, const float *weight,
                      const float *dy, float *dx, float *dweight, float *dbias,
                      int batch, int in_channels, int out_channels, int height,
                      int width, int kernel_h, int kernel_w, int stride_h,
                      int stride_w, int pad_h, int pad_w, int accumulate);

#ifdef __cplusplus
}
#endif
#endif
