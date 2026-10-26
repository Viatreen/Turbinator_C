#include "turbinator/gpu/ops.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tb_gpu_ops {
    int device_index;
    cudaStream_t stream;
    cublasHandle_t blas;
    char error[256];
};

static int tb_fail_cuda(tb_gpu_ops *ops, cudaError_t code, const char *operation) {
    if (ops) snprintf(ops->error, sizeof(ops->error), "%s: %s", operation, cudaGetErrorString(code));
    return 0;
}
static int tb_fail_blas(tb_gpu_ops *ops, cublasStatus_t code, const char *operation) {
    if (ops) snprintf(ops->error, sizeof(ops->error), "%s: cuBLAS status %d", operation, (int)code);
    return 0;
}
static int tb_select(tb_gpu_ops *ops) {
    if (!ops) return 0;
    cudaError_t code = cudaSetDevice(ops->device_index);
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, "cudaSetDevice");
}
static int tb_launch_ok(tb_gpu_ops *ops, const char *operation) {
    cudaError_t code = cudaGetLastError();
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, operation);
}

extern "C" int tb_gpu_device_count(void) {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}
extern "C" tb_gpu_ops *tb_gpu_ops_create(int device_index) {
    int count = tb_gpu_device_count();
    if (device_index < 0 || device_index >= count) return NULL;
    tb_gpu_ops *ops = (tb_gpu_ops *)calloc(1, sizeof(*ops));
    if (!ops) return NULL;
    ops->device_index = device_index;
    if (!tb_select(ops) || cudaStreamCreateWithFlags(&ops->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cublasCreate(&ops->blas) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(ops->blas, ops->stream) != CUBLAS_STATUS_SUCCESS) {
        if (ops->blas) cublasDestroy(ops->blas);
        if (ops->stream) cudaStreamDestroy(ops->stream);
        free(ops);
        return NULL;
    }
    ops->error[0] = 0;
    return ops;
}
extern "C" void tb_gpu_ops_destroy(tb_gpu_ops *ops) {
    if (!ops) return;
    tb_select(ops);
    if (ops->blas) cublasDestroy(ops->blas);
    if (ops->stream) cudaStreamDestroy(ops->stream);
    free(ops);
}
extern "C" const char *tb_gpu_last_error(const tb_gpu_ops *ops) { return ops ? ops->error : "invalid GPU context"; }
extern "C" int tb_gpu_device(const tb_gpu_ops *ops) { return ops ? ops->device_index : -1; }
extern "C" void *tb_gpu_stream(const tb_gpu_ops *ops) { return ops ? (void *)ops->stream : NULL; }
extern "C" int tb_gpu_sync(tb_gpu_ops *ops) {
    if (!tb_select(ops)) return 0;
    cudaError_t code = cudaStreamSynchronize(ops->stream);
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, "cudaStreamSynchronize");
}
extern "C" void *tb_gpu_malloc(tb_gpu_ops *ops, size_t bytes) {
    if (!tb_select(ops) || bytes == 0) return NULL;
    void *out = NULL;
    cudaError_t code = cudaMalloc(&out, bytes);
    if (code != cudaSuccess) { tb_fail_cuda(ops, code, "cudaMalloc"); return NULL; }
    return out;
}
extern "C" int tb_gpu_free(tb_gpu_ops *ops, void *device_pointer) {
    if (!tb_select(ops)) return 0;
    cudaError_t code = device_pointer ? cudaFree(device_pointer) : cudaSuccess;
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, "cudaFree");
}
extern "C" int tb_gpu_h2d(tb_gpu_ops *ops, void *dst, const void *src, size_t bytes) {
    if (!tb_select(ops) || (!dst && bytes) || (!src && bytes)) return 0;
    cudaError_t code = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ops->stream);
    if (code != cudaSuccess) return tb_fail_cuda(ops, code, "cudaMemcpy H2D");
    return tb_gpu_sync(ops);
}
extern "C" int tb_gpu_d2h(tb_gpu_ops *ops, void *dst, const void *src, size_t bytes) {
    if (!tb_select(ops) || (!dst && bytes) || (!src && bytes)) return 0;
    cudaError_t code = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ops->stream);
    if (code != cudaSuccess) return tb_fail_cuda(ops, code, "cudaMemcpy D2H");
    return tb_gpu_sync(ops);
}
extern "C" int tb_gpu_d2d(tb_gpu_ops *ops, void *dst, const void *src, size_t bytes) {
    if (!tb_select(ops) || (!dst && bytes) || (!src && bytes)) return 0;
    cudaError_t code = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ops->stream);
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, "cudaMemcpy D2D");
}
extern "C" int tb_gpu_zero(tb_gpu_ops *ops, float *dst, size_t count) {
    if (!tb_select(ops) || (!dst && count)) return 0;
    cudaError_t code = cudaMemsetAsync(dst, 0, count * sizeof(float), ops->stream);
    return code == cudaSuccess ? 1 : tb_fail_cuda(ops, code, "cudaMemset");
}

__global__ static void tb_fill_kernel(float *x, float value, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = value;
}
__global__ static void tb_scale_kernel(float *x, float scale, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}
__global__ static void tb_axpy_kernel(float *y, const float *x, float alpha, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += alpha * x[i];
}
extern "C" int tb_gpu_fill(tb_gpu_ops *ops, float *dst, float value, size_t count) {
    if (!tb_select(ops) || (!dst && count)) return 0;
    if (!count) return 1;
    tb_fill_kernel<<<(unsigned)((count + 255) / 256), 256, 0, ops->stream>>>(dst, value, count);
    return tb_launch_ok(ops, "fill kernel");
}
extern "C" int tb_gpu_copy(tb_gpu_ops *ops, float *dst, const float *src, size_t count) {
    return tb_gpu_d2d(ops, dst, src, count * sizeof(float));
}
extern "C" int tb_gpu_scale(tb_gpu_ops *ops, float *x, float scale, size_t count) {
    if (!tb_select(ops) || (!x && count)) return 0;
    if (!count) return 1;
    tb_scale_kernel<<<(unsigned)((count + 255) / 256), 256, 0, ops->stream>>>(x, scale, count);
    return tb_launch_ok(ops, "scale kernel");
}
extern "C" int tb_gpu_axpy(tb_gpu_ops *ops, float *y, const float *x, float alpha, size_t count) {
    if (!tb_select(ops) || ((!x || !y) && count)) return 0;
    if (!count) return 1;
    tb_axpy_kernel<<<(unsigned)((count + 255) / 256), 256, 0, ops->stream>>>(y, x, alpha, count);
    return tb_launch_ok(ops, "axpy kernel");
}
extern "C" int tb_gpu_l2_norm(tb_gpu_ops *ops, const float *x, size_t count, float *out_norm) {
    if (!tb_select(ops) || !x || !out_norm || count == 0 || count > 2147483647u) return 0;
    cublasStatus_t code = cublasSnrm2(ops->blas, (int)count, x, 1, out_norm);
    return code == CUBLAS_STATUS_SUCCESS ? 1 : tb_fail_blas(ops, code, "cublasSnrm2");
}

extern "C" int tb_gpu_gemm(tb_gpu_ops *ops, const float *a, const float *b, float *c,
                           int m, int n, int k, int transpose_a, int transpose_b,
                           float alpha, float beta) {
    if (!tb_select(ops) || !a || !b || !c || m <= 0 || n <= 0 || k <= 0) return 0;
    cublasOperation_t op_b = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_a = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    int ldb = transpose_b ? k : n;
    int lda = transpose_a ? m : k;
    cublasStatus_t code = cublasSgemm(ops->blas, op_b, op_a, n, m, k,
                                     &alpha, b, ldb, a, lda, &beta, c, n);
    return code == CUBLAS_STATUS_SUCCESS ? 1 : tb_fail_blas(ops, code, "cublasSgemm");
}

__global__ static void tb_bias_add_kernel(float *y, const float *bias, int rows, int cols) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)rows * cols) y[i] += bias[i % cols];
}
__global__ static void tb_bias_grad_kernel(const float *dy, float *dbias,
                                           int rows, int cols, int accumulate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float sum = 0.0f;
    for (int r = 0; r < rows; ++r) sum += dy[(size_t)r * cols + col];
    dbias[col] = accumulate ? dbias[col] + sum : sum;
}
extern "C" int tb_gpu_bias_add(tb_gpu_ops *ops, float *y, const float *bias, int rows, int cols) {
    if (!tb_select(ops) || !y || !bias || rows <= 0 || cols <= 0) return 0;
    size_t n = (size_t)rows * cols;
    tb_bias_add_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ops->stream>>>(y, bias, rows, cols);
    return tb_launch_ok(ops, "bias-add kernel");
}
extern "C" int tb_gpu_bias_grad(tb_gpu_ops *ops, const float *dy, float *dbias,
                                int rows, int cols, int accumulate) {
    if (!tb_select(ops) || !dy || !dbias || rows <= 0 || cols <= 0) return 0;
    tb_bias_grad_kernel<<<(unsigned)((cols + 255) / 256), 256, 0, ops->stream>>>(dy, dbias, rows, cols, accumulate);
    return tb_launch_ok(ops, "bias-gradient kernel");
}
extern "C" int tb_gpu_linear_fwd(tb_gpu_ops *ops, const float *x, const float *w,
                                 const float *bias, float *y, int batch, int in_dim, int out_dim) {
    if (!tb_gpu_gemm(ops, x, w, y, batch, out_dim, in_dim, 0, 1, 1.0f, 0.0f)) return 0;
    return bias ? tb_gpu_bias_add(ops, y, bias, batch, out_dim) : 1;
}
extern "C" int tb_gpu_linear_bwd(tb_gpu_ops *ops, const float *x, const float *w,
                                 const float *dy, float *dx, float *dw, float *db,
                                 int batch, int in_dim, int out_dim, int accumulate) {
    if (!ops || !x || !w || !dy || batch <= 0 || in_dim <= 0 || out_dim <= 0) return 0;
    float beta = accumulate ? 1.0f : 0.0f;
    if (dx && !tb_gpu_gemm(ops, dy, w, dx, batch, in_dim, out_dim, 0, 0, 1.0f, beta)) return 0;
    if (dw && !tb_gpu_gemm(ops, dy, x, dw, out_dim, in_dim, batch, 1, 0, 1.0f, beta)) return 0;
    return !db || tb_gpu_bias_grad(ops, dy, db, batch, out_dim, accumulate);
}

__global__ static void tb_layernorm_fwd_kernel(const float *x, const float *gamma,
                                                const float *beta, float *y,
                                                float *means, float *invstd,
                                                int cols, float eps) {
    int row = blockIdx.x;
    __shared__ float part[256];
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += x[(size_t)row * cols + c];
    part[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    float mean = part[0] / cols;
    sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float d = x[(size_t)row * cols + c] - mean;
        sum += d * d;
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    float rstd = rsqrtf(part[0] / cols + eps);
    if (threadIdx.x == 0) {
        if (means) means[row] = mean;
        if (invstd) invstd[row] = rstd;
    }
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float normalized = (x[(size_t)row * cols + c] - mean) * rstd;
        y[(size_t)row * cols + c] = normalized * (gamma ? gamma[c] : 1.0f) + (beta ? beta[c] : 0.0f);
    }
}
__global__ static void tb_layernorm_dx_kernel(const float *x, const float *gamma,
                                               const float *dy, const float *means,
                                               const float *invstd, float *dx,
                                               int cols, int accumulate) {
    int row = blockIdx.x;
    __shared__ float sum1[256], sum2[256];
    float s1 = 0.0f, s2 = 0.0f;
    float mean = means[row], rstd = invstd[row];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float g = dy[(size_t)row * cols + c] * (gamma ? gamma[c] : 1.0f);
        float xn = (x[(size_t)row * cols + c] - mean) * rstd;
        s1 += g;
        s2 += g * xn;
    }
    sum1[threadIdx.x] = s1;
    sum2[threadIdx.x] = s2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sum1[threadIdx.x] += sum1[threadIdx.x + s]; sum2[threadIdx.x] += sum2[threadIdx.x + s]; }
        __syncthreads();
    }
    float m1 = sum1[0] / cols, m2 = sum2[0] / cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t i = (size_t)row * cols + c;
        float g = dy[i] * (gamma ? gamma[c] : 1.0f);
        float xn = (x[i] - mean) * rstd;
        float value = rstd * (g - m1 - xn * m2);
        dx[i] = accumulate ? dx[i] + value : value;
    }
}
__global__ static void tb_layernorm_param_kernel(const float *x, const float *dy,
                                                  const float *means, const float *invstd,
                                                  float *dgamma, float *dbeta,
                                                  int rows, int cols, int accumulate) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float dg = 0.0f, db = 0.0f;
    for (int r = 0; r < rows; ++r) {
        size_t i = (size_t)r * cols + col;
        float g = dy[i];
        db += g;
        dg += g * ((x[i] - means[r]) * invstd[r]);
    }
    if (dgamma) dgamma[col] = accumulate ? dgamma[col] + dg : dg;
    if (dbeta) dbeta[col] = accumulate ? dbeta[col] + db : db;
}
extern "C" int tb_gpu_layernorm_fwd(tb_gpu_ops *ops, const float *x, const float *gamma,
                                     const float *beta, float *y, float *mean, float *inv_std,
                                     int rows, int cols, float epsilon) {
    if (!tb_select(ops) || !x || !y || rows <= 0 || cols <= 0 || epsilon <= 0.0f) return 0;
    tb_layernorm_fwd_kernel<<<rows, 256, 0, ops->stream>>>(x, gamma, beta, y, mean, inv_std, cols, epsilon);
    return tb_launch_ok(ops, "layernorm forward kernel");
}
extern "C" int tb_gpu_layernorm_bwd(tb_gpu_ops *ops, const float *x, const float *gamma,
                                     const float *dy, const float *mean, const float *inv_std,
                                     float *dx, float *dgamma, float *dbeta,
                                     int rows, int cols, int accumulate) {
    if (!tb_select(ops) || !x || !dy || !mean || !inv_std || rows <= 0 || cols <= 0) return 0;
    if (dx) {
        tb_layernorm_dx_kernel<<<rows, 256, 0, ops->stream>>>(x, gamma, dy, mean, inv_std, dx, cols, accumulate);
        if (!tb_launch_ok(ops, "layernorm input gradient kernel")) return 0;
    }
    if (dgamma || dbeta) {
        tb_layernorm_param_kernel<<<(unsigned)((cols + 255) / 256), 256, 0, ops->stream>>>(
            x, dy, mean, inv_std, dgamma, dbeta, rows, cols, accumulate);
        if (!tb_launch_ok(ops, "layernorm parameter gradient kernel")) return 0;
    }
    return 1;
}

__global__ static void tb_silu_fwd_kernel(const float *x, float *y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] / (1.0f + expf(-x[i]));
}
__global__ static void tb_silu_bwd_kernel(const float *x, const float *dy, float *dx, size_t n, int accumulate) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sigmoid = 1.0f / (1.0f + expf(-x[i]));
        float grad = dy[i] * (sigmoid + x[i] * sigmoid * (1.0f - sigmoid));
        dx[i] = accumulate ? dx[i] + grad : grad;
    }
}
extern "C" int tb_gpu_silu_fwd(tb_gpu_ops *ops, const float *x, float *y, size_t count) {
    if (!tb_select(ops) || ((!x || !y) && count)) return 0;
    if (!count) return 1;
    tb_silu_fwd_kernel<<<(unsigned)((count + 255) / 256), 256, 0, ops->stream>>>(x, y, count);
    return tb_launch_ok(ops, "SiLU forward kernel");
}
extern "C" int tb_gpu_silu_bwd(tb_gpu_ops *ops, const float *x, const float *dy,
                               float *dx, size_t count, int accumulate) {
    if (!tb_select(ops) || ((!x || !dy || !dx) && count)) return 0;
    if (!count) return 1;
    tb_silu_bwd_kernel<<<(unsigned)((count + 255) / 256), 256, 0, ops->stream>>>(x, dy, dx, count, accumulate);
    return tb_launch_ok(ops, "SiLU backward kernel");
}

__global__ static void tb_softmax_fwd_kernel(const float *x, float *y, int cols) {
    int row = blockIdx.x;
    __shared__ float part[256];
    float local_max = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local_max = fmaxf(local_max, x[(size_t)row * cols + c]);
    part[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] = fmaxf(part[threadIdx.x], part[threadIdx.x + s]);
        __syncthreads();
    }
    float max_val = part[0];
    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) local_sum += expf(x[(size_t)row * cols + c] - max_val);
    part[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    float reciprocal = 1.0f / fmaxf(part[0], 1.0e-30f);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        y[(size_t)row * cols + c] = expf(x[(size_t)row * cols + c] - max_val) * reciprocal;
}
__global__ static void tb_softmax_bwd_kernel(const float *y, const float *dy, float *dx, int cols, int accumulate) {
    int row = blockIdx.x;
    __shared__ float part[256];
    float local = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t i = (size_t)row * cols + c;
        local += dy[i] * y[i];
    }
    part[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
        __syncthreads();
    }
    float dot = part[0];
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        size_t i = (size_t)row * cols + c;
        float value = y[i] * (dy[i] - dot);
        dx[i] = accumulate ? dx[i] + value : value;
    }
}
extern "C" int tb_gpu_softmax_fwd(tb_gpu_ops *ops, const float *x, float *y, int rows, int cols) {
    if (!tb_select(ops) || !x || !y || rows <= 0 || cols <= 0) return 0;
    tb_softmax_fwd_kernel<<<rows, 256, 0, ops->stream>>>(x, y, cols);
    return tb_launch_ok(ops, "softmax forward kernel");
}
extern "C" int tb_gpu_softmax_bwd(tb_gpu_ops *ops, const float *y, const float *dy,
                                  float *dx, int rows, int cols, int accumulate) {
    if (!tb_select(ops) || !y || !dy || !dx || rows <= 0 || cols <= 0) return 0;
    tb_softmax_bwd_kernel<<<rows, 256, 0, ops->stream>>>(y, dy, dx, cols, accumulate);
    return tb_launch_ok(ops, "softmax backward kernel");
}

__global__ static void tb_conv_fwd_kernel(const float *x, const float *w, const float *bias,
                                          float *y, int batch, int cin, int cout,
                                          int height, int width, int kh, int kw,
                                          int sh, int sw, int ph, int pw, int oh, int ow) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cout * oh * ow;
    if (index >= total) return;
    int ox = (int)(index % ow);
    int oy = (int)((index / ow) % oh);
    int co = (int)((index / ((size_t)ow * oh)) % cout);
    int b = (int)(index / ((size_t)ow * oh * cout));
    float sum = bias ? bias[co] : 0.0f;
    for (int ci = 0; ci < cin; ++ci) for (int ky = 0; ky < kh; ++ky) for (int kx = 0; kx < kw; ++kx) {
        int iy = oy * sh + ky - ph;
        int ix = ox * sw + kx - pw;
        if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
            size_t xi = (((size_t)b * cin + ci) * height + iy) * width + ix;
            size_t wi = (((size_t)co * cin + ci) * kh + ky) * kw + kx;
            sum += x[xi] * w[wi];
        }
    }
    y[index] = sum;
}
__global__ static void tb_conv_dx_kernel(const float *w, const float *dy, float *dx,
                                         int batch, int cin, int cout, int height, int width,
                                         int kh, int kw, int sh, int sw, int ph, int pw,
                                         int oh, int ow, int accumulate) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)batch * cin * height * width;
    if (index >= total) return;
    int ix = (int)(index % width);
    int iy = (int)((index / width) % height);
    int ci = (int)((index / ((size_t)width * height)) % cin);
    int b = (int)(index / ((size_t)width * height * cin));
    float sum = 0.0f;
    for (int co = 0; co < cout; ++co) for (int ky = 0; ky < kh; ++ky) for (int kx = 0; kx < kw; ++kx) {
        int ny = iy + ph - ky, nx = ix + pw - kx;
        if (ny < 0 || nx < 0 || ny % sh || nx % sw) continue;
        int oy = ny / sh, ox = nx / sw;
        if (oy >= oh || ox >= ow) continue;
        size_t yi = (((size_t)b * cout + co) * oh + oy) * ow + ox;
        size_t wi = (((size_t)co * cin + ci) * kh + ky) * kw + kx;
        sum += dy[yi] * w[wi];
    }
    dx[index] = accumulate ? dx[index] + sum : sum;
}
__global__ static void tb_conv_dw_kernel(const float *x, const float *dy, float *dw,
                                         int batch, int cin, int cout, int height, int width,
                                         int kh, int kw, int sh, int sw, int ph, int pw,
                                         int oh, int ow, int accumulate) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)cout * cin * kh * kw;
    if (index >= total) return;
    int kx = (int)(index % kw);
    int ky = (int)((index / kw) % kh);
    int ci = (int)((index / ((size_t)kw * kh)) % cin);
    int co = (int)(index / ((size_t)kw * kh * cin));
    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) for (int oy = 0; oy < oh; ++oy) for (int ox = 0; ox < ow; ++ox) {
        int iy = oy * sh + ky - ph, ix = ox * sw + kx - pw;
        if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
        size_t xi = (((size_t)b * cin + ci) * height + iy) * width + ix;
        size_t yi = (((size_t)b * cout + co) * oh + oy) * ow + ox;
        sum += x[xi] * dy[yi];
    }
    dw[index] = accumulate ? dw[index] + sum : sum;
}
__global__ static void tb_conv_db_kernel(const float *dy, float *db,
                                         int batch, int cout, int oh, int ow, int accumulate) {
    int co = blockIdx.x * blockDim.x + threadIdx.x;
    if (co >= cout) return;
    float sum = 0.0f;
    for (int b = 0; b < batch; ++b) for (int oy = 0; oy < oh; ++oy) for (int ox = 0; ox < ow; ++ox)
        sum += dy[(((size_t)b * cout + co) * oh + oy) * ow + ox];
    db[co] = accumulate ? db[co] + sum : sum;
}
static int tb_conv_dims(int height, int width, int kh, int kw, int sh, int sw, int ph, int pw, int *oh, int *ow) {
    if (height <= 0 || width <= 0 || kh <= 0 || kw <= 0 || sh <= 0 || sw <= 0 || ph < 0 || pw < 0) return 0;
    *oh = (height + 2 * ph - kh) / sh + 1;
    *ow = (width + 2 * pw - kw) / sw + 1;
    return *oh > 0 && *ow > 0;
}
extern "C" int tb_gpu_conv2d_fwd(tb_gpu_ops *ops, const float *x, const float *weight,
                                  const float *bias, float *y, int batch, int cin, int cout,
                                  int height, int width, int kh, int kw, int sh, int sw, int ph, int pw) {
    int oh, ow;
    if (!tb_select(ops) || !x || !weight || !y || batch <= 0 || cin <= 0 || cout <= 0 ||
        !tb_conv_dims(height, width, kh, kw, sh, sw, ph, pw, &oh, &ow)) return 0;
    size_t total = (size_t)batch * cout * oh * ow;
    tb_conv_fwd_kernel<<<(unsigned)((total + 255) / 256), 256, 0, ops->stream>>>(
        x, weight, bias, y, batch, cin, cout, height, width, kh, kw, sh, sw, ph, pw, oh, ow);
    return tb_launch_ok(ops, "conv2d forward kernel");
}
extern "C" int tb_gpu_conv2d_bwd(tb_gpu_ops *ops, const float *x, const float *weight,
                                  const float *dy, float *dx, float *dw, float *db,
                                  int batch, int cin, int cout, int height, int width,
                                  int kh, int kw, int sh, int sw, int ph, int pw, int accumulate) {
    int oh, ow;
    if (!tb_select(ops) || !x || !weight || !dy || batch <= 0 || cin <= 0 || cout <= 0 ||
        !tb_conv_dims(height, width, kh, kw, sh, sw, ph, pw, &oh, &ow)) return 0;
    if (dx) {
        size_t n = (size_t)batch * cin * height * width;
        tb_conv_dx_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ops->stream>>>(
            weight, dy, dx, batch, cin, cout, height, width, kh, kw, sh, sw, ph, pw, oh, ow, accumulate);
        if (!tb_launch_ok(ops, "conv2d input gradient kernel")) return 0;
    }
    if (dw) {
        size_t n = (size_t)cout * cin * kh * kw;
        tb_conv_dw_kernel<<<(unsigned)((n + 255) / 256), 256, 0, ops->stream>>>(
            x, dy, dw, batch, cin, cout, height, width, kh, kw, sh, sw, ph, pw, oh, ow, accumulate);
        if (!tb_launch_ok(ops, "conv2d weight gradient kernel")) return 0;
    }
    if (db) {
        tb_conv_db_kernel<<<(unsigned)((cout + 255) / 256), 256, 0, ops->stream>>>(dy, db, batch, cout, oh, ow, accumulate);
        if (!tb_launch_ok(ops, "conv2d bias gradient kernel")) return 0;
    }
    return 1;
}
