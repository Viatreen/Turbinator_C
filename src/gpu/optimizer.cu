#include "turbinator/gpu/optimizer.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tb_gpu_rmsprop {
    tb_gpu_ops *ops;
    float *square_average;
    size_t count;
    float learning_rate;
    float decay;
    float epsilon;
    float weight_decay;
    float maximum_gradient_norm;
    uint64_t steps;
};

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t count;
    uint64_t steps;
    float learning_rate;
    float decay;
    float epsilon;
    float weight_decay;
    float maximum_gradient_norm;
    uint32_t checksum;
} tb_optimizer_header;

static uint32_t tb_fnv1a(uint32_t state, const unsigned char *p, size_t n) {
    for (size_t i = 0; i < n; ++i) { state ^= p[i]; state *= UINT32_C(16777619); }
    return state;
}

__global__ static void tb_rmsprop_step_kernel(float *parameters, float *gradients,
                                               float *moments, size_t count,
                                               float lr, float decay, float eps,
                                               float wd, float clip_scale) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float p = parameters[i];
    float g = gradients[i] * clip_scale + wd * p;
    float m = decay * moments[i] + (1.0f - decay) * g * g;
    moments[i] = m;
    parameters[i] = p - lr * g * rsqrtf(m + eps);
    gradients[i] = 0.0f;
}

__global__ static void tb_ema_kernel(float *target, const float *source,
                                     size_t count, float tau) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) target[i] = (1.0f - tau) * target[i] + tau * source[i];
}

extern "C" tb_gpu_rmsprop *tb_gpu_rmsprop_create(tb_gpu_ops *ops, size_t count,
                                                 float lr, float decay, float eps,
                                                 float wd, float max_norm) {
    if (!ops || count == 0 || !isfinite(lr) || lr <= 0 ||
        !isfinite(decay) || decay <= 0 || decay >= 1 ||
        !isfinite(eps) || eps <= 0 || !isfinite(wd) || wd < 0 ||
        !isfinite(max_norm) || max_norm <= 0) return NULL;
    tb_gpu_rmsprop *o = (tb_gpu_rmsprop *)calloc(1, sizeof(*o));
    if (!o) return NULL;
    o->ops = ops;
    o->count = count;
    o->learning_rate = lr;
    o->decay = decay;
    o->epsilon = eps;
    o->weight_decay = wd;
    o->maximum_gradient_norm = max_norm;
    o->square_average = (float *)tb_gpu_malloc(ops, count * sizeof(float));
    if (!o->square_average || !tb_gpu_zero(ops, o->square_average, count)) {
        tb_gpu_rmsprop_destroy(o);
        return NULL;
    }
    return o;
}
extern "C" void tb_gpu_rmsprop_destroy(tb_gpu_rmsprop *o) {
    if (!o) return;
    if (o->square_average) tb_gpu_free(o->ops, o->square_average);
    free(o);
}
extern "C" int tb_gpu_rmsprop_step(tb_gpu_rmsprop *o, float *params,
                                    float *grads, float *out_norm) {
    if (!o || !params || !grads) return 0;
    float norm = 0;
    if (!tb_gpu_l2_norm(o->ops, grads, o->count, &norm) || !isfinite(norm)) return 0;
    if (out_norm) *out_norm = norm;
    float scale = norm > o->maximum_gradient_norm ? o->maximum_gradient_norm / norm : 1.0f;
    if (cudaSetDevice(tb_gpu_device(o->ops)) != cudaSuccess) return 0;
    tb_rmsprop_step_kernel<<<(unsigned)((o->count + 255) / 256), 256, 0, (cudaStream_t)tb_gpu_stream(o->ops)>>>(
        params, grads, o->square_average, o->count, o->learning_rate,
        o->decay, o->epsilon, o->weight_decay, scale);
    if (cudaGetLastError() != cudaSuccess || !tb_gpu_sync(o->ops)) return 0;
    ++o->steps;
    return 1;
}
extern "C" uint64_t tb_gpu_rmsprop_steps(const tb_gpu_rmsprop *o) { return o ? o->steps : 0; }
extern "C" float *tb_gpu_rmsprop_moments(tb_gpu_rmsprop *o) { return o ? o->square_average : NULL; }
extern "C" int tb_gpu_rmsprop_set_lr(tb_gpu_rmsprop *o, float lr) {
    if (!o || !isfinite(lr) || lr <= 0) return 0;
    o->learning_rate = lr;
    return 1;
}
extern "C" int tb_gpu_ema_update(tb_gpu_ops *ops, float *target,
                                  const float *source, size_t count, float tau) {
    if (!ops || !target || !source || !count || !isfinite(tau) || tau < 0 || tau > 1 ||
        cudaSetDevice(tb_gpu_device(ops)) != cudaSuccess) return 0;
    tb_ema_kernel<<<(unsigned)((count + 255) / 256), 256, 0, (cudaStream_t)tb_gpu_stream(ops)>>>(target, source, count, tau);
    return cudaGetLastError() == cudaSuccess && tb_gpu_sync(ops);
}

extern "C" int tb_gpu_rmsprop_save(const tb_gpu_rmsprop *o, const char *path) {
    if (!o || !path) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    tb_optimizer_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, "TBRMSP1", 7);
    h.version = 1;
    h.count = (uint64_t)o->count;
    h.steps = o->steps;
    h.learning_rate = o->learning_rate;
    h.decay = o->decay;
    h.epsilon = o->epsilon;
    h.weight_decay = o->weight_decay;
    h.maximum_gradient_norm = o->maximum_gradient_norm;
    const size_t chunk_floats = 1u << 20;
    float *host = (float *)malloc(chunk_floats * sizeof(float));
    if (!host) { fclose(file); return 0; }
    int ok = fwrite(&h, sizeof(h), 1, file) == 1;
    uint32_t checksum = UINT32_C(2166136261);
    for (size_t pos = 0; ok && pos < o->count; pos += chunk_floats) {
        size_t n = o->count - pos < chunk_floats ? o->count - pos : chunk_floats;
        ok = tb_gpu_d2h(o->ops, host, o->square_average + pos, n * sizeof(float)) &&
             fwrite(host, sizeof(float), n, file) == n;
        if (ok) checksum = tb_fnv1a(checksum, (const unsigned char *)host, n * sizeof(float));
    }
    if (ok) {
        h.checksum = checksum;
        ok = fseek(file, 0, SEEK_SET) == 0 && fwrite(&h, sizeof(h), 1, file) == 1;
    }
    free(host);
    if (fclose(file) != 0) ok = 0;
    if (!ok) remove(path);
    return ok;
}

extern "C" int tb_gpu_rmsprop_load(tb_gpu_rmsprop *o, const char *path) {
    if (!o || !path) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    tb_optimizer_header h;
    int ok = fread(&h, sizeof(h), 1, file) == 1 &&
             memcmp(h.magic, "TBRMSP1", 7) == 0 && h.version == 1 && h.count == o->count &&
             isfinite(h.learning_rate) && h.learning_rate > 0 &&
             isfinite(h.decay) && h.decay > 0 && h.decay < 1 &&
             isfinite(h.epsilon) && h.epsilon > 0 &&
             isfinite(h.weight_decay) && h.weight_decay >= 0 &&
             isfinite(h.maximum_gradient_norm) && h.maximum_gradient_norm > 0;
    const size_t chunk_floats = 1u << 20;
    float *host = ok ? (float *)malloc(chunk_floats * sizeof(float)) : NULL;
    if (!host) ok = 0;
    uint32_t checksum = UINT32_C(2166136261);
    for (size_t pos = 0; ok && pos < o->count; pos += chunk_floats) {
        size_t n = o->count - pos < chunk_floats ? o->count - pos : chunk_floats;
        ok = fread(host, sizeof(float), n, file) == n;
        if (ok) checksum = tb_fnv1a(checksum, (const unsigned char *)host, n * sizeof(float));
    }
    if (ok) ok = checksum == h.checksum && fgetc(file) == EOF;
    if (ok) ok = fseek(file, (long)sizeof(h), SEEK_SET) == 0;
    for (size_t pos = 0; ok && pos < o->count; pos += chunk_floats) {
        size_t n = o->count - pos < chunk_floats ? o->count - pos : chunk_floats;
        ok = fread(host, sizeof(float), n, file) == n &&
             tb_gpu_h2d(o->ops, o->square_average + pos, host, n * sizeof(float));
    }
    free(host);
    fclose(file);
    if (ok) {
        o->steps = h.steps;
        o->learning_rate = h.learning_rate;
        o->decay = h.decay;
        o->epsilon = h.epsilon;
        o->weight_decay = h.weight_decay;
        o->maximum_gradient_norm = h.maximum_gradient_norm;
    }
    return ok;
}
