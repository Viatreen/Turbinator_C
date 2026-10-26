#include "turbinator/training/cuda_features.h"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tb_cuda_feature_batch {
    int device_id;
    size_t max_batch;
    cudaStream_t stream;
    float *obs;
    float *mission;
    unsigned char *cameras;
    float *features;
    char error[192];
};

static int tb_feature_fail(tb_cuda_feature_batch *batch, const char *stage,
                           cudaError_t status) {
    if (batch) snprintf(batch->error, sizeof(batch->error), "%s: %s",
                        stage, cudaGetErrorString(status));
    return 0;
}

extern "C" tb_cuda_feature_batch *tb_cuda_feature_batch_create(int device_id,
                                                               size_t max_batch) {
    if (max_batch == 0 || max_batch > 1024 || cudaSetDevice(device_id) != cudaSuccess)
        return NULL;
    tb_cuda_feature_batch *batch = (tb_cuda_feature_batch *)calloc(1, sizeof(*batch));
    if (!batch) return NULL;
    batch->device_id = device_id;
    batch->max_batch = max_batch;
    if (cudaStreamCreateWithFlags(&batch->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc((void **)&batch->obs, max_batch * 4148 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&batch->mission, max_batch * 10 * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **)&batch->cameras, max_batch * 16384) != cudaSuccess ||
        cudaMalloc((void **)&batch->features, max_batch * 142 * sizeof(float)) != cudaSuccess) {
        tb_cuda_feature_batch_destroy(batch);
        return NULL;
    }
    return batch;
}

extern "C" void tb_cuda_feature_batch_destroy(tb_cuda_feature_batch *batch) {
    if (!batch) return;
    cudaSetDevice(batch->device_id);
    if (batch->stream) cudaStreamSynchronize(batch->stream);
    cudaFree(batch->obs);
    cudaFree(batch->mission);
    cudaFree(batch->cameras);
    cudaFree(batch->features);
    if (batch->stream) cudaStreamDestroy(batch->stream);
    free(batch);
}

extern "C" const char *tb_cuda_feature_batch_error(const tb_cuda_feature_batch *batch) {
    return batch ? batch->error : "CUDA feature batch unavailable";
}

extern "C" int tb_cuda_extract_features_host(tb_cuda_feature_batch *batch,
                                               const float *obs, const float *mission,
                                               const unsigned char *cameras,
                                               size_t count, float *out_features) {
    if (!batch || !obs || !mission || !cameras || !out_features ||
        count == 0 || count > batch->max_batch) return 0;
    batch->error[0] = 0;
    cudaError_t status = cudaSetDevice(batch->device_id);
    if (status != cudaSuccess) return tb_feature_fail(batch, "select device", status);
    status = cudaMemcpyAsync(batch->obs, obs, count * 4148 * sizeof(float),
                             cudaMemcpyHostToDevice, batch->stream);
    if (status != cudaSuccess) return tb_feature_fail(batch, "copy observations", status);
    status = cudaMemcpyAsync(batch->mission, mission, count * 10 * sizeof(float),
                             cudaMemcpyHostToDevice, batch->stream);
    if (status != cudaSuccess) return tb_feature_fail(batch, "copy mission", status);
    status = cudaMemcpyAsync(batch->cameras, cameras, count * 16384,
                             cudaMemcpyHostToDevice, batch->stream);
    if (status != cudaSuccess) return tb_feature_fail(batch, "copy cameras", status);
    if (!tb_cuda_extract_features_device(batch->obs, batch->mission, batch->cameras,
                                         batch->features, count, batch->stream))
        return tb_feature_fail(batch, "extract features", cudaGetLastError());
    status = cudaMemcpyAsync(out_features, batch->features, count * 142 * sizeof(float),
                             cudaMemcpyDeviceToHost, batch->stream);
    if (status != cudaSuccess) return tb_feature_fail(batch, "copy features", status);
    status = cudaStreamSynchronize(batch->stream);
    return status == cudaSuccess ? 1 : tb_feature_fail(batch, "synchronize features", status);
}

static __device__ float tb_cuda_clip(float value, float lo, float hi) {
    if (!isfinite(value)) return 0.0f;
    return fminf(fmaxf(value, lo), hi);
}

static __device__ float tb_cuda_lidar_feature(float range_m) {
    if (!isfinite(range_m) || range_m < 0.0f) return -1.0f;
    return log1pf(fminf(range_m, 120.0f)) / log1pf(120.0f);
}

static __device__ float tb_cuda_scalar_feature(int index, float value) {
    float scale = 1.0f;
    if (index < 2) scale = 300.0f;
    else if (index == 2) scale = 100.0f;
    else if (index < 6) scale = 50.0f;
    else if (index >= 10 && index < 13) scale = 5.0f;
    else if (index >= 13 && index < 16) scale = 30.0f;
    else if (index >= 28 && index < 32) scale = 3.14159265358979323846f / 4.0f;
    else if (index >= 32 && index < 40) scale = 5.0f;
    else if (index == 44 || index == 45) scale = 3.14159265358979323846f / 2.0f;
    return tb_cuda_clip(value / scale, -1.0f, 1.0f);
}

static __global__ void tb_features_kernel(const float *obs,
                                          const float *mission,
                                          const unsigned char *cameras,
                                          float *features,
                                          size_t batch_size) {
    const size_t env = (size_t)blockIdx.x;
    const int feature = (int)threadIdx.x;
    if (env >= batch_size || feature >= 142) return;
    const float *o = obs + env * 4148;
    const float *m = mission + env * 10;
    const unsigned char *c = cameras ? cameras + env * 16384 : NULL;
    float value = 0.0f;
    if (feature < 48) {
        value = tb_cuda_scalar_feature(feature, o[feature]);
    } else if (feature < 52) {
        value = tb_cuda_clip(o[4144 + feature - 48], 0.0f, 1.0f);
    } else if (feature < 62) {
        value = feature >= 59 && m[6] > 0.5f ? 0.0f :
                tb_cuda_clip(m[feature - 52], -1.0f, 1.0f);
    } else if (feature < 126) {
        const int sector = feature - 62;
        const int head = sector / 32;
        const int el_sector = (sector % 32) / 8;
        const int az_sector = sector % 8;
        for (int el = 0; el < 4; ++el) {
            for (int az = 0; az < 16; ++az) {
                const int idx = 48 + head * 2048 + (el_sector * 4 + el) * 128 + az_sector * 16 + az;
                value += tb_cuda_lidar_feature(o[idx]);
            }
        }
        value /= 64.0f;
    } else if (c) {
        const int quadrant = feature - 126;
        const int cam = quadrant / 4;
        const int qy = (quadrant % 4) / 2;
        const int qx = quadrant % 2;
        unsigned int sum = 0;
        for (int y = 0; y < 32; ++y)
            for (int x = 0; x < 32; ++x)
                sum += c[cam * 4096 + (qy * 32 + y) * 64 + qx * 32 + x];
        value = (float)sum / (1024.0f * 255.0f);
    }
    features[env * 142 + feature] = value;
}

extern "C" int tb_cuda_extract_features_device(const float *device_obs,
                                                  const float *device_mission,
                                                  const unsigned char *device_cameras,
                                                  float *device_features,
                                                  size_t batch_size,
                                                  void *stream) {
    if (!device_obs || !device_mission || !device_features || batch_size == 0 || batch_size > 0x7fffffffU)
        return 0;
    tb_features_kernel<<<(unsigned int)batch_size, 256, 0, (cudaStream_t)stream>>>(
        device_obs, device_mission, device_cameras, device_features, batch_size);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" int tb_cuda_synchronize(void *stream) {
    return cudaStreamSynchronize((cudaStream_t)stream) == cudaSuccess;
}
