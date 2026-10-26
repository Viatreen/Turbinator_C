#include "turbinator/training/cuda_features.h"

#include <cuda_runtime.h>
#include <math.h>

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
