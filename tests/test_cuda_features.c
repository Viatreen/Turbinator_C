#include "turbinator/training/cuda_features.h"
#include "turbinator/training/training.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    enum { COUNT = 3 };
    float *obs = (float *)calloc(COUNT * TB_TRAIN_OBSERVATION_DIM, sizeof(float));
    float *mission = (float *)calloc(COUNT * TB_TRAIN_MISSION_DIM, sizeof(float));
    uint8_t *cameras = (uint8_t *)calloc(COUNT * TB_TRAIN_CAMERA_PIXELS, sizeof(uint8_t));
    float *actual = (float *)calloc(COUNT * TB_TRAIN_FEATURE_DIM, sizeof(float));
    float expected[TB_TRAIN_FEATURE_DIM];
    if (!obs || !mission || !cameras || !actual) return 1;
    for (int b = 0; b < COUNT; ++b) {
        for (int i = 0; i < TB_TRAIN_OBSERVATION_DIM; ++i)
            obs[(size_t)b * TB_TRAIN_OBSERVATION_DIM + i] =
                i >= 48 && i < 4144 ? 0.5f + (float)((i + 17 * b) % 119) :
                (float)((i * 11 + b * 7) % 81 - 40) * 0.125f;
        for (int i = 0; i < TB_TRAIN_MISSION_DIM; ++i)
            mission[(size_t)b * TB_TRAIN_MISSION_DIM + i] = (float)(i - 5 + b) * 0.2f;
        for (int i = 0; i < TB_TRAIN_CAMERA_PIXELS; ++i)
            cameras[(size_t)b * TB_TRAIN_CAMERA_PIXELS + i] = (uint8_t)((i * 13 + b * 29) & 255);
    }
    const char *device_env = getenv("TB_CUDA_TEST_DEVICE");
    int device = device_env ? atoi(device_env) : 0;
    tb_cuda_feature_batch *gpu = tb_cuda_feature_batch_create(device, COUNT);
    if (!gpu || !tb_cuda_extract_features_host(gpu, obs, mission, cameras, COUNT, actual)) {
        fprintf(stderr, "CUDA feature extraction failed: %s\n",
                tb_cuda_feature_batch_error(gpu));
        tb_cuda_feature_batch_destroy(gpu);
        free(obs); free(mission); free(cameras); free(actual);
        return 1;
    }
    for (int b = 0; b < COUNT; ++b) {
        tb_train_extract_features_cpu(obs + (size_t)b * TB_TRAIN_OBSERVATION_DIM,
            mission + (size_t)b * TB_TRAIN_MISSION_DIM,
            cameras + (size_t)b * TB_TRAIN_CAMERA_PIXELS, expected);
        for (int i = 0; i < TB_TRAIN_FEATURE_DIM; ++i) {
            float found = actual[(size_t)b * TB_TRAIN_FEATURE_DIM + i];
            if (!isfinite(found) || fabsf(found - expected[i]) > 2.0e-5f) {
                fprintf(stderr, "feature mismatch batch=%d index=%d cpu=%g cuda=%g\n",
                        b, i, expected[i], found);
                tb_cuda_feature_batch_destroy(gpu);
                free(obs); free(mission); free(cameras); free(actual);
                return 1;
            }
        }
    }
    tb_cuda_feature_batch_destroy(gpu);
    free(obs); free(mission); free(cameras); free(actual);
    puts("CUDA feature batch matches C11 extraction");
    return 0;
}
