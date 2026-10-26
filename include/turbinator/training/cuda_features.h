#ifndef TURBINATOR_CUDA_FEATURES_H
#define TURBINATOR_CUDA_FEATURES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Device pointers. Layouts are [batch, 4148], [batch, 10],
 * [batch, 16384], and [batch, 142]. stream is a cudaStream_t or NULL.
 * Returns 1 if the kernel launch succeeded; synchronize to catch runtime
 * errors before consuming the output. Built only when CUDA is available. */
int tb_cuda_extract_features_device(const float *device_obs,
                                    const float *device_mission,
                                    const unsigned char *device_cameras,
                                    float *device_features,
                                    size_t batch_size,
                                    void *stream);
int tb_cuda_synchronize(void *stream);

typedef struct tb_cuda_feature_batch tb_cuda_feature_batch;

/* Persistent staging buffers keep the compact C11 learner within laptop GPU
 * memory while extracting all 142 sensor features in one CUDA launch. */
tb_cuda_feature_batch *tb_cuda_feature_batch_create(int device_id, size_t max_batch);
void tb_cuda_feature_batch_destroy(tb_cuda_feature_batch *batch);
const char *tb_cuda_feature_batch_error(const tb_cuda_feature_batch *batch);
int tb_cuda_extract_features_host(tb_cuda_feature_batch *batch,
                                  const float *obs_4148,
                                  const float *mission_10,
                                  const unsigned char *cameras_16384,
                                  size_t count,
                                  float *out_features_142);

#ifdef __cplusplus
}
#endif

#endif
