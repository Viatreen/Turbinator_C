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

#ifdef __cplusplus
}
#endif

#endif
