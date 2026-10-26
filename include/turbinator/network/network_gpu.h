#ifndef TURBINATOR_NETWORK_NETWORK_GPU_H
#define TURBINATOR_NETWORK_NETWORK_GPU_H

#include "turbinator/network/network.h"
#include "turbinator/gpu/ops.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tb_net_gpu tb_net_gpu;

/* The caller owns contiguous FP32 parameter and gradient buffers on the
 * selected CUDA device. One context retains intermediates for one forward /
 * backward pair and may be reused at any batch size up to max_batch. */
tb_net_gpu *tb_net_gpu_create(const tb_net_model *layout, tb_gpu_ops *ops,
                              float *device_parameters, float *device_gradients,
                              int max_batch);
void tb_net_gpu_destroy(tb_net_gpu *ctx);
const char *tb_net_gpu_last_error(const tb_net_gpu *ctx);

int tb_net_gpu_actor_forward(tb_net_gpu *ctx,
                             const float *device_obs_4148,
                             const float *device_mission_10,
                             const uint8_t *device_cameras_16384,
                             int batch,
                             float *device_action_15,
                             float *device_log_std_15,
                             float *device_summary_3072,
                             float *device_aux_9,
                             float *device_world_1552);
int tb_net_gpu_actor_backward(tb_net_gpu *ctx,
                              const float *device_d_action_15,
                              const float *device_d_log_std_15,
                              const float *device_d_summary_3072,
                              const float *device_d_aux_9,
                              const float *device_d_world_1552);

int tb_net_gpu_critic_forward(tb_net_gpu *ctx,
                              const float *device_summary_3072,
                              const float *device_privileged_64,
                              const float *device_action_15,
                              int batch,
                              float *device_logits_q1_64,
                              float *device_logits_q2_64);
int tb_net_gpu_critic_backward(tb_net_gpu *ctx,
                               const float *device_d_logits_q1_64,
                               const float *device_d_logits_q2_64,
                               float *device_d_summary_3072,
                               float *device_d_action_15,
                               int accumulate_parameter_gradients);

#ifdef __cplusplus
}
#endif
#endif
