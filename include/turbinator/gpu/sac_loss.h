#ifndef TURBINATOR_GPU_SAC_LOSS_H
#define TURBINATOR_GPU_SAC_LOSS_H

#include "turbinator/gpu/ops.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float gamma;
    float alpha;
    float value_min;
    float value_max;
    float gaussian_sigma_ratio;
    float q_weight;
    float q_scale;
    float reference_weight;
    float aux_weight;
    float world_weight;
} tb_gpu_sac_loss_config;

void tb_gpu_sac_loss_config_default(tb_gpu_sac_loss_config *out);

/* All pointers are CUDA device buffers. Each sample has two 64-atom critic
 * distributions. The target uses clipped double-Q and an entropy term. */
int tb_gpu_sac_build_targets(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg,
    const float *target_logits_q1, const float *target_logits_q2,
    const float *next_action_15, const float *next_logstd_15,
    const float *rewards, const float *dones, int batch,
    float *target_probabilities_64);

/* Device scalar outputs are optional. Caller zeros them before use. */
int tb_gpu_sac_critic_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg,
    const float *logits_q1, const float *logits_q2,
    const float *target_probabilities_64, int batch,
    float *d_logits_q1, float *d_logits_q2,
    float *device_loss_scalar, float *device_q_mean_scalar);

/* Generates distribution gradients for policy Q, reference guidance, and
 * entropy gradients. d_action is combined with critic d_action by caller. */
int tb_gpu_sac_policy_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg,
    const float *logits_q1, const float *logits_q2,
    const float *action_15, const float *logstd_15,
    const float *reference_action_15, int batch,
    float *d_logits_q1, float *d_logits_q2,
    float *d_action_15, float *d_logstd_15,
    float *device_loss_scalar);

/* Privileged state is used only to build training targets, never as actor
 * input. World loss skips true terminal transitions. */
int tb_gpu_sac_aux_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg,
    const float *privileged_64, const float *obs_4148,
    const float *next_obs_4148, const float *next_summary_3072,
    const float *rewards, const float *dones,
    const float *aux_prediction_9, const float *world_prediction_1552,
    int batch, float *d_aux_9, float *d_world_1552,
    float *device_loss_scalar);

/* Computes mean policy entropy from Gaussian control heads and Bernoulli
 * shoot head; device scalar is overwritten. */
int tb_gpu_sac_mean_entropy(tb_gpu_ops *ops, const float *action_15,
                            const float *logstd_15, int batch,
                            float *device_mean_entropy_scalar);

#ifdef __cplusplus
}
#endif
#endif
