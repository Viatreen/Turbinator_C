#ifndef TURBINATOR_GPU_OPTIMIZER_H
#define TURBINATOR_GPU_OPTIMIZER_H

#include <stddef.h>
#include <stdint.h>
#include "turbinator/gpu/ops.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tb_gpu_rmsprop tb_gpu_rmsprop;

/* One-moment RMSprop is used to leave room for network, target and replay
 * buffers on 16 GB V100 devices. Gradients are clipped by global L2 norm. */
tb_gpu_rmsprop *tb_gpu_rmsprop_create(tb_gpu_ops *ops, size_t parameter_count,
                                     float learning_rate, float decay,
                                     float epsilon, float weight_decay,
                                     float maximum_gradient_norm);
void tb_gpu_rmsprop_destroy(tb_gpu_rmsprop *optimizer);
int tb_gpu_rmsprop_step(tb_gpu_rmsprop *optimizer, float *parameters,
                        float *gradients, float *out_gradient_norm);
uint64_t tb_gpu_rmsprop_steps(const tb_gpu_rmsprop *optimizer);
float *tb_gpu_rmsprop_moments(tb_gpu_rmsprop *optimizer);
int tb_gpu_rmsprop_set_lr(tb_gpu_rmsprop *optimizer, float learning_rate);

/* Atomic target updates: target = (1 - tau) target + tau source. */
int tb_gpu_ema_update(tb_gpu_ops *ops, float *target,
                      const float *source, size_t count, float tau);

/* Files contain optimizer moments only. Parameters are saved by the model
 * checkpoint layer. The schema is versioned and validates parameter count. */
int tb_gpu_rmsprop_save(const tb_gpu_rmsprop *optimizer, const char *path);
int tb_gpu_rmsprop_load(tb_gpu_rmsprop *optimizer, const char *path);

#ifdef __cplusplus
}
#endif
#endif
