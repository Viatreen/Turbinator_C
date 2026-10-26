#ifndef TURBINATOR_GPU_TRAINER_H
#define TURBINATOR_GPU_TRAINER_H

#include <stddef.h>
#include <stdint.h>
#include "turbinator/gpu/replay.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TB_GPU_TRAINER_MAX_DEVICES 4

typedef struct tb_gpu_trainer tb_gpu_trainer;

typedef struct {
    int device_count;
    int device_ids[TB_GPU_TRAINER_MAX_DEVICES];
    int per_device_batch;
    size_t per_device_replay_capacity;
    uint64_t seed;
    float gamma;
    float tau;
    float actor_learning_rate;
    float critic_learning_rate;
    float alpha_learning_rate;
    float initial_alpha;
    float target_entropy;
    float maximum_gradient_norm;
    float weight_decay;
    int synchronization_interval;
} tb_gpu_train_config;

typedef struct {
    float critic_loss;
    float actor_loss;
    float auxiliary_loss;
    float q_mean;
    float alpha;
    float gradient_norm;
    uint64_t updates;
    size_t replay_size;
    int active_devices;
} tb_gpu_train_metrics;

void tb_gpu_train_config_default(tb_gpu_train_config *out);
tb_gpu_trainer *tb_gpu_trainer_create(const tb_gpu_train_config *config);
void tb_gpu_trainer_destroy(tb_gpu_trainer *trainer);
const char *tb_gpu_trainer_last_error(const tb_gpu_trainer *trainer);
uint64_t tb_gpu_trainer_parameter_count(const tb_gpu_trainer *trainer);
uint64_t tb_gpu_trainer_update_count(const tb_gpu_trainer *trainer);
size_t tb_gpu_trainer_replay_size(const tb_gpu_trainer *trainer);
int tb_gpu_trainer_device_count(const tb_gpu_trainer *trainer);

/* The host batch is split across device-local replay shards. */
int tb_gpu_trainer_add(tb_gpu_trainer *trainer, const tb_gpu_host_batch *host, size_t count);

/* Performs one learner update on each active GPU; returns 1 if gradients were
 * applied, 0 when replay is not ready or an error occurred. */
int tb_gpu_trainer_update(tb_gpu_trainer *trainer, tb_gpu_train_metrics *metrics);

/* Deterministic normalized action inference for a host batch. */
int tb_gpu_trainer_act(tb_gpu_trainer *trainer, const float *obs_4148,
                       const float *mission_10, const uint8_t *cameras_16384,
                       int batch, float *out_action_15);

/* Saves parameter, target, optimizer and trainer metadata files into a
 * caller-supplied directory. A loaded trainer starts with empty replay. */
int tb_gpu_trainer_save(tb_gpu_trainer *trainer, const char *directory);
int tb_gpu_trainer_load(tb_gpu_trainer *trainer, const char *directory);

#ifdef __cplusplus
}
#endif
#endif
