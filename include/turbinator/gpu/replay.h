#ifndef TURBINATOR_GPU_REPLAY_H
#define TURBINATOR_GPU_REPLAY_H

#include <stddef.h>
#include <stdint.h>
#include "turbinator/gpu/ops.h"
#include "turbinator/training/training.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tb_gpu_replay tb_gpu_replay;

/* Host arrays have leading dimension `count`. All pointers must be non-NULL.
 * Cameras are uint8, all other fields are float. */
typedef struct {
    const float *obs;
    const float *mission;
    const uint8_t *cameras;
    const float *privileged;
    const float *actions;
    const float *reference_actions;
    const float *rewards;
    const float *next_obs;
    const float *next_mission;
    const uint8_t *next_cameras;
    const float *next_privileged;
    const float *dones;
} tb_gpu_host_batch;

typedef struct {
    float *obs;
    float *mission;
    uint8_t *cameras;
    float *privileged;
    float *actions;
    float *reference_actions;
    float *rewards;
    float *next_obs;
    float *next_mission;
    uint8_t *next_cameras;
    float *next_privileged;
    float *dones;
    size_t count;
} tb_gpu_sample_batch;

tb_gpu_replay *tb_gpu_replay_create(tb_gpu_ops *ops, size_t capacity, uint64_t seed);
void tb_gpu_replay_destroy(tb_gpu_replay *replay);
size_t tb_gpu_replay_size(const tb_gpu_replay *replay);
size_t tb_gpu_replay_capacity(const tb_gpu_replay *replay);
int tb_gpu_replay_add(tb_gpu_replay *replay, const tb_gpu_host_batch *host, size_t count);
tb_gpu_sample_batch *tb_gpu_sample_batch_create(tb_gpu_ops *ops, size_t count);
void tb_gpu_sample_batch_destroy(tb_gpu_ops *ops, tb_gpu_sample_batch *batch);
int tb_gpu_replay_sample(tb_gpu_replay *replay, tb_gpu_sample_batch *output);

#ifdef __cplusplus
}
#endif
#endif
