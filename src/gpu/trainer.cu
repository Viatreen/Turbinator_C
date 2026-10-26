#include "turbinator/gpu/trainer.h"
#include "turbinator/gpu/optimizer.h"
#include "turbinator/gpu/sac_loss.h"
#include "turbinator/network/network.h"
#include "turbinator/network/network_gpu.h"

#include <cuda_runtime.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define tb_mkdir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define tb_mkdir(path) mkdir((path), 0775)
#endif

typedef struct {
    tb_gpu_ops *ops;
    float *parameters;
    float *gradients;
    float *target_parameters;
    tb_net_gpu *current;
    tb_net_gpu *target;
    tb_gpu_rmsprop *actor_optimizer;
    tb_gpu_rmsprop *critic_optimizer;
    tb_gpu_replay *replay;
    tb_gpu_sample_batch *sample;
    float *action, *logstd, *summary, *aux, *world;
    float *next_action, *next_logstd, *next_summary;
    float *q1, *q2, *target_q1, *target_q2;
    float *dq1, *dq2, *target_probs;
    float *d_action, *d_action_q, *d_action_unused, *d_logstd;
    float *d_summary_policy, *d_summary_critic;
    float *d_aux, *d_world;
    float *device_stats;
    int device_id;
} TbGpuShard;

struct tb_gpu_trainer {
    tb_gpu_train_config cfg;
    tb_net_model *layout;
    TbGpuShard shards[TB_GPU_TRAINER_MAX_DEVICES];
    int device_count;
    uint64_t updates;
    float log_alpha;
    char error[256];
};

static int tb_error(tb_gpu_trainer *t, const char *message) {
    if (t) snprintf(t->error, sizeof(t->error), "%s", message);
    return 0;
}

extern "C" void tb_gpu_train_config_default(tb_gpu_train_config *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->device_count = 0;
    out->per_device_batch = 2;
    out->per_device_replay_capacity = 512;
    out->seed = 42;
    out->gamma = 0.992f;
    out->tau = 0.005f;
    out->actor_learning_rate = 1.5e-4f;
    out->critic_learning_rate = 3.0e-4f;
    out->alpha_learning_rate = 5.0e-5f;
    out->initial_alpha = 0.05f;
    out->target_entropy = 12.75f;
    out->maximum_gradient_norm = 1.5f;
    out->weight_decay = 1.0e-4f;
    out->synchronization_interval = 32;
}

static int tb_resolve_devices(tb_gpu_trainer *t) {
    int available = tb_gpu_device_count();
    if (available <= 0) return tb_error(t, "no CUDA device available");
    int wanted = t->cfg.device_count;
    if (wanted < 0 || wanted > TB_GPU_TRAINER_MAX_DEVICES) return tb_error(t, "invalid device count");
    if (wanted == 0) {
        for (int i = 0; i < available && t->device_count < TB_GPU_TRAINER_MAX_DEVICES; ++i) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
            if (prop.major < 7 || prop.totalGlobalMem < UINT64_C(12) * 1024 * 1024 * 1024) continue;
            t->shards[t->device_count].device_id = i;
            ++t->device_count;
        }
    } else {
        for (int i = 0; i < wanted; ++i) {
            int device = t->cfg.device_ids[i];
            if (device < 0 || device >= available) return tb_error(t, "requested CUDA device unavailable");
            for (int j = 0; j < i; ++j) if (t->cfg.device_ids[j] == device)
                return tb_error(t, "duplicate CUDA device id");
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, device) != cudaSuccess || prop.major < 7)
                return tb_error(t, "CUDA device requires compute capability 7.0 or newer");
            t->shards[i].device_id = device;
            ++t->device_count;
        }
    }
    if (t->device_count == 0) return tb_error(t, "no suitable GPU with at least 12 GB of memory");
    return 1;
}

static float *tb_alloc_float(tb_gpu_ops *ops, size_t count) {
    return (float *)tb_gpu_malloc(ops, count * sizeof(float));
}
static int tb_shard_alloc_scratch(tb_gpu_trainer *t, TbGpuShard *s) {
    size_t b = (size_t)t->cfg.per_device_batch;
#define TB_SCRATCH(field, count) s->field = tb_alloc_float(s->ops, (count)); if (!s->field) return 0
    TB_SCRATCH(action, b * 15); TB_SCRATCH(logstd, b * 15);
    TB_SCRATCH(summary, b * 3072); TB_SCRATCH(aux, b * 9); TB_SCRATCH(world, b * 1552);
    TB_SCRATCH(next_action, b * 15); TB_SCRATCH(next_logstd, b * 15);
    TB_SCRATCH(next_summary, b * 3072);
    TB_SCRATCH(q1, b * 64); TB_SCRATCH(q2, b * 64);
    TB_SCRATCH(target_q1, b * 64); TB_SCRATCH(target_q2, b * 64);
    TB_SCRATCH(dq1, b * 64); TB_SCRATCH(dq2, b * 64);
    TB_SCRATCH(target_probs, b * 64);
    TB_SCRATCH(d_action, b * 15); TB_SCRATCH(d_action_q, b * 15);
    TB_SCRATCH(d_action_unused, b * 15); TB_SCRATCH(d_logstd, b * 15);
    TB_SCRATCH(d_summary_policy, b * 3072); TB_SCRATCH(d_summary_critic, b * 3072);
    TB_SCRATCH(d_aux, b * 9); TB_SCRATCH(d_world, b * 1552);
    TB_SCRATCH(device_stats, 5);
#undef TB_SCRATCH
    return 1;
}

static void tb_shard_destroy(TbGpuShard *s) {
    if (!s || !s->ops) return;
    tb_net_gpu_destroy(s->current);
    tb_net_gpu_destroy(s->target);
    tb_gpu_sample_batch_destroy(s->ops, s->sample);
    tb_gpu_replay_destroy(s->replay);
    tb_gpu_rmsprop_destroy(s->actor_optimizer);
    tb_gpu_rmsprop_destroy(s->critic_optimizer);
#define TB_RELEASE(field) tb_gpu_free(s->ops, s->field)
    TB_RELEASE(action); TB_RELEASE(logstd); TB_RELEASE(summary); TB_RELEASE(aux); TB_RELEASE(world);
    TB_RELEASE(next_action); TB_RELEASE(next_logstd); TB_RELEASE(next_summary);
    TB_RELEASE(q1); TB_RELEASE(q2); TB_RELEASE(target_q1); TB_RELEASE(target_q2);
    TB_RELEASE(dq1); TB_RELEASE(dq2); TB_RELEASE(target_probs);
    TB_RELEASE(d_action); TB_RELEASE(d_action_q); TB_RELEASE(d_action_unused); TB_RELEASE(d_logstd);
    TB_RELEASE(d_summary_policy); TB_RELEASE(d_summary_critic); TB_RELEASE(d_aux); TB_RELEASE(d_world);
    TB_RELEASE(device_stats); TB_RELEASE(parameters); TB_RELEASE(gradients); TB_RELEASE(target_parameters);
#undef TB_RELEASE
    tb_gpu_ops_destroy(s->ops);
    memset(s, 0, sizeof(*s));
}

extern "C" void tb_gpu_trainer_destroy(tb_gpu_trainer *t) {
    if (!t) return;
    for (int i = 0; i < t->device_count; ++i) tb_shard_destroy(&t->shards[i]);
    tb_net_destroy(t->layout);
    free(t);
}

static int tb_shard_create(tb_gpu_trainer *t, TbGpuShard *s) {
    s->ops = tb_gpu_ops_create(s->device_id);
    if (!s->ops) return tb_error(t, "failed to create CUDA/cuBLAS context");
    uint64_t count = tb_net_parameter_count(t->layout);
    s->parameters = tb_alloc_float(s->ops, (size_t)count);
    s->gradients = tb_alloc_float(s->ops, (size_t)count);
    s->target_parameters = tb_alloc_float(s->ops, (size_t)count);
    if (!s->parameters || !s->gradients || !s->target_parameters)
        return tb_error(t, "insufficient GPU memory for 618M parameters, gradients and target");
    if (!tb_gpu_zero(s->ops, s->gradients, (size_t)count)) return tb_error(t, "gradient initialization failed");
    uint64_t actor = tb_net_actor_parameter_count(t->layout);
    uint64_t critic = tb_net_critic_parameter_count(t->layout);
    s->actor_optimizer = tb_gpu_rmsprop_create(s->ops, (size_t)actor,
        t->cfg.actor_learning_rate, 0.99f, 1.0e-8f, t->cfg.weight_decay,
        t->cfg.maximum_gradient_norm);
    s->critic_optimizer = tb_gpu_rmsprop_create(s->ops, (size_t)critic,
        t->cfg.critic_learning_rate, 0.99f, 1.0e-8f, t->cfg.weight_decay,
        t->cfg.maximum_gradient_norm);
    if (!s->actor_optimizer || !s->critic_optimizer)
        return tb_error(t, "insufficient GPU memory for optimizer states");
    s->replay = tb_gpu_replay_create(s->ops, t->cfg.per_device_replay_capacity,
                                    t->cfg.seed + (uint64_t)s->device_id);
    s->sample = tb_gpu_sample_batch_create(s->ops, (size_t)t->cfg.per_device_batch);
    if (!s->replay || !s->sample || !tb_shard_alloc_scratch(t, s))
        return tb_error(t, "insufficient GPU memory for replay or minibatch buffers");
    s->current = tb_net_gpu_create(t->layout, s->ops, s->parameters, s->gradients,
                                   t->cfg.per_device_batch);
    s->target = tb_net_gpu_create(t->layout, s->ops, s->target_parameters, NULL,
                                  t->cfg.per_device_batch);
    if (!s->current || !s->target) return tb_error(t, "failed to create 618M network contexts");
    return 1;
}

static int tb_initialize_all(tb_gpu_trainer *t) {
    const size_t chunk = 1u << 20;
    float *host = (float *)malloc(chunk * sizeof(float));
    if (!host) return tb_error(t, "host initialization buffer allocation failed");
    uint64_t total = tb_net_parameter_count(t->layout);
    int ok = 1;
    for (uint64_t pos = 0; ok && pos < total; pos += chunk) {
        size_t n = total - pos < chunk ? (size_t)(total - pos) : chunk;
        ok = tb_net_initialize_slice(t->layout, pos, host, n, t->cfg.seed);
        for (int d = 0; ok && d < t->device_count; ++d) {
            TbGpuShard *s = &t->shards[d];
            ok = tb_gpu_h2d(s->ops, s->parameters + pos, host, n * sizeof(float)) &&
                 tb_gpu_h2d(s->ops, s->target_parameters + pos, host, n * sizeof(float));
        }
    }
    free(host);
    return ok ? 1 : tb_error(t, "model initialization failed");
}

extern "C" tb_gpu_trainer *tb_gpu_trainer_create(const tb_gpu_train_config *config) {
    tb_gpu_train_config defaults;
    tb_gpu_train_config_default(&defaults);
    if (!config) config = &defaults;
    if (config->per_device_batch < 1 || config->per_device_batch > 1024 ||
        config->per_device_replay_capacity < (size_t)config->per_device_batch ||
        !isfinite(config->gamma) || config->gamma <= 0 || config->gamma > 1 ||
        !isfinite(config->tau) || config->tau <= 0 || config->tau > 1 ||
        !isfinite(config->initial_alpha) || config->initial_alpha <= 0) return NULL;
    tb_gpu_trainer *t = (tb_gpu_trainer *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->cfg = *config;
    t->log_alpha = logf(config->initial_alpha);
    t->layout = tb_net_create_618m();
    if (!t->layout || !tb_net_validate_layout(t->layout) || !tb_resolve_devices(t)) {
        tb_gpu_trainer_destroy(t); return NULL;
    }
    for (int d = 0; d < t->device_count; ++d) {
        if (!tb_shard_create(t, &t->shards[d])) { tb_gpu_trainer_destroy(t); return NULL; }
    }
    if (!tb_initialize_all(t)) { tb_gpu_trainer_destroy(t); return NULL; }
    return t;
}

extern "C" const char *tb_gpu_trainer_last_error(const tb_gpu_trainer *t) {
    return t ? t->error : "GPU trainer unavailable";
}
extern "C" uint64_t tb_gpu_trainer_parameter_count(const tb_gpu_trainer *t) {
    return t ? tb_net_parameter_count(t->layout) : 0;
}
extern "C" uint64_t tb_gpu_trainer_update_count(const tb_gpu_trainer *t) { return t ? t->updates : 0; }
extern "C" int tb_gpu_trainer_device_count(const tb_gpu_trainer *t) { return t ? t->device_count : 0; }
extern "C" size_t tb_gpu_trainer_replay_size(const tb_gpu_trainer *t) {
    if (!t) return 0;
    size_t sum = 0;
    for (int i = 0; i < t->device_count; ++i) sum += tb_gpu_replay_size(t->shards[i].replay);
    return sum;
}

static tb_gpu_host_batch tb_host_offset(const tb_gpu_host_batch *h, size_t offset) {
    tb_gpu_host_batch out = *h;
    out.obs += offset * TB_TRAIN_OBSERVATION_DIM;
    out.mission += offset * TB_TRAIN_MISSION_DIM;
    out.cameras += offset * TB_TRAIN_CAMERA_PIXELS;
    out.privileged += offset * TB_TRAIN_PRIVILEGED_DIM;
    out.actions += offset * TB_TRAIN_ACTION_DIM;
    out.reference_actions += offset * TB_TRAIN_ACTION_DIM;
    out.rewards += offset;
    out.next_obs += offset * TB_TRAIN_OBSERVATION_DIM;
    out.next_mission += offset * TB_TRAIN_MISSION_DIM;
    out.next_cameras += offset * TB_TRAIN_CAMERA_PIXELS;
    out.next_privileged += offset * TB_TRAIN_PRIVILEGED_DIM;
    out.dones += offset;
    return out;
}
extern "C" int tb_gpu_trainer_add(tb_gpu_trainer *t, const tb_gpu_host_batch *host, size_t count) {
    if (!t || !host || !count) return 0;
    size_t offset = 0;
    for (int d = 0; d < t->device_count; ++d) {
        size_t n = count / (size_t)t->device_count + ((size_t)d < count % (size_t)t->device_count);
        if (!n) continue;
        tb_gpu_host_batch part = tb_host_offset(host, offset);
        if (!tb_gpu_replay_add(t->shards[d].replay, &part, n))
            return tb_error(t, "failed to add transitions to GPU replay shard");
        offset += n;
    }
    return 1;
}

extern "C" int tb_gpu_trainer_act(tb_gpu_trainer *t, const float *obs,
                                   const float *mission, const uint8_t *cameras,
                                   int batch, float *out_action) {
    if (!t || !obs || !mission || !cameras || !out_action || batch <= 0) return 0;
    TbGpuShard *s = &t->shards[0];
    int max_batch = t->cfg.per_device_batch;
    for (int pos = 0; pos < batch; pos += max_batch) {
        int n = batch - pos < max_batch ? batch - pos : max_batch;
        if (!tb_gpu_h2d(s->ops, s->sample->obs, obs + (size_t)pos * TB_TRAIN_OBSERVATION_DIM,
                        (size_t)n * TB_TRAIN_OBSERVATION_DIM * sizeof(float)) ||
            !tb_gpu_h2d(s->ops, s->sample->mission, mission + (size_t)pos * TB_TRAIN_MISSION_DIM,
                        (size_t)n * TB_TRAIN_MISSION_DIM * sizeof(float)) ||
            !tb_gpu_h2d(s->ops, s->sample->cameras, cameras + (size_t)pos * TB_TRAIN_CAMERA_PIXELS,
                        (size_t)n * TB_TRAIN_CAMERA_PIXELS) ||
            !tb_net_gpu_actor_forward(s->current, s->sample->obs, s->sample->mission,
                                      s->sample->cameras, n, s->action, s->logstd,
                                      s->summary, NULL, NULL) ||
            !tb_gpu_d2h(s->ops, out_action + (size_t)pos * TB_TRAIN_ACTION_DIM,
                        s->action, (size_t)n * TB_TRAIN_ACTION_DIM * sizeof(float)))
            return tb_error(t, "GPU actor inference failed");
    }
    return 1;
}

static int tb_update_one(tb_gpu_trainer *t, TbGpuShard *s,
                         tb_gpu_train_metrics *metrics, float *entropy_sum) {
    int batch = t->cfg.per_device_batch;
    uint64_t total = tb_net_parameter_count(t->layout);
    uint64_t actor_count = tb_net_actor_parameter_count(t->layout);
    uint64_t critic_count = tb_net_critic_parameter_count(t->layout);
    tb_gpu_sac_loss_config loss_cfg;
    tb_gpu_sac_loss_config_default(&loss_cfg);
    loss_cfg.gamma = t->cfg.gamma;
    loss_cfg.alpha = expf(t->log_alpha);
    loss_cfg.q_weight = t->updates >= 1000 ? 0.05f : 0.0f;
    loss_cfg.reference_weight = t->updates < 100000 ? 0.85f : 0.35f;
    if (!tb_gpu_replay_sample(s->replay, s->sample) ||
        !tb_gpu_zero(s->ops, s->gradients, (size_t)total) ||
        !tb_gpu_zero(s->ops, s->device_stats, 5))
        return tb_error(t, "GPU replay sampling or gradient reset failed");

    tb_gpu_sample_batch *b = s->sample;
    /* Bellman target uses target actor and target critics, with no gradients. */
    if (!tb_net_gpu_actor_forward(s->target, b->next_obs, b->next_mission,
                                  b->next_cameras, batch, s->next_action,
                                  s->next_logstd, s->next_summary, NULL, NULL) ||
        !tb_net_gpu_critic_forward(s->target, s->next_summary, b->next_privileged,
                                   s->next_action, batch, s->target_q1, s->target_q2) ||
        !tb_gpu_sac_build_targets(s->ops, &loss_cfg, s->target_q1, s->target_q2,
                                  s->next_action, s->next_logstd,
                                  b->rewards, b->dones, batch, s->target_probs))
        return tb_error(t, "target actor, critic or Bellman projection failed");

    if (!tb_net_gpu_actor_forward(s->current, b->obs, b->mission, b->cameras,
                                  batch, s->action, s->logstd, s->summary,
                                  s->aux, s->world) ||
        !tb_net_gpu_critic_forward(s->current, s->summary, b->privileged,
                                   s->action, batch, s->q1, s->q2) ||
        !tb_gpu_sac_policy_grad(s->ops, &loss_cfg, s->q1, s->q2,
                                s->action, s->logstd, b->reference_actions,
                                batch, s->dq1, s->dq2, s->d_action,
                                s->d_logstd, s->device_stats + 1) ||
        !tb_net_gpu_critic_backward(s->current, s->dq1, s->dq2,
                                    s->d_summary_policy, s->d_action_q, 0) ||
        !tb_gpu_axpy(s->ops, s->d_action, s->d_action_q, 1.0f,
                     (size_t)batch * TB_TRAIN_ACTION_DIM))
        return tb_error(t, "SAC policy loss or critic action gradient failed");

    if (!tb_gpu_sac_aux_grad(s->ops, &loss_cfg, b->privileged, b->obs,
                             b->next_obs, s->next_summary, b->rewards, b->dones,
                             s->aux, s->world, batch, s->d_aux, s->d_world,
                             s->device_stats + 2))
        return tb_error(t, "auxiliary or world-model loss failed");

    /* Critic value loss uses replay actions; actor trace remains live. */
    if (!tb_net_gpu_critic_forward(s->current, s->summary, b->privileged,
                                   b->actions, batch, s->q1, s->q2) ||
        !tb_gpu_sac_critic_grad(s->ops, &loss_cfg, s->q1, s->q2,
                                s->target_probs, batch, s->dq1, s->dq2,
                                s->device_stats, s->device_stats + 3) ||
        !tb_net_gpu_critic_backward(s->current, s->dq1, s->dq2,
                                    s->d_summary_critic, s->d_action_unused, 1))
        return tb_error(t, "distributional critic loss or backward failed");

    if (!tb_gpu_scale(s->ops, s->d_summary_policy, 0.25f,
                      (size_t)batch * 3072) ||
        !tb_gpu_axpy(s->ops, s->d_summary_policy, s->d_summary_critic, 0.25f,
                     (size_t)batch * 3072) ||
        !tb_net_gpu_actor_backward(s->current, s->d_action, s->d_logstd,
                                   s->d_summary_policy, s->d_aux, s->d_world) ||
        !tb_gpu_sac_mean_entropy(s->ops, s->action, s->logstd,
                                 batch, s->device_stats + 4))
        return tb_error(t, "actor encoder backward or entropy computation failed");

    float stats[5] = {0};
    if (!tb_gpu_d2h(s->ops, stats, s->device_stats, sizeof(stats)))
        return tb_error(t, "failed to read GPU loss metrics");
    for (int i = 0; i < 5; ++i) if (!isfinite(stats[i]))
        return tb_error(t, "nonfinite GPU loss or entropy");
    float actor_norm = 0.0f, critic_norm = 0.0f;
    if (!tb_gpu_rmsprop_step(s->actor_optimizer, s->parameters, s->gradients,
                             &actor_norm) ||
        !tb_gpu_rmsprop_step(s->critic_optimizer,
                             s->parameters + actor_count,
                             s->gradients + actor_count, &critic_norm) ||
        !tb_gpu_ema_update(s->ops, s->target_parameters, s->parameters,
                           (size_t)total, t->cfg.tau))
        return tb_error(t, "GPU optimizer or target update failed");
    (void)critic_count;
    metrics->critic_loss += stats[0];
    metrics->actor_loss += stats[1];
    metrics->auxiliary_loss += stats[2];
    metrics->q_mean += stats[3];
    metrics->gradient_norm += 0.5f * (actor_norm + critic_norm);
    *entropy_sum += stats[4];
    return 1;
}

/* CPU-mediated replica synchronization works on Windows, where the CUDA
 * toolkit does not provide NCCL. We synchronize parameters, target weights,
 * and RMSprop moments together so replicas resume from identical states. */
static int tb_average_buffer(tb_gpu_trainer *t, float **buffers, uint64_t count) {
    const size_t chunk = 1u << 18;
    float *all = (float *)malloc(chunk * (size_t)t->device_count * sizeof(float));
    float *mean = (float *)malloc(chunk * sizeof(float));
    if (!all || !mean) { free(all); free(mean); return tb_error(t, "replica synchronization buffer allocation failed"); }
    int ok = 1;
    for (uint64_t pos = 0; ok && pos < count; pos += chunk) {
        size_t n = count - pos < chunk ? (size_t)(count - pos) : chunk;
        for (int d = 0; d < t->device_count; ++d) {
            if (!tb_gpu_d2h(t->shards[d].ops, all + (size_t)d * chunk,
                            buffers[d] + pos, n * sizeof(float))) { ok = 0; break; }
        }
        if (!ok) break;
        for (size_t i = 0; i < n; ++i) {
            double sum = 0;
            for (int d = 0; d < t->device_count; ++d) sum += all[(size_t)d * chunk + i];
            mean[i] = (float)(sum / t->device_count);
        }
        for (int d = 0; d < t->device_count; ++d) {
            if (!tb_gpu_h2d(t->shards[d].ops, buffers[d] + pos, mean,
                            n * sizeof(float))) { ok = 0; break; }
        }
    }
    free(all); free(mean);
    return ok ? 1 : tb_error(t, "replica synchronization transfer failed");
}

static int tb_synchronize_replicas(tb_gpu_trainer *t) {
    if (t->device_count <= 1) return 1;
    float *buffers[TB_GPU_TRAINER_MAX_DEVICES];
    for (int d = 0; d < t->device_count; ++d) buffers[d] = t->shards[d].parameters;
    if (!tb_average_buffer(t, buffers, tb_net_parameter_count(t->layout))) return 0;
    for (int d = 0; d < t->device_count; ++d) buffers[d] = t->shards[d].target_parameters;
    if (!tb_average_buffer(t, buffers, tb_net_parameter_count(t->layout))) return 0;
    for (int d = 0; d < t->device_count; ++d)
        buffers[d] = tb_gpu_rmsprop_moments(t->shards[d].actor_optimizer);
    if (!tb_average_buffer(t, buffers, tb_net_actor_parameter_count(t->layout))) return 0;
    for (int d = 0; d < t->device_count; ++d)
        buffers[d] = tb_gpu_rmsprop_moments(t->shards[d].critic_optimizer);
    return tb_average_buffer(t, buffers, tb_net_critic_parameter_count(t->layout));
}

extern "C" int tb_gpu_trainer_update(tb_gpu_trainer *t, tb_gpu_train_metrics *out) {
    if (!t || !out) return 0;
    t->error[0] = 0;
    for (int d = 0; d < t->device_count; ++d)
        if (tb_gpu_replay_size(t->shards[d].replay) < (size_t)t->cfg.per_device_batch) return 0;
    tb_gpu_train_metrics m = {0};
    float entropy_sum = 0.0f;
    for (int d = 0; d < t->device_count; ++d)
        if (!tb_update_one(t, &t->shards[d], &m, &entropy_sum)) return 0;
    ++t->updates;
    float entropy = entropy_sum / t->device_count;
    t->log_alpha += t->cfg.alpha_learning_rate * (t->cfg.target_entropy - entropy);
    t->log_alpha = fmaxf(logf(0.005f), fminf(logf(0.2f), t->log_alpha));
    if (t->device_count > 1 && t->cfg.synchronization_interval > 0 &&
        t->updates % (uint64_t)t->cfg.synchronization_interval == 0 &&
        !tb_synchronize_replicas(t)) return 0;
    m.critic_loss /= t->device_count;
    m.actor_loss /= t->device_count;
    m.auxiliary_loss /= t->device_count;
    m.q_mean /= t->device_count;
    m.gradient_norm /= t->device_count;
    m.alpha = expf(t->log_alpha);
    m.updates = t->updates;
    m.replay_size = tb_gpu_trainer_replay_size(t);
    m.active_devices = t->device_count;
    *out = m;
    return 1;
}

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t checksum;
    uint64_t count;
} TbWeightHeader;

typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t parameters;
    uint64_t updates;
    float log_alpha;
    uint32_t checksum;
} TbTrainerHeader;

static uint32_t tb_hash_bytes(uint32_t h, const unsigned char *bytes, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        h ^= bytes[i];
        h *= UINT32_C(16777619);
    }
    return h;
}

static int tb_checkpoint_path(char *out, size_t capacity,
                              const char *directory, const char *filename) {
    int n = snprintf(out, capacity, "%s/%s", directory, filename);
    return n > 0 && (size_t)n < capacity;
}

static int tb_weight_file(const char *path, tb_gpu_ops *ops, float *device,
                          uint64_t count, int save) {
    FILE *f = fopen(path, save ? "wb" : "rb");
    if (!f) return 0;
    const size_t chunk = 1u << 20;
    float *host = (float *)malloc(chunk * sizeof(float));
    TbWeightHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "TBWGHT1", 7);
    header.version = 1;
    header.count = count;
    int ok = host != NULL;
    if (save) {
        uint32_t hash = UINT32_C(2166136261);
        if (ok) ok = fwrite(&header, sizeof(header), 1, f) == 1;
        for (uint64_t offset = 0; ok && offset < count; offset += chunk) {
            size_t n = count - offset < chunk ? (size_t)(count - offset) : chunk;
            ok = tb_gpu_d2h(ops, host, device + offset, n * sizeof(float)) &&
                 fwrite(host, sizeof(float), n, f) == n;
            if (ok) hash = tb_hash_bytes(hash, (unsigned char *)host, n * sizeof(float));
        }
        if (ok) {
            header.checksum = hash;
            ok = fseek(f, 0, SEEK_SET) == 0 && fwrite(&header, sizeof(header), 1, f) == 1;
        }
    } else {
        TbWeightHeader read_header;
        if (ok) ok = fread(&read_header, sizeof(read_header), 1, f) == 1 &&
                     memcmp(read_header.magic, "TBWGHT1", 7) == 0 &&
                     read_header.version == 1 && read_header.count == count;
        uint32_t hash = UINT32_C(2166136261);
        for (uint64_t offset = 0; ok && offset < count; offset += chunk) {
            size_t n = count - offset < chunk ? (size_t)(count - offset) : chunk;
            ok = fread(host, sizeof(float), n, f) == n;
            if (ok) hash = tb_hash_bytes(hash, (unsigned char *)host, n * sizeof(float));
        }
        if (ok) ok = hash == read_header.checksum && fgetc(f) == EOF &&
                     fseek(f, (long)sizeof(read_header), SEEK_SET) == 0;
        for (uint64_t offset = 0; ok && offset < count; offset += chunk) {
            size_t n = count - offset < chunk ? (size_t)(count - offset) : chunk;
            ok = fread(host, sizeof(float), n, f) == n &&
                 tb_gpu_h2d(ops, device + offset, host, n * sizeof(float));
        }
    }
    free(host);
    if (fclose(f) != 0) ok = 0;
    if (save && !ok) remove(path);
    return ok;
}

static int tb_checkpoint_meta(const char *path, uint64_t parameters,
                              uint64_t *updates, float *log_alpha, int save) {
    TbTrainerHeader header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, "TBTRNR1", 7);
    header.version = 1;
    header.parameters = parameters;
    if (save) {
        header.updates = *updates;
        header.log_alpha = *log_alpha;
        header.checksum = tb_hash_bytes(UINT32_C(2166136261),
            (unsigned char *)&header, offsetof(TbTrainerHeader, checksum));
        FILE *f = fopen(path, "wb");
        if (!f) return 0;
        int ok = fwrite(&header, sizeof(header), 1, f) == 1;
        if (fclose(f) != 0) ok = 0;
        return ok;
    }
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    int ok = fread(&header, sizeof(header), 1, f) == 1 && fgetc(f) == EOF &&
        memcmp(header.magic, "TBTRNR1", 7) == 0 && header.version == 1 &&
        header.parameters == parameters && isfinite(header.log_alpha) &&
        header.checksum == tb_hash_bytes(UINT32_C(2166136261),
            (unsigned char *)&header, offsetof(TbTrainerHeader, checksum));
    fclose(f);
    if (ok) { *updates = header.updates; *log_alpha = header.log_alpha; }
    return ok;
}

static int tb_broadcast_buffer(tb_gpu_trainer *t, float **buffers, uint64_t count) {
    if (t->device_count <= 1) return 1;
    const size_t chunk = 1u << 20;
    float *host = (float *)malloc(chunk * sizeof(float));
    if (!host) return tb_error(t, "checkpoint broadcast host allocation failed");
    int ok = 1;
    for (uint64_t offset = 0; ok && offset < count; offset += chunk) {
        size_t n = count - offset < chunk ? (size_t)(count - offset) : chunk;
        ok = tb_gpu_d2h(t->shards[0].ops, host, buffers[0] + offset, n * sizeof(float));
        for (int d = 1; ok && d < t->device_count; ++d)
            ok = tb_gpu_h2d(t->shards[d].ops, buffers[d] + offset, host, n * sizeof(float));
    }
    free(host);
    return ok ? 1 : tb_error(t, "checkpoint broadcast transfer failed");
}

extern "C" int tb_gpu_trainer_save(tb_gpu_trainer *t, const char *directory) {
    if (!t || !directory || !directory[0]) return 0;
    t->error[0] = 0;
    if (tb_mkdir(directory) != 0 && errno != EEXIST)
        return tb_error(t, "checkpoint directory creation failed");
    if (!tb_synchronize_replicas(t)) return 0;
    char path[1024];
    TbGpuShard *s = &t->shards[0];
    uint64_t total = tb_net_parameter_count(t->layout);
    if (!tb_checkpoint_path(path, sizeof(path), directory, "parameters.bin") ||
        !tb_weight_file(path, s->ops, s->parameters, total, 1))
        return tb_error(t, "model parameter checkpoint write failed");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "targets.bin") ||
        !tb_weight_file(path, s->ops, s->target_parameters, total, 1))
        return tb_error(t, "target parameter checkpoint write failed");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "actor_optimizer.bin") ||
        !tb_gpu_rmsprop_save(s->actor_optimizer, path))
        return tb_error(t, "actor optimizer checkpoint write failed");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "critic_optimizer.bin") ||
        !tb_gpu_rmsprop_save(s->critic_optimizer, path))
        return tb_error(t, "critic optimizer checkpoint write failed");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "trainer.meta") ||
        !tb_checkpoint_meta(path, total, &t->updates, &t->log_alpha, 1))
        return tb_error(t, "trainer metadata checkpoint write failed");
    return 1;
}

extern "C" int tb_gpu_trainer_load(tb_gpu_trainer *t, const char *directory) {
    if (!t || !directory || !directory[0]) return 0;
    t->error[0] = 0;
    char path[1024];
    TbGpuShard *s = &t->shards[0];
    uint64_t total = tb_net_parameter_count(t->layout);
    uint64_t updates = 0;
    float log_alpha = 0;
    if (!tb_checkpoint_path(path, sizeof(path), directory, "trainer.meta") ||
        !tb_checkpoint_meta(path, total, &updates, &log_alpha, 0))
        return tb_error(t, "trainer checkpoint metadata invalid");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "parameters.bin") ||
        !tb_weight_file(path, s->ops, s->parameters, total, 0))
        return tb_error(t, "model parameter checkpoint invalid");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "targets.bin") ||
        !tb_weight_file(path, s->ops, s->target_parameters, total, 0))
        return tb_error(t, "target parameter checkpoint invalid");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "actor_optimizer.bin") ||
        !tb_gpu_rmsprop_load(s->actor_optimizer, path))
        return tb_error(t, "actor optimizer checkpoint invalid");
    if (!tb_checkpoint_path(path, sizeof(path), directory, "critic_optimizer.bin") ||
        !tb_gpu_rmsprop_load(s->critic_optimizer, path))
        return tb_error(t, "critic optimizer checkpoint invalid");
    float *buffers[TB_GPU_TRAINER_MAX_DEVICES];
    for (int d = 0; d < t->device_count; ++d) buffers[d] = t->shards[d].parameters;
    if (!tb_broadcast_buffer(t, buffers, total)) return 0;
    for (int d = 0; d < t->device_count; ++d) buffers[d] = t->shards[d].target_parameters;
    if (!tb_broadcast_buffer(t, buffers, total)) return 0;
    for (int d = 0; d < t->device_count; ++d)
        buffers[d] = tb_gpu_rmsprop_moments(t->shards[d].actor_optimizer);
    if (!tb_broadcast_buffer(t, buffers, tb_net_actor_parameter_count(t->layout))) return 0;
    for (int d = 0; d < t->device_count; ++d)
        buffers[d] = tb_gpu_rmsprop_moments(t->shards[d].critic_optimizer);
    if (!tb_broadcast_buffer(t, buffers, tb_net_critic_parameter_count(t->layout))) return 0;
    for (int d = 0; d < t->device_count; ++d)
        if (!tb_gpu_zero(t->shards[d].ops, t->shards[d].gradients, (size_t)total))
            return tb_error(t, "gradient reset after checkpoint load failed");
    t->updates = updates;
    t->log_alpha = log_alpha;
    return 1;
}
