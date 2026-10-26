#include "turbinator/training/training.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TB_PI 3.14159265358979323846f
#define TB_NET_LAYERS 3
#define TB_MAX_WIDTH 512
#define TB_CRITIC_INPUT (TB_TRAIN_FEATURE_DIM + TB_TRAIN_PRIVILEGED_DIM + TB_TRAIN_ACTION_DIM)

typedef struct tb_dense {
    int inputs, outputs;
    float *w, *b, *gw, *gb, *mw, *vw, *mb, *vb;
} tb_dense;

typedef struct tb_net {
    tb_dense layer[TB_NET_LAYERS];
    uint64_t steps;
} tb_net;

typedef struct tb_net_cache {
    float activation[TB_NET_LAYERS + 1][TB_MAX_WIDTH];
} tb_net_cache;

typedef struct tb_replay_transition {
    float feature[TB_TRAIN_FEATURE_DIM];
    float next_feature[TB_TRAIN_FEATURE_DIM];
    float privileged[TB_TRAIN_PRIVILEGED_DIM];
    float next_privileged[TB_TRAIN_PRIVILEGED_DIM];
    float action[TB_TRAIN_ACTION_DIM];
    float reference_action[TB_TRAIN_ACTION_DIM];
    float reward;
    uint8_t done;
} tb_replay_transition;

typedef struct tb_actor_state {
    float action[TB_TRAIN_ACTION_DIM];
    float mu[TB_TRAIN_ACTION_DIM];
    float std[TB_TRAIN_ACTION_DIM];
    float eps[TB_TRAIN_ACTION_DIM];
    float log_std[TB_TRAIN_ACTION_DIM];
    uint8_t std_unclipped[TB_TRAIN_ACTION_DIM];
    float log_probability;
} tb_actor_state;

struct tb_sac_agent {
    tb_train_config cfg;
    tb_net actor, actor_target, critic1, critic2, critic1_target, critic2_target;
    tb_replay_transition *replay;
    size_t size, cursor;
    uint64_t rng, updates;
    float log_alpha, alpha_m, alpha_v;
};

static float tb_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static float tb_finite(float x) {
    return isfinite(x) ? x : 0.0f;
}

static float tb_lidar_feature(float range_m) {
    if (!isfinite(range_m) || range_m < 0.0f) return -1.0f;
    return log1pf(tb_clampf(range_m, 0.0f, 120.0f)) / log1pf(120.0f);
}

static float tb_scalar_feature(int index, float value) {
    float scale = 1.0f;
    if (index < 2) scale = 300.0f;
    else if (index == 2) scale = 100.0f;
    else if (index < 6) scale = 50.0f;
    else if (index >= 10 && index < 13) scale = 5.0f;
    else if (index >= 13 && index < 16) scale = 30.0f;
    else if (index >= 28 && index < 32) scale = TB_PI / 4.0f;
    else if (index >= 32 && index < 40) scale = 5.0f;
    else if (index == 44 || index == 45) scale = TB_PI / 2.0f;
    return tb_clampf(tb_finite(value) / scale, -1.0f, 1.0f);
}

static uint64_t tb_rand_u64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static float tb_uniform(uint64_t *state) {
    return (float)((tb_rand_u64(state) >> 40) + 0.5) / 16777216.0f;
}

static float tb_normal(uint64_t *state) {
    const float u1 = fmaxf(tb_uniform(state), 1.0e-7f);
    const float u2 = tb_uniform(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TB_PI * u2);
}

void tb_train_extract_features_cpu(const float obs[TB_TRAIN_OBSERVATION_DIM],
                                   const float mission[TB_TRAIN_MISSION_DIM],
                                   const uint8_t *cameras,
                                   float out[TB_TRAIN_FEATURE_DIM]) {
    if (!obs || !mission || !out) return;
    for (int i = 0; i < 48; ++i) out[i] = tb_scalar_feature(i, obs[i]);
    for (int i = 0; i < 4; ++i) out[48 + i] = tb_clampf(tb_finite(obs[4144 + i]), 0.0f, 1.0f);
    for (int i = 0; i < 10; ++i) {
        const float safe_value = (i >= 7 && tb_finite(mission[6]) > 0.5f) ? 0.0f : mission[i];
        out[52 + i] = tb_clampf(tb_finite(safe_value), -1.0f, 1.0f);
    }
    /* Two LiDAR heads; each head gets a 4-elevation x 8-azimuth grid. */
    for (int head = 0; head < 2; ++head) {
        for (int el_sector = 0; el_sector < 4; ++el_sector) {
            for (int az_sector = 0; az_sector < 8; ++az_sector) {
                float sum = 0.0f;
                for (int el = 0; el < 4; ++el) {
                    for (int az = 0; az < 16; ++az) {
                        const int index = 48 + head * 2048 + (el_sector * 4 + el) * 128 + az_sector * 16 + az;
                        sum += tb_lidar_feature(obs[index]);
                    }
                }
                out[62 + head * 32 + el_sector * 8 + az_sector] = sum / 64.0f;
            }
        }
    }
    /* Quadrant means preserve coarse target location without a learned image stem. */
    for (int cam = 0; cam < 4; ++cam) {
        for (int qy = 0; qy < 2; ++qy) {
            for (int qx = 0; qx < 2; ++qx) {
                unsigned int sum = 0;
                if (cameras) {
                    for (int y = 0; y < 32; ++y)
                        for (int x = 0; x < 32; ++x)
                            sum += cameras[cam * 4096 + (qy * 32 + y) * 64 + qx * 32 + x];
                }
                out[126 + cam * 4 + qy * 2 + qx] = (float)sum / (1024.0f * 255.0f);
            }
        }
    }
}

void tb_train_config_default(tb_train_config *out) {
    if (!out) return;
    out->seed = 42;
    out->replay_capacity = 8192;
    out->hidden_dim = 128;
    out->gamma = 0.992f;
    out->tau = 0.005f;
    out->actor_lr = 1.5e-4f;
    out->critic_lr = 3.0e-4f;
    out->alpha_lr = 5.0e-5f;
    out->initial_alpha = 0.05f;
    out->target_entropy = -0.85f * TB_TRAIN_ACTION_DIM;
    out->reference_weight = 0.5f;
    out->max_grad_norm = 1.5f;
    out->reward_scale = 1.0f;
}

static void tb_dense_free(tb_dense *l) {
    free(l->w); free(l->b); free(l->gw); free(l->gb);
    free(l->mw); free(l->vw); free(l->mb); free(l->vb);
    memset(l, 0, sizeof(*l));
}

static int tb_dense_init(tb_dense *l, int inputs, int outputs, uint64_t *rng) {
    const size_t count = (size_t)inputs * (size_t)outputs;
    l->inputs = inputs;
    l->outputs = outputs;
    l->w = (float *)calloc(count, sizeof(float));
    l->b = (float *)calloc((size_t)outputs, sizeof(float));
    l->gw = (float *)calloc(count, sizeof(float));
    l->gb = (float *)calloc((size_t)outputs, sizeof(float));
    l->mw = (float *)calloc(count, sizeof(float));
    l->vw = (float *)calloc(count, sizeof(float));
    l->mb = (float *)calloc((size_t)outputs, sizeof(float));
    l->vb = (float *)calloc((size_t)outputs, sizeof(float));
    if (!l->w || !l->b || !l->gw || !l->gb || !l->mw || !l->vw || !l->mb || !l->vb) return 0;
    const float scale = sqrtf(2.0f / (float)(inputs + outputs));
    for (size_t i = 0; i < count; ++i) l->w[i] = tb_normal(rng) * scale;
    return 1;
}

static void tb_net_free(tb_net *n) {
    for (int i = 0; i < TB_NET_LAYERS; ++i) tb_dense_free(&n->layer[i]);
    n->steps = 0;
}

static int tb_net_init(tb_net *n, int input, int hidden, int output, uint64_t *rng) {
    if (!tb_dense_init(&n->layer[0], input, hidden, rng) ||
        !tb_dense_init(&n->layer[1], hidden, hidden, rng) ||
        !tb_dense_init(&n->layer[2], hidden, output, rng)) {
        tb_net_free(n);
        return 0;
    }
    return 1;
}

static void tb_net_copy(tb_net *dst, const tb_net *src) {
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        const tb_dense *s = &src->layer[li];
        tb_dense *d = &dst->layer[li];
        memcpy(d->w, s->w, (size_t)s->inputs * (size_t)s->outputs * sizeof(float));
        memcpy(d->b, s->b, (size_t)s->outputs * sizeof(float));
    }
}

static void tb_net_ema(tb_net *target, const tb_net *source, float tau) {
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        const tb_dense *s = &source->layer[li];
        tb_dense *d = &target->layer[li];
        const size_t count = (size_t)s->inputs * (size_t)s->outputs;
        for (size_t k = 0; k < count; ++k) d->w[k] += tau * (s->w[k] - d->w[k]);
        for (int k = 0; k < s->outputs; ++k) d->b[k] += tau * (s->b[k] - d->b[k]);
    }
}

static const float *tb_net_forward(const tb_net *n, const float *input, tb_net_cache *cache) {
    memcpy(cache->activation[0], input, (size_t)n->layer[0].inputs * sizeof(float));
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        const tb_dense *l = &n->layer[li];
        const float *x = cache->activation[li];
        float *y = cache->activation[li + 1];
        for (int o = 0; o < l->outputs; ++o) {
            const float *row = l->w + (size_t)o * (size_t)l->inputs;
            float value = l->b[o];
            for (int i = 0; i < l->inputs; ++i) value += row[i] * x[i];
            y[o] = li == TB_NET_LAYERS - 1 ? value : tanhf(value);
        }
    }
    return cache->activation[TB_NET_LAYERS];
}

static void tb_net_backward(tb_net *n, const tb_net_cache *cache, const float *output_gradient,
                            int accumulate, float *input_gradient) {
    float current[TB_MAX_WIDTH], previous[TB_MAX_WIDTH];
    const int output_width = n->layer[TB_NET_LAYERS - 1].outputs;
    memcpy(current, output_gradient, (size_t)output_width * sizeof(float));
    for (int li = TB_NET_LAYERS - 1; li >= 0; --li) {
        tb_dense *l = &n->layer[li];
        const float *x = cache->activation[li];
        memset(previous, 0, (size_t)l->inputs * sizeof(float));
        for (int o = 0; o < l->outputs; ++o) {
            const float g = tb_clampf(tb_finite(current[o]), -1.0e4f, 1.0e4f);
            const size_t offset = (size_t)o * (size_t)l->inputs;
            if (accumulate) l->gb[o] += g;
            for (int i = 0; i < l->inputs; ++i) {
                if (accumulate) l->gw[offset + (size_t)i] += g * x[i];
                previous[i] += l->w[offset + (size_t)i] * g;
            }
        }
        if (li > 0) {
            const float *a = cache->activation[li];
            for (int i = 0; i < l->inputs; ++i) current[i] = previous[i] * (1.0f - a[i] * a[i]);
        } else if (input_gradient) {
            memcpy(input_gradient, previous, (size_t)l->inputs * sizeof(float));
        }
    }
}

static float tb_net_step(tb_net *n, float lr, float max_grad_norm, size_t batch_size) {
    double square_sum = 0.0;
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        tb_dense *l = &n->layer[li];
        const size_t count = (size_t)l->inputs * (size_t)l->outputs;
        for (size_t k = 0; k < count; ++k) square_sum += (double)l->gw[k] * l->gw[k];
        for (int k = 0; k < l->outputs; ++k) square_sum += (double)l->gb[k] * l->gb[k];
    }
    const float norm = (float)(sqrt(square_sum) / (double)batch_size);
    const float clip = norm > max_grad_norm ? max_grad_norm / (norm + 1.0e-8f) : 1.0f;
    const float scale = clip / (float)batch_size;
    n->steps++;
    const float correction1 = 1.0f - powf(0.9f, (float)n->steps);
    const float correction2 = 1.0f - powf(0.999f, (float)n->steps);
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        tb_dense *l = &n->layer[li];
        const size_t count = (size_t)l->inputs * (size_t)l->outputs;
        for (size_t k = 0; k < count; ++k) {
            const float g = tb_clampf(tb_finite(l->gw[k] * scale), -100.0f, 100.0f);
            l->mw[k] = 0.9f * l->mw[k] + 0.1f * g;
            l->vw[k] = 0.999f * l->vw[k] + 0.001f * g * g;
            l->w[k] -= lr * (l->mw[k] / correction1) / (sqrtf(l->vw[k] / correction2) + 1.0e-8f);
            l->gw[k] = 0.0f;
        }
        for (int k = 0; k < l->outputs; ++k) {
            const float g = tb_clampf(tb_finite(l->gb[k] * scale), -100.0f, 100.0f);
            l->mb[k] = 0.9f * l->mb[k] + 0.1f * g;
            l->vb[k] = 0.999f * l->vb[k] + 0.001f * g * g;
            l->b[k] -= lr * (l->mb[k] / correction1) / (sqrtf(l->vb[k] / correction2) + 1.0e-8f);
            l->gb[k] = 0.0f;
        }
    }
    return norm;
}

static void tb_critic_input(const float *feature, const float *privileged, const float *action,
                            float out[TB_CRITIC_INPUT]) {
    memcpy(out, feature, TB_TRAIN_FEATURE_DIM * sizeof(float));
    for (int i = 0; i < TB_TRAIN_PRIVILEGED_DIM; ++i)
        out[TB_TRAIN_FEATURE_DIM + i] = tb_clampf(tb_finite(privileged[i]), -8.0f, 8.0f) / 8.0f;
    memcpy(out + TB_TRAIN_FEATURE_DIM + TB_TRAIN_PRIVILEGED_DIM,
           action, TB_TRAIN_ACTION_DIM * sizeof(float));
}

static void tb_actor_sample(const tb_net *actor, const float *feature, uint64_t *rng,
                            int deterministic, tb_net_cache *cache, tb_actor_state *state) {
    const float *raw = tb_net_forward(actor, feature, cache);
    state->log_probability = 0.0f;
    for (int i = 0; i < TB_TRAIN_ACTION_DIM; ++i) {
        const float raw_ls = tb_finite(raw[TB_TRAIN_ACTION_DIM + i]);
        const float log_std = tb_clampf(raw_ls, -3.5f, -1.6f);
        const float std = expf(log_std);
        const float eps = deterministic ? 0.0f : tb_normal(rng) * (i == 14 ? 0.15f : 1.0f);
        const float z = tb_clampf(tb_finite(raw[i]) + std * eps, -8.0f, 8.0f);
        const float action = tanhf(z);
        state->mu[i] = tb_finite(raw[i]);
        state->std[i] = std;
        state->eps[i] = eps;
        state->log_std[i] = log_std;
        state->std_unclipped[i] = (uint8_t)(raw_ls > -3.5f && raw_ls < -1.6f);
        state->action[i] = action;
        if (!deterministic) {
            const float gaussian = -0.5f * eps * eps - log_std - 0.5f * logf(2.0f * TB_PI);
            state->log_probability += gaussian - logf(1.0f - action * action + 1.0e-6f);
        }
    }
}

static int tb_config_valid(const tb_train_config *c) {
    return c && c->replay_capacity > 0 && c->replay_capacity <= SIZE_MAX / sizeof(tb_replay_transition) &&
           c->hidden_dim >= 16 && c->hidden_dim <= 256 && c->gamma >= 0.0f && c->gamma <= 1.0f &&
           c->tau > 0.0f && c->tau <= 1.0f && c->actor_lr > 0.0f && c->critic_lr > 0.0f &&
           c->alpha_lr > 0.0f && c->initial_alpha > 0.0f && c->max_grad_norm > 0.0f;
}

tb_sac_agent *tb_sac_create(const tb_train_config *config) {
    tb_train_config defaults;
    if (!config) {
        tb_train_config_default(&defaults);
        config = &defaults;
    }
    if (!tb_config_valid(config)) return NULL;
    tb_sac_agent *a = (tb_sac_agent *)calloc(1, sizeof(*a));
    if (!a) return NULL;
    a->cfg = *config;
    a->rng = config->seed ? config->seed : UINT64_C(0x9e3779b97f4a7c15);
    a->log_alpha = logf(config->initial_alpha);
    a->replay = (tb_replay_transition *)calloc(config->replay_capacity, sizeof(tb_replay_transition));
    if (!a->replay ||
        !tb_net_init(&a->actor, TB_TRAIN_FEATURE_DIM, config->hidden_dim, 2 * TB_TRAIN_ACTION_DIM, &a->rng) ||
        !tb_net_init(&a->actor_target, TB_TRAIN_FEATURE_DIM, config->hidden_dim, 2 * TB_TRAIN_ACTION_DIM, &a->rng) ||
        !tb_net_init(&a->critic1, TB_CRITIC_INPUT, config->hidden_dim, 1, &a->rng) ||
        !tb_net_init(&a->critic2, TB_CRITIC_INPUT, config->hidden_dim, 1, &a->rng) ||
        !tb_net_init(&a->critic1_target, TB_CRITIC_INPUT, config->hidden_dim, 1, &a->rng) ||
        !tb_net_init(&a->critic2_target, TB_CRITIC_INPUT, config->hidden_dim, 1, &a->rng)) {
        tb_sac_destroy(a);
        return NULL;
    }
    for (int i = 0; i < TB_TRAIN_ACTION_DIM; ++i) {
        a->actor.layer[2].b[i] = 0.0f;
        a->actor.layer[2].b[TB_TRAIN_ACTION_DIM + i] = -2.2f;
    }
    tb_net_copy(&a->actor_target, &a->actor);
    tb_net_copy(&a->critic1_target, &a->critic1);
    tb_net_copy(&a->critic2_target, &a->critic2);
    return a;
}

void tb_sac_destroy(tb_sac_agent *a) {
    if (!a) return;
    tb_net_free(&a->actor); tb_net_free(&a->actor_target);
    tb_net_free(&a->critic1); tb_net_free(&a->critic2);
    tb_net_free(&a->critic1_target); tb_net_free(&a->critic2_target);
    free(a->replay);
    free(a);
}

size_t tb_sac_replay_size(const tb_sac_agent *a) { return a ? a->size : 0; }
uint64_t tb_sac_update_count(const tb_sac_agent *a) { return a ? a->updates : 0; }

int tb_sac_act_features(tb_sac_agent *a, const float feature[TB_TRAIN_FEATURE_DIM],
                        int deterministic, float out_action[TB_TRAIN_ACTION_DIM],
                        float *out_log_probability) {
    if (!a || !feature || !out_action) return 0;
    tb_net_cache cache;
    tb_actor_state state;
    tb_actor_sample(&a->actor, feature, &a->rng, deterministic, &cache, &state);
    memcpy(out_action, state.action, sizeof(state.action));
    if (out_log_probability) *out_log_probability = state.log_probability;
    return 1;
}

int tb_sac_act(tb_sac_agent *a, const float obs[TB_TRAIN_OBSERVATION_DIM],
               const float mission[TB_TRAIN_MISSION_DIM], const uint8_t *cameras,
               int deterministic, float out_action[TB_TRAIN_ACTION_DIM], float *out_log_probability) {
    if (!obs || !mission) return 0;
    float feature[TB_TRAIN_FEATURE_DIM];
    tb_train_extract_features_cpu(obs, mission, cameras, feature);
    return tb_sac_act_features(a, feature, deterministic, out_action, out_log_probability);
}

int tb_sac_add_transition_features(tb_sac_agent *a,
                                   const float feature[TB_TRAIN_FEATURE_DIM],
                                   const float privileged[TB_TRAIN_PRIVILEGED_DIM],
                                   const float action[TB_TRAIN_ACTION_DIM],
                                   const float reference_action[TB_TRAIN_ACTION_DIM], float reward,
                                   const float next_feature[TB_TRAIN_FEATURE_DIM],
                                   const float next_privileged[TB_TRAIN_PRIVILEGED_DIM], int done) {
    if (!a || !feature || !privileged || !action || !reference_action ||
        !next_feature || !next_privileged) return 0;
    tb_replay_transition *t = &a->replay[a->cursor];
    memcpy(t->feature, feature, sizeof(t->feature));
    memcpy(t->next_feature, next_feature, sizeof(t->next_feature));
    memcpy(t->privileged, privileged, sizeof(t->privileged));
    memcpy(t->next_privileged, next_privileged, sizeof(t->next_privileged));
    for (int i = 0; i < TB_TRAIN_ACTION_DIM; ++i) {
        t->action[i] = tb_clampf(tb_finite(action[i]), -1.0f, 1.0f);
        t->reference_action[i] = tb_clampf(tb_finite(reference_action[i]), -1.0f, 1.0f);
    }
    t->reward = tb_clampf(tb_finite(reward), -100.0f, 100.0f);
    t->done = done ? 1 : 0;
    a->cursor = (a->cursor + 1) % a->cfg.replay_capacity;
    if (a->size < a->cfg.replay_capacity) ++a->size;
    return 1;
}

int tb_sac_add_transition(tb_sac_agent *a, const float obs[TB_TRAIN_OBSERVATION_DIM],
                          const float mission[TB_TRAIN_MISSION_DIM], const uint8_t *cameras,
                          const float privileged[TB_TRAIN_PRIVILEGED_DIM],
                          const float action[TB_TRAIN_ACTION_DIM],
                          const float reference_action[TB_TRAIN_ACTION_DIM], float reward,
                          const float next_obs[TB_TRAIN_OBSERVATION_DIM],
                          const float next_mission[TB_TRAIN_MISSION_DIM], const uint8_t *next_cameras,
                          const float next_privileged[TB_TRAIN_PRIVILEGED_DIM], int done) {
    if (!a || !obs || !mission || !privileged || !action || !reference_action ||
        !next_obs || !next_mission || !next_privileged) return 0;
    float feature[TB_TRAIN_FEATURE_DIM], next_feature[TB_TRAIN_FEATURE_DIM];
    tb_train_extract_features_cpu(obs, mission, cameras, feature);
    tb_train_extract_features_cpu(next_obs, next_mission, next_cameras, next_feature);
    return tb_sac_add_transition_features(a, feature, privileged, action,
                                          reference_action, reward, next_feature,
                                          next_privileged, done);
}

static void tb_alpha_step(tb_sac_agent *a, float mean_logp) {
    const float g = -(mean_logp + a->cfg.target_entropy);
    const float t = (float)(a->updates + 1);
    a->alpha_m = 0.9f * a->alpha_m + 0.1f * g;
    a->alpha_v = 0.999f * a->alpha_v + 0.001f * g * g;
    a->log_alpha -= a->cfg.alpha_lr * (a->alpha_m / (1.0f - powf(0.9f, t))) /
                    (sqrtf(a->alpha_v / (1.0f - powf(0.999f, t))) + 1.0e-8f);
    a->log_alpha = tb_clampf(a->log_alpha, logf(0.005f), logf(0.2f));
}

int tb_sac_update(tb_sac_agent *a, size_t batch_size, tb_train_metrics *out) {
    if (!a || a->size == 0 || batch_size == 0 || batch_size > 4096) return 0;
    tb_train_metrics metrics = {0};
    const float alpha = expf(a->log_alpha);
    /* Draw one index set for both critic and actor passes. */
    size_t *indices = (size_t *)malloc(batch_size * sizeof(size_t));
    if (!indices) return 0;
    for (size_t b = 0; b < batch_size; ++b) indices[b] = (size_t)(tb_rand_u64(&a->rng) % a->size);

    for (size_t b = 0; b < batch_size; ++b) {
        const tb_replay_transition *t = &a->replay[indices[b]];
        tb_net_cache next_actor_cache, q1_target_cache, q2_target_cache, q1_cache, q2_cache;
        tb_actor_state next_actor;
        float critic_in[TB_CRITIC_INPUT];
        tb_actor_sample(&a->actor_target, t->next_feature, &a->rng, 0, &next_actor_cache, &next_actor);
        tb_critic_input(t->next_feature, t->next_privileged, next_actor.action, critic_in);
        const float q1_target = tb_net_forward(&a->critic1_target, critic_in, &q1_target_cache)[0];
        const float q2_target = tb_net_forward(&a->critic2_target, critic_in, &q2_target_cache)[0];
        const float entropy_bonus = tb_clampf(alpha * tb_clampf(next_actor.log_probability, -100.0f, 100.0f),
                                              -0.5f, 0.5f);
        const float boot = fminf(tb_finite(q1_target), tb_finite(q2_target)) - entropy_bonus;
        const float target = tb_clampf(t->reward * a->cfg.reward_scale +
                                       (t->done ? 0.0f : a->cfg.gamma * boot), -100.0f, 100.0f);
        tb_critic_input(t->feature, t->privileged, t->action, critic_in);
        const float q1 = tb_net_forward(&a->critic1, critic_in, &q1_cache)[0];
        const float q2 = tb_net_forward(&a->critic2, critic_in, &q2_cache)[0];
        const float e1 = tb_clampf(tb_finite(q1) - target, -100.0f, 100.0f);
        const float e2 = tb_clampf(tb_finite(q2) - target, -100.0f, 100.0f);
        tb_net_backward(&a->critic1, &q1_cache, &e1, 1, NULL);
        tb_net_backward(&a->critic2, &q2_cache, &e2, 1, NULL);
        metrics.critic_loss += 0.5f * (e1 * e1 + e2 * e2);
    }
    const float q1_norm = tb_net_step(&a->critic1, a->cfg.critic_lr, a->cfg.max_grad_norm, batch_size);
    const float q2_norm = tb_net_step(&a->critic2, a->cfg.critic_lr, a->cfg.max_grad_norm, batch_size);
    metrics.critic_grad_norm = 0.5f * (q1_norm + q2_norm);

    for (size_t b = 0; b < batch_size; ++b) {
        const tb_replay_transition *t = &a->replay[indices[b]];
        tb_net_cache actor_cache, q1_cache, q2_cache, qref_cache;
        tb_actor_state actor;
        float critic_in[TB_CRITIC_INPUT], critic_input_grad[TB_CRITIC_INPUT];
        float output_grad[2 * TB_TRAIN_ACTION_DIM] = {0};
        const float q_seed = 1.0f;
        tb_actor_sample(&a->actor, t->feature, &a->rng, 0, &actor_cache, &actor);
        tb_critic_input(t->feature, t->privileged, actor.action, critic_in);
        const float q1 = tb_net_forward(&a->critic1, critic_in, &q1_cache)[0];
        const float q2 = tb_net_forward(&a->critic2, critic_in, &q2_cache)[0];
        const float q = fminf(tb_finite(q1), tb_finite(q2));
        tb_net *selected = q1 <= q2 ? &a->critic1 : &a->critic2;
        const tb_net_cache *selected_cache = q1 <= q2 ? &q1_cache : &q2_cache;
        tb_net_backward(selected, selected_cache, &q_seed, 0, critic_input_grad);
        tb_critic_input(t->feature, t->privileged, t->reference_action, critic_in);
        const float q_ref1 = tb_net_forward(&a->critic1, critic_in, &qref_cache)[0];
        const float advantage = tb_clampf(q_ref1 - q, -5.0f, 5.0f);
        const float gate = 0.5f + 0.5f / (1.0f + expf(-1.5f * advantage));
        float bc = 0.0f;
        for (int i = 0; i < TB_TRAIN_ACTION_DIM; ++i) {
            const float action = actor.action[i];
            const float da_dz = 1.0f - action * action;
            const float dqda = critic_input_grad[TB_TRAIN_FEATURE_DIM + TB_TRAIN_PRIVILEGED_DIM + i];
            const float det_action = tanhf(actor.mu[i]);
            const float ref_error = det_action - t->reference_action[i];
            bc += ref_error * ref_error / TB_TRAIN_ACTION_DIM;
            const float d_bc = a->cfg.reference_weight * gate *
                               2.0f * ref_error * (1.0f - det_action * det_action) / TB_TRAIN_ACTION_DIM;
            output_grad[i] = alpha * 2.0f * action - dqda * da_dz + d_bc +
                             0.001f * actor.mu[i];
            const float d_ls = alpha * (-1.0f + 2.0f * action * actor.std[i] * actor.eps[i]) -
                               dqda * da_dz * actor.std[i] * actor.eps[i];
            output_grad[TB_TRAIN_ACTION_DIM + i] = actor.std_unclipped[i] ? d_ls : 0.0f;
        }
        tb_net_backward(&a->actor, &actor_cache, output_grad, 1, NULL);
        metrics.actor_loss += alpha * actor.log_probability - q + a->cfg.reference_weight * gate * bc;
        metrics.reference_loss += bc;
        metrics.q_mean += q;
        metrics.entropy -= actor.log_probability;
    }
    metrics.actor_grad_norm = tb_net_step(&a->actor, a->cfg.actor_lr, a->cfg.max_grad_norm, batch_size);
    metrics.critic_loss /= (float)batch_size;
    metrics.actor_loss /= (float)batch_size;
    metrics.reference_loss /= (float)batch_size;
    metrics.q_mean /= (float)batch_size;
    metrics.entropy /= (float)batch_size;
    tb_alpha_step(a, -metrics.entropy);
    tb_net_ema(&a->actor_target, &a->actor, a->cfg.tau);
    tb_net_ema(&a->critic1_target, &a->critic1, a->cfg.tau);
    tb_net_ema(&a->critic2_target, &a->critic2, a->cfg.tau);
    a->updates++;
    metrics.alpha = expf(a->log_alpha);
    metrics.update_count = a->updates;
    free(indices);
    if (out) *out = metrics;
    return 1;
}

void tb_train_action_to_physical(const float normalized[TB_TRAIN_ACTION_DIM], int bwb_mode,
                                 float physical[TB_TRAIN_ACTION_DIM]) {
    if (!normalized || !physical) return;
    for (int i = 0; i < 4; ++i) physical[i] = 0.5f * (tb_clampf(normalized[i], -1.0f, 1.0f) + 1.0f);
    for (int i = 4; i < 8; ++i) physical[i] = tb_clampf(normalized[i], -1.0f, 1.0f) * 2.0f;
    for (int i = 8; i < 12; ++i) physical[i] = tb_clampf(normalized[i], -1.0f, 1.0f) * (TB_PI / 4.0f);
    physical[12] = (bwb_mode ? 0.0f : TB_PI / 4.0f) + tb_clampf(normalized[12], -1.0f, 1.0f) * (TB_PI / 6.0f);
    physical[13] = tb_clampf(normalized[13], -1.0f, 1.0f) * (TB_PI / 6.0f);
    physical[14] = normalized[14] > 0.0f ? 1.0f : 0.0f;
}

void tb_train_action_from_physical(const float physical[TB_TRAIN_ACTION_DIM], int bwb_mode,
                                   float normalized[TB_TRAIN_ACTION_DIM]) {
    if (!physical || !normalized) return;
    for (int i = 0; i < 4; ++i) normalized[i] = tb_clampf(physical[i] * 2.0f - 1.0f, -1.0f, 1.0f);
    for (int i = 4; i < 8; ++i) normalized[i] = tb_clampf(physical[i] / 2.0f, -1.0f, 1.0f);
    for (int i = 8; i < 12; ++i) normalized[i] = tb_clampf(physical[i] / (TB_PI / 4.0f), -1.0f, 1.0f);
    normalized[12] = tb_clampf((physical[12] - (bwb_mode ? 0.0f : TB_PI / 4.0f)) / (TB_PI / 6.0f), -1.0f, 1.0f);
    normalized[13] = tb_clampf(physical[13] / (TB_PI / 6.0f), -1.0f, 1.0f);
    normalized[14] = physical[14] > 0.5f ? 0.8f : -0.8f;
}

static int tb_write(FILE *file, const void *value, size_t item_size, size_t count) {
    return fwrite(value, item_size, count, file) == count;
}

static int tb_read(FILE *file, void *value, size_t item_size, size_t count) {
    return fread(value, item_size, count, file) == count;
}

static int tb_net_write(FILE *file, const tb_net *net) {
    if (!tb_write(file, &net->steps, sizeof(net->steps), 1)) return 0;
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        const tb_dense *l = &net->layer[li];
        const size_t count = (size_t)l->inputs * (size_t)l->outputs;
        if (!tb_write(file, l->w, sizeof(float), count) || !tb_write(file, l->b, sizeof(float), l->outputs) ||
            !tb_write(file, l->mw, sizeof(float), count) || !tb_write(file, l->vw, sizeof(float), count) ||
            !tb_write(file, l->mb, sizeof(float), l->outputs) || !tb_write(file, l->vb, sizeof(float), l->outputs)) return 0;
    }
    return 1;
}

static int tb_net_read(FILE *file, tb_net *net) {
    if (!tb_read(file, &net->steps, sizeof(net->steps), 1)) return 0;
    for (int li = 0; li < TB_NET_LAYERS; ++li) {
        tb_dense *l = &net->layer[li];
        const size_t count = (size_t)l->inputs * (size_t)l->outputs;
        if (!tb_read(file, l->w, sizeof(float), count) || !tb_read(file, l->b, sizeof(float), l->outputs) ||
            !tb_read(file, l->mw, sizeof(float), count) || !tb_read(file, l->vw, sizeof(float), count) ||
            !tb_read(file, l->mb, sizeof(float), l->outputs) || !tb_read(file, l->vb, sizeof(float), l->outputs)) return 0;
    }
    return 1;
}

int tb_sac_save(const tb_sac_agent *a, const char *path) {
    if (!a || !path) return 0;
    FILE *file = fopen(path, "wb");
    if (!file) return 0;
    const char magic[8] = {'T','B','S','A','C','0','1','\0'};
    int ok = tb_write(file, magic, 1, sizeof(magic)) &&
             tb_write(file, &a->cfg, sizeof(a->cfg), 1) &&
             tb_write(file, &a->rng, sizeof(a->rng), 1) &&
             tb_write(file, &a->updates, sizeof(a->updates), 1) &&
             tb_write(file, &a->log_alpha, sizeof(float), 1) &&
             tb_write(file, &a->alpha_m, sizeof(float), 1) &&
             tb_write(file, &a->alpha_v, sizeof(float), 1) &&
             tb_net_write(file, &a->actor) && tb_net_write(file, &a->actor_target) &&
             tb_net_write(file, &a->critic1) && tb_net_write(file, &a->critic2) &&
             tb_net_write(file, &a->critic1_target) && tb_net_write(file, &a->critic2_target);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

tb_sac_agent *tb_sac_load(const char *path) {
    if (!path) return NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    char magic[8];
    tb_train_config cfg;
    const char expected[8] = {'T','B','S','A','C','0','1','\0'};
    if (!tb_read(file, magic, 1, sizeof(magic)) || memcmp(magic, expected, sizeof(magic)) != 0 ||
        !tb_read(file, &cfg, sizeof(cfg), 1)) {
        fclose(file);
        return NULL;
    }
    tb_sac_agent *a = tb_sac_create(&cfg);
    if (!a) { fclose(file); return NULL; }
    const int ok = tb_read(file, &a->rng, sizeof(a->rng), 1) &&
                   tb_read(file, &a->updates, sizeof(a->updates), 1) &&
                   tb_read(file, &a->log_alpha, sizeof(float), 1) &&
                   tb_read(file, &a->alpha_m, sizeof(float), 1) &&
                   tb_read(file, &a->alpha_v, sizeof(float), 1) &&
                   tb_net_read(file, &a->actor) && tb_net_read(file, &a->actor_target) &&
                   tb_net_read(file, &a->critic1) && tb_net_read(file, &a->critic2) &&
                   tb_net_read(file, &a->critic1_target) && tb_net_read(file, &a->critic2_target);
    fclose(file);
    if (!ok) { tb_sac_destroy(a); return NULL; }
    return a;
}
