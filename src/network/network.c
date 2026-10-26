#include "turbinator/network/network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct tb_net_model {
    tb_net_tensor_info tensors[TB_NET_MAX_TENSORS];
    size_t tensor_count;
    uint64_t parameter_count;
    uint64_t actor_parameters;
    int failed;
};

static void add_tensor(tb_net_model *m, const char *name, tb_net_tensor_kind kind,
                       int rank, int n0, int n1, int n2, int n3, int fan_in) {
    if (!m || m->failed || !name || strlen(name) >= sizeof(m->tensors[0].name) ||
        m->tensor_count >= TB_NET_MAX_TENSORS || rank < 1 || rank > 4) {
        if (m) m->failed = 1;
        return;
    }
    int dims[4] = {n0, n1, n2, n3};
    uint64_t count = 1;
    for (int i = 0; i < rank; ++i) {
        if (dims[i] <= 0) { m->failed = 1; return; }
        count *= (uint64_t)dims[i];
    }
    tb_net_tensor_info *t = &m->tensors[m->tensor_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    t->offset = m->parameter_count;
    t->count = count;
    memcpy(t->shape, dims, sizeof(dims));
    t->rank = rank;
    t->fan_in = fan_in;
    t->kind = kind;
    m->parameter_count += count;
}

static int suffix_name(char *dst, size_t cap, const char *prefix, const char *suffix) {
    size_t a = strlen(prefix), b = strlen(suffix);
    if (a + b + 1 > cap) return 0;
    memcpy(dst, prefix, a);
    memcpy(dst + a, suffix, b + 1);
    return 1;
}

static void add_linear(tb_net_model *m, const char *name, int in_dim, int out_dim, int bias) {
    char n[96];
    if (!suffix_name(n, sizeof(n), name, ".weight")) { m->failed = 1; return; }
    add_tensor(m, n, TB_NET_MATRIX, 2, out_dim, in_dim, 0, 0, in_dim);
    if (bias) {
        if (!suffix_name(n, sizeof(n), name, ".bias")) { m->failed = 1; return; }
        add_tensor(m, n, TB_NET_VECTOR, 1, out_dim, 0, 0, 0, in_dim);
    }
}

static void add_norm(tb_net_model *m, const char *name, int features) {
    char n[96];
    if (!suffix_name(n, sizeof(n), name, ".weight")) { m->failed = 1; return; }
    add_tensor(m, n, TB_NET_NORM_GAIN, 1, features, 0, 0, 0, features);
    if (!suffix_name(n, sizeof(n), name, ".bias")) { m->failed = 1; return; }
    add_tensor(m, n, TB_NET_NORM_BIAS, 1, features, 0, 0, 0, features);
}

static void add_conv(tb_net_model *m, const char *name, int in_channels,
                     int out_channels, int kernel_h, int kernel_w) {
    char n[96];
    if (!suffix_name(n, sizeof(n), name, ".weight")) { m->failed = 1; return; }
    add_tensor(m, n, TB_NET_CONV_KERNEL, 4, out_channels, in_channels,
               kernel_h, kernel_w, in_channels * kernel_h * kernel_w);
}

static void add_two_layer_stem(tb_net_model *m, const char *name, int input_dim) {
    char n[96];
    snprintf(n, sizeof(n), "%s.0", name); add_linear(m, n, input_dim, TB_NET_D_MODEL, 1);
    snprintf(n, sizeof(n), "%s.1", name); add_norm(m, n, TB_NET_D_MODEL);
    snprintf(n, sizeof(n), "%s.3", name); add_linear(m, n, TB_NET_D_MODEL, TB_NET_D_MODEL, 1);
    snprintf(n, sizeof(n), "%s.4", name); add_norm(m, n, TB_NET_D_MODEL);
}

static void build_actor(tb_net_model *m) {
    const int d = TB_NET_D_MODEL;
    char n[96];
    add_two_layer_stem(m, "proprio_stem", 52 + 24 + TB_NET_ACTION_DIM);
    add_two_layer_stem(m, "mission_stem", 10 + 24 + TB_NET_ACTION_DIM);

    add_conv(m, "lidar_stem.conv1", 6, 144, 3, 5);
    add_norm(m, "lidar_stem.gn1", 144);
    add_conv(m, "lidar_stem.conv2", 144, 288, 3, 5);
    add_norm(m, "lidar_stem.gn2", 288);
    add_conv(m, "lidar_stem.conv3", 288, 288, 2, 3);
    add_norm(m, "lidar_stem.gn3", 288);
    add_linear(m, "lidar_stem.sector_proj.0", 288 * 2 * 4, d, 1);
    add_norm(m, "lidar_stem.sector_proj.1", d);
    add_tensor(m, "lidar_stem.sector_emb", TB_NET_EMBEDDING, 3, 1, 4, d, 0, d);

    add_conv(m, "camera_stem.conv.0", 1, 96, 5, 5);
    add_norm(m, "camera_stem.conv.1", 96);
    add_conv(m, "camera_stem.conv.3", 96, 192, 3, 3);
    add_norm(m, "camera_stem.conv.4", 192);
    add_conv(m, "camera_stem.conv.6", 192, 192, 3, 3);
    add_norm(m, "camera_stem.conv.7", 192);
    add_linear(m, "camera_stem.proj.0", 192 * 3, d, 1);
    add_norm(m, "camera_stem.proj.1", d);
    add_tensor(m, "camera_stem.cam_id_emb", TB_NET_EMBEDDING, 3, 1, 4, d, 0, d);

    add_tensor(m, "modality_emb", TB_NET_EMBEDDING, 3, 1, TB_NET_SENSOR_TOKENS, d, 0, d);
    for (int i = 0; i < TB_NET_FUSION_LAYERS; ++i) {
        snprintf(n, sizeof(n), "blocks.%d.norm1", i); add_norm(m, n, d);
        snprintf(n, sizeof(n), "blocks.%d.qkv", i); add_linear(m, n, d, 3 * d, 0);
        snprintf(n, sizeof(n), "blocks.%d.out_proj", i); add_linear(m, n, d, d, 0);
        snprintf(n, sizeof(n), "blocks.%d.norm2", i); add_norm(m, n, d);
        snprintf(n, sizeof(n), "blocks.%d.ffn.0", i); add_linear(m, n, d, TB_NET_FFN_DIM, 1);
        snprintf(n, sizeof(n), "blocks.%d.ffn.2", i); add_linear(m, n, TB_NET_FFN_DIM, d, 1);
    }
    for (int i = 0; i < TB_NET_MOE_MODULES; ++i) {
        snprintf(n, sizeof(n), "aero_energy_moe.%d.norm", i); add_norm(m, n, d);
        snprintf(n, sizeof(n), "aero_energy_moe.%d.router.0", i); add_linear(m, n, d, d / 2, 1);
        snprintf(n, sizeof(n), "aero_energy_moe.%d.router.2", i); add_linear(m, n, d / 2, TB_NET_MOE_EXPERTS, 1);
        for (int e = 0; e < TB_NET_MOE_EXPERTS; ++e) {
            snprintf(n, sizeof(n), "aero_energy_moe.%d.experts_gate_up.%d", i, e);
            add_linear(m, n, d, 2 * TB_NET_MOE_EXPERT_DIM, 0);
            snprintf(n, sizeof(n), "aero_energy_moe.%d.experts_down.%d", i, e);
            add_linear(m, n, TB_NET_MOE_EXPERT_DIM, d, 0);
        }
    }
    add_linear(m, "world_model_energy_head.net.0", 2 * d, TB_NET_WORLD_DIM, 1);
    add_norm(m, "world_model_energy_head.net.1", TB_NET_WORLD_DIM);
    add_linear(m, "world_model_energy_head.net.3", TB_NET_WORLD_DIM, TB_NET_WORLD_DIM, 1);
    add_norm(m, "world_model_energy_head.net.4", TB_NET_WORLD_DIM);
    add_linear(m, "world_model_energy_head.net.6", TB_NET_WORLD_DIM, d + 16, 1);
    add_norm(m, "trunk_norm", d);

    add_tensor(m, "action_expert.action_queries", TB_NET_EMBEDDING, 3,
               1, TB_NET_ACTION_HORIZON, d, 0, d);
    add_norm(m, "action_expert.q_norm", d);
    add_norm(m, "action_expert.kv_norm", d);
    add_linear(m, "action_expert.q_proj", d, d, 0);
    add_linear(m, "action_expert.kv_proj", d, 2 * d, 0);
    add_linear(m, "action_expert.out_proj", d, d, 0);
    add_norm(m, "action_expert.ffn_norm", d);
    add_linear(m, "action_expert.ffn.0", d, 2 * d, 1);
    add_linear(m, "action_expert.ffn.2", 2 * d, d, 1);

    add_two_layer_stem(m, "reflex_skip", 2 * d + 24 + TB_NET_ACTION_DIM);
    add_linear(m, "mu_head", d, TB_NET_ACTION_DIM, 1);
    add_linear(m, "log_std_head", d, TB_NET_ACTION_DIM, 1);
    add_tensor(m, "residual_gate_logit", TB_NET_VECTOR, 3,
               1, 1, TB_NET_ACTION_DIM, 0, TB_NET_ACTION_DIM);
    add_linear(m, "aux_perception_head.0", 2 * d, d, 1);
    add_norm(m, "aux_perception_head.1", d);
    add_linear(m, "aux_perception_head.3", d, 9, 1);
    m->actor_parameters = m->parameter_count;
}

static void build_critics(tb_net_model *m) {
    char n[96];
    for (int q = 1; q <= 2; ++q) {
        int in_dim = 2 * TB_NET_D_MODEL + 64 + TB_NET_ACTION_DIM;
        for (int layer = 0; layer < TB_NET_CRITIC_LAYERS; ++layer) {
            snprintf(n, sizeof(n), "critic.q%d_net.%d", q, layer * 3);
            add_linear(m, n, in_dim, TB_NET_CRITIC_HIDDEN, 1);
            snprintf(n, sizeof(n), "critic.q%d_net.%d", q, layer * 3 + 1);
            add_norm(m, n, TB_NET_CRITIC_HIDDEN);
            in_dim = TB_NET_CRITIC_HIDDEN;
        }
        snprintf(n, sizeof(n), "critic.q%d_net.%d", q, TB_NET_CRITIC_LAYERS * 3);
        add_linear(m, n, TB_NET_CRITIC_HIDDEN, TB_NET_ATOMS, 1);
    }
}

tb_net_model *tb_net_create_618m(void) {
    tb_net_model *m = (tb_net_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    build_actor(m);
    build_critics(m);
    if (!tb_net_validate_layout(m)) { free(m); return NULL; }
    return m;
}

void tb_net_destroy(tb_net_model *m) { free(m); }
uint64_t tb_net_parameter_count(const tb_net_model *m) { return m ? m->parameter_count : 0; }
uint64_t tb_net_actor_parameter_count(const tb_net_model *m) { return m ? m->actor_parameters : 0; }
uint64_t tb_net_critic_parameter_count(const tb_net_model *m) {
    return m ? m->parameter_count - m->actor_parameters : 0;
}
size_t tb_net_tensor_count(const tb_net_model *m) { return m ? m->tensor_count : 0; }
const tb_net_tensor_info *tb_net_tensor_at(const tb_net_model *m, size_t index) {
    return m && index < m->tensor_count ? &m->tensors[index] : NULL;
}
const tb_net_tensor_info *tb_net_find_tensor(const tb_net_model *m, const char *name) {
    if (!m || !name) return NULL;
    for (size_t i = 0; i < m->tensor_count; ++i)
        if (strcmp(m->tensors[i].name, name) == 0) return &m->tensors[i];
    return NULL;
}
int tb_net_validate_layout(const tb_net_model *m) {
    if (!m || m->failed || m->actor_parameters != TB_NET_ACTOR_PARAMETERS ||
        m->parameter_count != TB_NET_TOTAL_PARAMETERS ||
        m->tensor_count == 0 || m->tensor_count > TB_NET_MAX_TENSORS) return 0;
    uint64_t expected = 0;
    for (size_t i = 0; i < m->tensor_count; ++i) {
        const tb_net_tensor_info *t = &m->tensors[i];
        if (t->offset != expected || t->count == 0 || t->fan_in <= 0) return 0;
        expected += t->count;
        for (size_t j = 0; j < i; ++j)
            if (!strcmp(t->name, m->tensors[j].name)) return 0;
    }
    return expected == TB_NET_TOTAL_PARAMETERS;
}

static uint64_t splitmix64(uint64_t x) {
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}
static float uniform(uint64_t seed, uint64_t index) {
    return (float)((splitmix64(seed ^ index) >> 40) * (1.0 / 16777216.0));
}

int tb_net_initialize_slice(const tb_net_model *m, uint64_t offset,
                            float *destination, size_t count, uint64_t seed) {
    if (!m || !destination || offset > m->parameter_count ||
        count > m->parameter_count - offset) return 0;
    uint64_t end = offset + count;
    for (size_t ti = 0; ti < m->tensor_count; ++ti) {
        const tb_net_tensor_info *t = &m->tensors[ti];
        uint64_t a = t->offset > offset ? t->offset : offset;
        uint64_t b = t->offset + t->count < end ? t->offset + t->count : end;
        if (a >= b) continue;
        float scale = 0.0f;
        if (t->kind == TB_NET_MATRIX || t->kind == TB_NET_CONV_KERNEL) {
            scale = 1.0f / sqrtf((float)t->fan_in);
            if (!strcmp(t->name, "mu_head.weight")) scale = 1.0e-4f;
            if (!strcmp(t->name, "log_std_head.weight")) scale = 0.01f;
        }
        const int log_std_bias = !strcmp(t->name, "log_std_head.bias");
        const int residual_gate = !strcmp(t->name, "residual_gate_logit");
        for (uint64_t j = a; j < b; ++j) {
            float v = 0.0f;
            if (t->kind == TB_NET_NORM_GAIN) v = 1.0f;
            else if (t->kind == TB_NET_EMBEDDING)
                v = (uniform(seed, j) + uniform(seed + 17, j) +
                     uniform(seed + 29, j) + uniform(seed + 43, j) - 2.0f) * 0.02f;
            else if (t->kind == TB_NET_MATRIX || t->kind == TB_NET_CONV_KERNEL) {
                v = (2.0f * uniform(seed, j) - 1.0f) * scale;
            } else if (log_std_bias) v = -2.2f;
            else if (residual_gate) v = -1.6f;
            destination[j - offset] = v;
        }
    }
    return 1;
}
