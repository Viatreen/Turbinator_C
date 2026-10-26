#include "turbinator/network/network_gpu.h"
#include "onboard_features.cuh"

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TB_CRITIC_INPUT (TB_NET_D_MODEL * 2 + 64 + TB_NET_ACTION_DIM)
#define TB_TRACE_ALLOCS 4096
#define TB_ACTOR_NODES 1024

typedef struct {
    float *allocs[TB_TRACE_ALLOCS];
    int allocation_count;
} TbGpuTrace;

typedef struct {
    float *input;
    float *linear[TB_NET_CRITIC_LAYERS];
    float *norm[TB_NET_CRITIC_LAYERS];
    float *activation[TB_NET_CRITIC_LAYERS];
    float *mean[TB_NET_CRITIC_LAYERS];
    float *inv_std[TB_NET_CRITIC_LAYERS];
    float *logits;
} TbCriticBranch;

typedef enum {
    TB_NODE_INPUT, TB_NODE_LINEAR, TB_NODE_NORM, TB_NODE_SILU,
    TB_NODE_ADD, TB_NODE_CONCAT, TB_NODE_EMBED,
    TB_NODE_SELF_ATTN, TB_NODE_CROSS_ATTN, TB_NODE_SWIGLU,
    TB_NODE_MOE_MIX, TB_NODE_POOL, TB_NODE_PICK,
    TB_NODE_ACTION, TB_NODE_CLAMP, TB_NODE_PACK, TB_NODE_SOFTMAX,
    TB_NODE_BROADCAST_ADD, TB_NODE_QUERY, TB_NODE_CIRC_PAD,
    TB_NODE_CONV, TB_NODE_GN, TB_NODE_CAM_SUMMARY, TB_NODE_LIDAR_SECTOR
} TbNodeOp;

typedef struct {
    TbNodeOp op;
    int a, b;
    int rows, cols;
    int aux[8];
    int pad_h, pad_w;
    float *v, *g;
    float *mean, *inv_std;
    const tb_net_tensor_info *weight, *bias;
} TbActorNode;

struct tb_net_gpu {
    const tb_net_model *layout;
    tb_gpu_ops *ops;
    float *parameters;
    float *gradients;
    int max_batch;
    int actor_batch;
    int critic_batch;
    TbGpuTrace actor_trace;
    TbGpuTrace critic_trace;
    TbCriticBranch q[2];
    TbActorNode actor_nodes[TB_ACTOR_NODES];
    int actor_node_count;
    int actor_action_node, actor_logstd_node, actor_summary_node;
    int actor_aux_node, actor_world_node;
    const float *actor_mission;
    char error[256];
};

static void trace_clear(tb_net_gpu *ctx, TbGpuTrace *trace) {
    if (!ctx || !trace) return;
    for (int i = trace->allocation_count - 1; i >= 0; --i)
        tb_gpu_free(ctx->ops, trace->allocs[i]);
    trace->allocation_count = 0;
}

static float *trace_alloc(tb_net_gpu *ctx, TbGpuTrace *trace, size_t floats) {
    if (!ctx || !trace || !floats || trace->allocation_count >= TB_TRACE_ALLOCS) return NULL;
    float *p = (float *)tb_gpu_malloc(ctx->ops, floats * sizeof(float));
    if (p) trace->allocs[trace->allocation_count++] = p;
    return p;
}

static int net_error(tb_net_gpu *ctx, const char *message) {
    if (ctx) snprintf(ctx->error, sizeof(ctx->error), "%s%s%s", message,
                      tb_gpu_last_error(ctx->ops)[0] ? ": " : "",
                      tb_gpu_last_error(ctx->ops));
    return 0;
}

static float *parameter(tb_net_gpu *ctx, const char *name) {
    const tb_net_tensor_info *t = tb_net_find_tensor(ctx->layout, name);
    if (!t) { net_error(ctx, "missing parameter tensor"); return NULL; }
    return ctx->parameters + t->offset;
}

static float *gradient(tb_net_gpu *ctx, const char *name) {
    const tb_net_tensor_info *t = tb_net_find_tensor(ctx->layout, name);
    if (!t || !ctx->gradients) return NULL;
    return ctx->gradients + t->offset;
}

static int named_parameter(tb_net_gpu *ctx, char *name, size_t cap,
                           int q_index, int layer, const char *suffix,
                           float **param, float **grad) {
    if (snprintf(name, cap, "critic.q%d_net.%d.%s", q_index + 1, layer, suffix) >= (int)cap)
        return net_error(ctx, "critic tensor name too long");
    *param = parameter(ctx, name);
    if (grad) *grad = gradient(ctx, name);
    return *param != NULL;
}

__global__ static void concat_rows_forward(const float *a, const float *b,
                                            float *out, int rows, int ca, int cb) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * (ca + cb)) return;
    int r = (int)(i / (ca + cb)), c = (int)(i % (ca + cb));
    out[i] = c < ca ? a[(size_t)r * ca + c] : b[(size_t)r * cb + c - ca];
}
__global__ static void concat_rows_backward(const float *dy, float *da, float *db,
                                             int rows, int ca, int cb) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * (ca + cb)) return;
    int r = (int)(i / (ca + cb)), c = (int)(i % (ca + cb));
    if (c < ca) atomicAdd(da + (size_t)r * ca + c, dy[i]);
    else atomicAdd(db + (size_t)r * cb + c - ca, dy[i]);
}
__global__ static void clamp_forward(const float *x, float *y, size_t count,
                                     float low, float high) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) y[i] = fminf(high, fmaxf(low, x[i]));
}
__global__ static void clamp_backward(const float *x, const float *dy, float *dx,
                                      size_t count, float low, float high) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count && x[i] > low && x[i] < high) atomicAdd(dx + i, dy[i]);
}
__global__ static void embed_forward(const float *x, const float *embed,
                                     float *y, int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim) y[i] = x[i] + embed[i % ((size_t)tokens * dim)];
}
__global__ static void embed_backward(const float *dy, float *dx, float *dembed,
                                      int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim) {
        atomicAdd(dx + i, dy[i]);
        if (dembed) atomicAdd(dembed + i % ((size_t)tokens * dim), dy[i]);
    }
}
__global__ static void pool_forward(const float *x, float *y,
                                    int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * dim) return;
    int b = (int)(i / dim), c = (int)(i % dim);
    float sum = 0.0f;
    for (int t = 0; t < tokens; ++t) sum += x[((size_t)b * tokens + t) * dim + c];
    y[i] = sum / (float)tokens;
}
__global__ static void pool_backward(const float *dy, float *dx,
                                     int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * tokens * dim) return;
    int b = (int)(i / ((size_t)tokens * dim)), c = (int)(i % dim);
    atomicAdd(dx + i, dy[(size_t)b * dim + c] / (float)tokens);
}
__global__ static void pick_forward(const float *x, float *y,
                                    int batch, int tokens, int dim, int which) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * dim) return;
    int b = (int)(i / dim), c = (int)(i % dim);
    y[i] = x[((size_t)b * tokens + which) * dim + c];
}
__global__ static void pick_backward(const float *dy, float *dx,
                                     int batch, int tokens, int dim, int which) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * dim) return;
    int b = (int)(i / dim), c = (int)(i % dim);
    atomicAdd(dx + ((size_t)b * tokens + which) * dim + c, dy[i]);
}

__global__ static void attention_scores(const float *q, const float *kv,
                                         float *prob, int batch, int q_tokens,
                                         int kv_tokens, int self_qkv) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * TB_NET_HEADS * q_tokens;
    if (row >= total) return;
    int query = row % q_tokens;
    int head = (row / q_tokens) % TB_NET_HEADS;
    int b = row / (q_tokens * TB_NET_HEADS);
    const float *q_vec = q + ((size_t)b * q_tokens + query) *
        (self_qkv ? 3 * TB_NET_D_MODEL : TB_NET_D_MODEL) + head * TB_NET_HEAD_DIM;
    float max_score = -1.0e30f;
    for (int k = 0; k < kv_tokens; ++k) {
        const float *key = kv + ((size_t)b * kv_tokens + k) *
            (self_qkv ? 3 * TB_NET_D_MODEL : 2 * TB_NET_D_MODEL) +
            TB_NET_D_MODEL + head * TB_NET_HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < TB_NET_HEAD_DIM; ++d) dot += q_vec[d] * key[d];
        float score = dot * 0.125f;
        prob[(size_t)row * kv_tokens + k] = score;
        max_score = fmaxf(max_score, score);
    }
    float sum = 0.0f;
    for (int k = 0; k < kv_tokens; ++k) {
        float value = expf(prob[(size_t)row * kv_tokens + k] - max_score);
        prob[(size_t)row * kv_tokens + k] = value;
        sum += value;
    }
    for (int k = 0; k < kv_tokens; ++k)
        prob[(size_t)row * kv_tokens + k] /= fmaxf(sum, 1.0e-20f);
}

__global__ static void attention_values(const float *kv, const float *prob,
                                         float *out, int batch, int q_tokens,
                                         int kv_tokens, int self_qkv) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * q_tokens * TB_NET_D_MODEL) return;
    int d = (int)(i % TB_NET_D_MODEL);
    int q = (int)((i / TB_NET_D_MODEL) % q_tokens);
    int b = (int)(i / ((size_t)q_tokens * TB_NET_D_MODEL));
    int head = d / TB_NET_HEAD_DIM;
    int row = (b * TB_NET_HEADS + head) * q_tokens + q;
    float sum = 0.0f;
    for (int k = 0; k < kv_tokens; ++k) {
        const float *value = kv + ((size_t)b * kv_tokens + k) *
            (self_qkv ? 3 * TB_NET_D_MODEL : 2 * TB_NET_D_MODEL) +
            (self_qkv ? 2 : 1) * TB_NET_D_MODEL + d;
        sum += prob[(size_t)row * kv_tokens + k] * *value;
    }
    out[i] = sum;
}

__global__ static void attention_backward(const float *q, const float *kv,
                                           const float *prob, const float *dout,
                                           float *dq, float *dkv,
                                           int batch, int q_tokens,
                                           int kv_tokens, int self_qkv) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * TB_NET_HEADS * q_tokens;
    if (row >= total) return;
    int query = row % q_tokens;
    int head = (row / q_tokens) % TB_NET_HEADS;
    int b = row / (q_tokens * TB_NET_HEADS);
    int qstride = self_qkv ? 3 * TB_NET_D_MODEL : TB_NET_D_MODEL;
    int kvstride = self_qkv ? 3 * TB_NET_D_MODEL : 2 * TB_NET_D_MODEL;
    const float *q_vec = q + ((size_t)b * q_tokens + query) * qstride + head * TB_NET_HEAD_DIM;
    const float *do_vec = dout + ((size_t)b * q_tokens + query) * TB_NET_D_MODEL + head * TB_NET_HEAD_DIM;
    float dprob[TB_NET_SENSOR_TOKENS];
    float weighted = 0.0f;
    for (int k = 0; k < kv_tokens; ++k) {
        const float *v = kv + ((size_t)b * kv_tokens + k) * kvstride +
            (self_qkv ? 2 : 1) * TB_NET_D_MODEL + head * TB_NET_HEAD_DIM;
        float dot = 0.0f;
        for (int d = 0; d < TB_NET_HEAD_DIM; ++d) dot += do_vec[d] * v[d];
        dprob[k] = dot;
        weighted += prob[(size_t)row * kv_tokens + k] * dot;
    }
    for (int k = 0; k < kv_tokens; ++k) {
        float p = prob[(size_t)row * kv_tokens + k];
        float ds = p * (dprob[k] - weighted) * 0.125f;
        size_t key_base = ((size_t)b * kv_tokens + k) * kvstride + TB_NET_D_MODEL + head * TB_NET_HEAD_DIM;
        size_t value_base = ((size_t)b * kv_tokens + k) * kvstride +
                            (self_qkv ? 2 : 1) * TB_NET_D_MODEL + head * TB_NET_HEAD_DIM;
        for (int d = 0; d < TB_NET_HEAD_DIM; ++d) {
            atomicAdd(dq + ((size_t)b * q_tokens + query) * qstride +
                head * TB_NET_HEAD_DIM + d, ds * kv[key_base + d]);
            atomicAdd(dkv + key_base + d, ds * q_vec[d]);
            atomicAdd(dkv + value_base + d, p * do_vec[d]);
        }
    }
}

/* Sensor preparation keeps the network input contract on the device. */
__global__ static void scalar_inputs(const float *obs, const float *mission,
                                     float *proprio, float *mission_vec,
                                     float *tactical, float *prior, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float *o = obs + (size_t)b * 4148;
    const float *m = mission + (size_t)b * 10;
    float *p = proprio + (size_t)b * 91;
    float *mv = mission_vec + (size_t)b * 49;
    float *t = tactical + (size_t)b * 24;
    float *np = prior + (size_t)b * TB_NET_ACTION_DIM;
    for (int i = 0; i < 52; ++i) {
        float raw = i < 48 ? o[i] : o[4144 + i - 48];
        raw = isfinite(raw) ? fmaxf(-1000.0f, fminf(1000.0f, raw)) : 0.0f;
        float scale = 1.0f;
        if (i < 3) scale = 0.05f;
        else if (i < 6) scale = 0.10f;
        else if (i >= 10 && i < 13) scale = 0.20f;
        else if (i >= 13 && i < 16) scale = 0.10f;
        else if (i >= 28 && i < 32) scale = 0.50f;
        else if (i >= 32 && i < 36) scale = 0.20f;
        float symlog = copysignf(log1pf(fabsf(raw * scale)), raw);
        p[i] = fmaxf(-8.0f, fminf(8.0f, symlog / sqrtf(1.0001f)));
    }
    tb_onboard_tactical_prior(o, m, t, np);
    for (int i = 0; i < 24; ++i) p[52 + i] = t[i];
    for (int i = 0; i < TB_NET_ACTION_DIM; ++i) p[76 + i] = np[i];
    for (int i = 0; i < 10; ++i) mv[i] = isfinite(m[i]) ? m[i] : 0.0f;
    for (int i = 0; i < 24; ++i) mv[10 + i] = t[i];
    for (int i = 0; i < TB_NET_ACTION_DIM; ++i) mv[34 + i] = np[i];
}
__global__ static void pack_tokens_forward(const float *p, const float *m,
                                           const float *l, const float *c,
                                           float *out, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * 10 * TB_NET_D_MODEL) return;
    int b = (int)(i / (10 * TB_NET_D_MODEL));
    int token = (int)((i / TB_NET_D_MODEL) % 10);
    int d = (int)(i % TB_NET_D_MODEL);
    if (token == 0) out[i] = p[(size_t)b * TB_NET_D_MODEL + d];
    else if (token == 1) out[i] = m[(size_t)b * TB_NET_D_MODEL + d];
    else if (token < 6) out[i] = l[((size_t)b * 4 + token - 2) * TB_NET_D_MODEL + d];
    else out[i] = c[((size_t)b * 4 + token - 6) * TB_NET_D_MODEL + d];
}
__global__ static void pack_tokens_backward(const float *dy, float *dp, float *dm,
                                            float *dl, float *dc, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * 10 * TB_NET_D_MODEL) return;
    int b = (int)(i / (10 * TB_NET_D_MODEL));
    int token = (int)((i / TB_NET_D_MODEL) % 10);
    int d = (int)(i % TB_NET_D_MODEL);
    if (token == 0) dp[(size_t)b * TB_NET_D_MODEL + d] += dy[i];
    else if (token == 1) dm[(size_t)b * TB_NET_D_MODEL + d] += dy[i];
    else if (token < 6) dl[((size_t)b * 4 + token - 2) * TB_NET_D_MODEL + d] += dy[i];
    else dc[((size_t)b * 4 + token - 6) * TB_NET_D_MODEL + d] += dy[i];
}
__global__ static void query_forward(const float *queries, float *out,
                                     int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim) out[i] = queries[i % ((size_t)tokens * dim)];
}
__global__ static void query_backward(const float *dy, float *dq,
                                      int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim) atomicAdd(dq + i % ((size_t)tokens * dim), dy[i]);
}
__global__ static void broadcast_add_forward(const float *x, const float *v,
                                             float *out, int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim)
        out[i] = x[i] + v[(i / ((size_t)tokens * dim)) * dim + i % dim];
}
__global__ static void broadcast_add_backward(const float *dy, float *dx, float *dv,
                                              int batch, int tokens, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)batch * tokens * dim) {
        dx[i] += dy[i];
        atomicAdd(dv + (i / ((size_t)tokens * dim)) * dim + i % dim, dy[i]);
    }
}
__global__ static void swiglu_forward(const float *x, float *y, int rows, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    int row = (int)(i / dim), col = (int)(i % dim);
    float gate = x[(size_t)row * 2 * dim + col];
    float up = x[(size_t)row * 2 * dim + dim + col];
    y[i] = gate * (1.0f / (1.0f + expf(-gate))) * up;
}
__global__ static void swiglu_backward(const float *x, const float *dy,
                                       float *dx, int rows, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    int row = (int)(i / dim), col = (int)(i % dim);
    float gate = x[(size_t)row * 2 * dim + col];
    float up = x[(size_t)row * 2 * dim + dim + col];
    float sig = 1.0f / (1.0f + expf(-gate));
    dx[(size_t)row * 2 * dim + col] += dy[i] * up * (sig + gate * sig * (1.0f - sig));
    dx[(size_t)row * 2 * dim + dim + col] += dy[i] * gate * sig;
}
__global__ static void moe_mix_forward(const float *prob,
                                       const float *e0, const float *e1, const float *e2,
                                       const float *e3, const float *e4, const float *e5,
                                       float *y, int rows, int dim) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * dim) return;
    int row = (int)(i / dim);
    const float *p = prob + (size_t)row * 6;
    int best = 0, second = 1;
    if (p[1] > p[0]) { best = 1; second = 0; }
    for (int e = 2; e < 6; ++e) {
        if (p[e] > p[best]) { second = best; best = e; }
        else if (p[e] > p[second]) second = e;
    }
    const float *experts[6] = {e0,e1,e2,e3,e4,e5};
    float denom = fmaxf(p[best] + p[second], 1.0e-6f);
    y[i] = (p[best] * experts[best][i] + p[second] * experts[second][i]) / denom;
}
__global__ static void moe_mix_backward(const float *prob,
                                        const float *e0, const float *e1, const float *e2,
                                        const float *e3, const float *e4, const float *e5,
                                        const float *dy, float *dp,
                                        float *g0, float *g1, float *g2,
                                        float *g3, float *g4, float *g5,
                                        int rows, int dim) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float *p = prob + (size_t)row * 6;
    int best = 0, second = 1;
    if (p[1] > p[0]) { best = 1; second = 0; }
    for (int e = 2; e < 6; ++e) {
        if (p[e] > p[best]) { second = best; best = e; }
        else if (p[e] > p[second]) second = e;
    }
    const float *experts[6] = {e0,e1,e2,e3,e4,e5};
    float *grads[6] = {g0,g1,g2,g3,g4,g5};
    float denom = fmaxf(p[best] + p[second], 1.0e-6f);
    float dot_best = 0.0f, dot_second = 0.0f;
    for (int d = 0; d < dim; ++d) {
        size_t i = (size_t)row * dim + d;
        float v = dy[i];
        grads[best][i] += v * p[best] / denom;
        grads[second][i] += v * p[second] / denom;
        dot_best += v * experts[best][i];
        dot_second += v * experts[second][i];
    }
    float weighted = (p[best] * dot_best + p[second] * dot_second) / denom;
    dp[(size_t)row * 6 + best] += (dot_best - weighted) / denom;
    dp[(size_t)row * 6 + second] += (dot_second - weighted) / denom;
}
__global__ static void action_forward(const float *raw, const float *prior,
                                      const float *mission, const float *gate,
                                      float *action, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * TB_NET_ACTION_DIM) return;
    int b = (int)(i / TB_NET_ACTION_DIM), d = (int)(i % TB_NET_ACTION_DIM);
    const float *m = mission + (size_t)b * 10;
    int is_inv = m[3] > 0.75f && m[0] < -0.25f;
    int is_cargo = m[2] > 0.75f;
    int is_ar = m[6] > 0.75f;
    int is_lateral = (m[4] > 0.75f && m[5] > 0.25f) || m[5] > 0.75f;
    int is_bwb = (is_ar && (m[4] < -0.25f || m[5] > 0.60f)) ||
                 (m[4] > 0.75f && m[2] > 0.25f);
    float scale = is_ar || is_lateral ? (d < 12 ? 0.012f : 0.18f) : 0.035f;
    if (is_bwb) scale = d < 12 ? 0.008f : 0.015f;
    if ((is_inv || is_cargo) && !is_bwb) scale = 1.0e-4f;
    float g = 1.0f / (1.0f + expf(-gate[d]));
    action[i] = fmaxf(-1.0f, fminf(1.0f,
        prior[i] + scale * g * tanhf(0.5f * raw[i])));
}
__global__ static void action_backward(const float *raw, const float *prior,
                                       const float *mission, const float *gate,
                                       const float *dy, float *dx, float *dg,
                                       int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * TB_NET_ACTION_DIM) return;
    int b = (int)(i / TB_NET_ACTION_DIM), d = (int)(i % TB_NET_ACTION_DIM);
    const float *m = mission + (size_t)b * 10;
    int is_inv = m[3] > 0.75f && m[0] < -0.25f;
    int is_cargo = m[2] > 0.75f;
    int is_ar = m[6] > 0.75f;
    int is_lateral = (m[4] > 0.75f && m[5] > 0.25f) || m[5] > 0.75f;
    int is_bwb = (is_ar && (m[4] < -0.25f || m[5] > 0.60f)) ||
                 (m[4] > 0.75f && m[2] > 0.25f);
    float scale = is_ar || is_lateral ? (d < 12 ? 0.012f : 0.18f) : 0.035f;
    if (is_bwb) scale = d < 12 ? 0.008f : 0.015f;
    if ((is_inv || is_cargo) && !is_bwb) scale = 1.0e-4f;
    float g = 1.0f / (1.0f + expf(-gate[d]));
    float t = tanhf(0.5f * raw[i]);
    float unclamped = prior[i] + scale * g * t;
    if (unclamped <= -1.0f || unclamped >= 1.0f) return;
    dx[i] += dy[i] * scale * g * 0.5f * (1.0f - t * t);
    if (dg) atomicAdd(dg + d, dy[i] * scale * t * g * (1.0f - g));
}
__global__ static void lidar_channels(const float *obs, float *out, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int b = (int)(i / (6 * 16 * 128));
    int channel = (int)((i / (16 * 128)) % 6);
    int bin = (int)(i % (16 * 128));
    int head = channel % 2;
    float range = obs[(size_t)b * 4148 + 48 + head * 2048 + bin];
    int valid = range > 0.0f && isfinite(range);
    float clamped = fmaxf(0.05f, fminf(30.0f, range));
    out[i] = channel < 2 ? (valid ? 1.0f / fmaxf(0.2f, clamped) : 0.0f) :
             channel < 4 ? (valid ? clamped / 30.0f : 1.0f) : (float)valid;
}
__global__ static void camera_luma(const uint8_t *camera, float *out, size_t count) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (float)camera[i] * (1.0f / 255.0f) - 0.5f;
}
__global__ static void circular_pad_forward(const float *x, float *y,
                                            int batch, int channels,
                                            int h, int w, int ph, int pw) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int oh = h + 2 * ph, ow = w + 2 * pw;
    if (i >= (size_t)batch * channels * oh * ow) return;
    int xx = (int)(i % ow) - pw;
    int yy = (int)((i / ow) % oh) - ph;
    int c = (int)((i / ((size_t)ow * oh)) % channels);
    int b = (int)(i / ((size_t)ow * oh * channels));
    xx = (xx + w) % w;
    yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
    y[i] = x[(((size_t)b * channels + c) * h + yy) * w + xx];
}
__global__ static void circular_pad_backward(const float *dy, float *dx,
                                             int batch, int channels,
                                             int h, int w, int ph, int pw) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    int oh = h + 2 * ph, ow = w + 2 * pw;
    if (i >= (size_t)batch * channels * oh * ow) return;
    int xx = (int)(i % ow) - pw;
    int yy = (int)((i / ow) % oh) - ph;
    int c = (int)((i / ((size_t)ow * oh)) % channels);
    int b = (int)(i / ((size_t)ow * oh * channels));
    xx = (xx + w) % w;
    yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
    atomicAdd(dx + (((size_t)b * channels + c) * h + yy) * w + xx, dy[i]);
}
__global__ static void groupnorm_stats(const float *x, float *mean, float *inv,
                                       int batch, int channels, int spatial, int groups) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch * groups) return;
    int b = row / groups, g = row % groups;
    int channels_per_group = channels / groups;
    int n = channels_per_group * spatial;
    size_t base = ((size_t)b * channels + g * channels_per_group) * spatial;
    double sum = 0.0, squares = 0.0;
    for (int i = 0; i < n; ++i) { float v = x[base + i]; sum += v; squares += (double)v * v; }
    float mu = (float)(sum / n);
    float var = fmaxf(0.0f, (float)(squares / n) - mu * mu);
    mean[row] = mu;
    inv[row] = rsqrtf(var + 1.0e-5f);
}
__global__ static void groupnorm_apply(const float *x, const float *gamma, const float *beta,
                                       const float *mean, const float *inv, float *y,
                                       int batch, int channels, int spatial, int groups) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * channels * spatial) return;
    int c = (int)((i / spatial) % channels);
    int b = (int)(i / ((size_t)channels * spatial));
    int g = c / (channels / groups);
    y[i] = (x[i] - mean[b * groups + g]) * inv[b * groups + g] * gamma[c] + beta[c];
}
__global__ static void groupnorm_backward(const float *x, const float *gamma,
                                          const float *dy, const float *mean,
                                          const float *inv, float *dx,
                                          float *dgamma, float *dbeta,
                                          int batch, int channels, int spatial, int groups) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch * groups) return;
    int b = row / groups, g = row % groups;
    int cpg = channels / groups;
    int n = cpg * spatial;
    size_t base = ((size_t)b * channels + g * cpg) * spatial;
    float mu = mean[row], inv_std = inv[row];
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; ++i) {
        int c = g * cpg + i / spatial;
        float v = dy[base + i] * gamma[c];
        sum1 += v;
        sum2 += v * ((x[base + i] - mu) * inv_std);
    }
    float m1 = (float)(sum1 / n), m2 = (float)(sum2 / n);
    for (int i = 0; i < n; ++i) {
        int c = g * cpg + i / spatial;
        float normalized = (x[base + i] - mu) * inv_std;
        float grad = dy[base + i];
        dx[base + i] += inv_std * (grad * gamma[c] - m1 - normalized * m2);
        if (dgamma) atomicAdd(dgamma + c, grad * normalized);
        if (dbeta) atomicAdd(dbeta + c, grad);
    }
}
__global__ static void camera_summary_forward(const float *x, float *y,
                                              int rows, int channels) {
    int rc = blockIdx.x * blockDim.x + threadIdx.x;
    if (rc >= rows * channels) return;
    int r = rc / channels, c = rc % channels;
    const float *feat = x + ((size_t)r * channels + c) * 256;
    float max_value = -1.0e30f;
    for (int i = 0; i < 256; ++i) max_value = fmaxf(max_value, feat[i]);
    float sum = 0.0f, ex = 0.0f, ey = 0.0f;
    for (int i = 0; i < 256; ++i) {
        float p = expf(feat[i] - max_value);
        sum += p;
        ex += p * ((i % 16) * (2.0f / 15.0f) - 1.0f);
        ey += p * ((i / 16) * (2.0f / 15.0f) - 1.0f);
    }
    y[(size_t)r * (3 * channels) + c] = ex / sum;
    y[(size_t)r * (3 * channels) + channels + c] = ey / sum;
    y[(size_t)r * (3 * channels) + 2 * channels + c] = max_value;
}
__global__ static void camera_summary_backward(const float *x, const float *y,
                                               const float *dy, float *dx,
                                               int rows, int channels) {
    int rc = blockIdx.x * blockDim.x + threadIdx.x;
    if (rc >= rows * channels) return;
    int r = rc / channels, c = rc % channels;
    const float *feat = x + ((size_t)r * channels + c) * 256;
    float *grad = dx + ((size_t)r * channels + c) * 256;
    float max_value = y[(size_t)r * (3 * channels) + 2 * channels + c];
    float ex = y[(size_t)r * (3 * channels) + c];
    float ey = y[(size_t)r * (3 * channels) + channels + c];
    float sum = 0.0f;
    int max_index = 0;
    for (int i = 0; i < 256; ++i) {
        sum += expf(feat[i] - max_value);
        if (feat[i] > feat[max_index]) max_index = i;
    }
    float gx = dy[(size_t)r * (3 * channels) + c];
    float gy = dy[(size_t)r * (3 * channels) + channels + c];
    float gm = dy[(size_t)r * (3 * channels) + 2 * channels + c];
    for (int i = 0; i < 256; ++i) {
        float px = (i % 16) * (2.0f / 15.0f) - 1.0f;
        float py = (i / 16) * (2.0f / 15.0f) - 1.0f;
        grad[i] += expf(feat[i] - max_value) / sum *
            (gx * (px - ex) + gy * (py - ey)) + (i == max_index ? gm : 0.0f);
    }
}
__global__ static void lidar_sector_forward(const float *x, float *y, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * 4 * 2304) return;
    int b = (int)(i / (4 * 2304));
    int sector = (int)((i / 2304) % 4);
    int f = (int)(i % 2304);
    int sub = f / (288 * 2), c = (f / 2) % 288, h = f % 2;
    y[i] = x[(((size_t)b * 288 + c) * 2 + h) * 16 + sector * 4 + sub];
}
__global__ static void lidar_sector_backward(const float *dy, float *dx, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * 4 * 2304) return;
    int b = (int)(i / (4 * 2304));
    int sector = (int)((i / 2304) % 4);
    int f = (int)(i % 2304);
    int sub = f / (288 * 2), c = (f / 2) % 288, h = f % 2;
    dx[(((size_t)b * 288 + c) * 2 + h) * 16 + sector * 4 + sub] += dy[i];
}

static TbActorNode *new_actor_node(tb_net_gpu *ctx, TbNodeOp op, int rows, int cols,
                                    int a, int b) {
    if (!ctx || ctx->actor_node_count >= TB_ACTOR_NODES || rows < 1 || cols < 1)
        return NULL;
    TbActorNode *n = &ctx->actor_nodes[ctx->actor_node_count++];
    memset(n, 0, sizeof(*n));
    n->op = op; n->rows = rows; n->cols = cols; n->a = a; n->b = b;
    n->v = trace_alloc(ctx, &ctx->actor_trace, (size_t)rows * cols);
    n->g = trace_alloc(ctx, &ctx->actor_trace, (size_t)rows * cols);
    if (!n->v || !n->g || !tb_gpu_zero(ctx->ops, n->g, (size_t)rows * cols))
        return NULL;
    return n;
}
static int actor_input(tb_net_gpu *ctx, const float *device_data, int rows, int cols) {
    if (!ctx || !device_data || rows < 1 || cols < 1 ||
        ctx->actor_node_count >= TB_ACTOR_NODES) return -1;
    TbActorNode *n = &ctx->actor_nodes[ctx->actor_node_count++];
    memset(n, 0, sizeof(*n));
    n->op = TB_NODE_INPUT;
    n->rows = rows; n->cols = cols; n->a = n->b = -1;
    n->v = (float *)device_data;
    n->g = trace_alloc(ctx, &ctx->actor_trace, (size_t)rows * cols);
    if (!n->g || !tb_gpu_zero(ctx->ops, n->g, (size_t)rows * cols)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_linear(tb_net_gpu *ctx, int input, const char *prefix, int out_dim) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    const tb_net_tensor_info *w, *bias;
    char name[96];
    if (snprintf(name, sizeof(name), "%s.weight", prefix) >= (int)sizeof(name)) return -1;
    w = tb_net_find_tensor(ctx->layout, name);
    if (snprintf(name, sizeof(name), "%s.bias", prefix) >= (int)sizeof(name)) return -1;
    bias = tb_net_find_tensor(ctx->layout, name);
    if (!w || w->shape[0] != out_dim || w->shape[1] != x->cols) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_LINEAR, x->rows, out_dim, input, -1);
    if (!n) return -1;
    n->weight = w; n->bias = bias;
    if (!tb_gpu_linear_fwd(ctx->ops, x->v, ctx->parameters + w->offset,
                           bias ? ctx->parameters + bias->offset : NULL,
                           n->v, x->rows, x->cols, out_dim)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_norm(tb_net_gpu *ctx, int input, const char *prefix) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    char name[96];
    if (snprintf(name, sizeof(name), "%s.weight", prefix) >= (int)sizeof(name)) return -1;
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, name);
    if (snprintf(name, sizeof(name), "%s.bias", prefix) >= (int)sizeof(name)) return -1;
    const tb_net_tensor_info *bias = tb_net_find_tensor(ctx->layout, name);
    if (!w || !bias || w->count != (uint64_t)x->cols) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_NORM, x->rows, x->cols, input, -1);
    if (!n) return -1;
    n->weight = w; n->bias = bias;
    n->mean = trace_alloc(ctx, &ctx->actor_trace, x->rows);
    n->inv_std = trace_alloc(ctx, &ctx->actor_trace, x->rows);
    if (!n->mean || !n->inv_std ||
        !tb_gpu_layernorm_fwd(ctx->ops, x->v, ctx->parameters + w->offset,
                              ctx->parameters + bias->offset, n->v,
                              n->mean, n->inv_std, x->rows, x->cols, 1.0e-5f)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_silu(tb_net_gpu *ctx, int input) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_SILU, x->rows, x->cols, input, -1);
    if (!n || !tb_gpu_silu_fwd(ctx->ops, x->v, n->v, (size_t)x->rows * x->cols)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_add(tb_net_gpu *ctx, int a, int b) {
    if (!ctx || a < 0 || b < 0 || a >= ctx->actor_node_count || b >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[a], *y = &ctx->actor_nodes[b];
    if (x->rows != y->rows || x->cols != y->cols) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_ADD, x->rows, x->cols, a, b);
    if (!n || !tb_gpu_copy(ctx->ops, n->v, x->v, (size_t)x->rows * x->cols) ||
        !tb_gpu_axpy(ctx->ops, n->v, y->v, 1.0f, (size_t)x->rows * x->cols)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_concat(tb_net_gpu *ctx, int a, int b) {
    if (!ctx || a < 0 || b < 0 || a >= ctx->actor_node_count || b >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[a], *y = &ctx->actor_nodes[b];
    if (x->rows != y->rows) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_CONCAT, x->rows, x->cols + y->cols, a, b);
    if (!n) return -1;
    size_t count = (size_t)n->rows * n->cols;
    concat_rows_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                          (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, y->v, n->v, n->rows, x->cols, y->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_embed(tb_net_gpu *ctx, int input, const char *name, int tokens) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, name);
    if (!w || x->cols != TB_NET_D_MODEL || x->rows % tokens ||
        w->count != (uint64_t)tokens * TB_NET_D_MODEL) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_EMBED, x->rows, x->cols, input, -1);
    if (!n) return -1;
    n->weight = w; n->aux[0] = tokens;
    size_t count = (size_t)x->rows * x->cols;
    embed_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                    (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, ctx->parameters + w->offset, n->v, x->rows / tokens, tokens, x->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_pool(tb_net_gpu *ctx, int input, int tokens) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    if (x->rows % tokens) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_POOL, x->rows / tokens, x->cols, input, -1);
    if (!n) return -1;
    n->aux[0] = tokens;
    size_t count = (size_t)n->rows * n->cols;
    pool_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                   (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, n->v, n->rows, tokens, n->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_pick(tb_net_gpu *ctx, int input, int tokens, int which) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count || which < 0 || which >= tokens) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    if (x->rows % tokens) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_PICK, x->rows / tokens, x->cols, input, -1);
    if (!n) return -1;
    n->aux[0] = tokens; n->aux[1] = which;
    size_t count = (size_t)n->rows * n->cols;
    pick_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                   (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, n->v, n->rows, tokens, n->cols, which);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_clamp(tb_net_gpu *ctx, int input, float low, float high) {
    if (!ctx || input < 0 || input >= ctx->actor_node_count) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_CLAMP, x->rows, x->cols, input, -1);
    if (!n) return -1;
    n->aux[0] = (int)(low * 1000); n->aux[1] = (int)(high * 1000);
    size_t count = (size_t)n->rows * n->cols;
    clamp_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                      (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(x->v, n->v, count, low, high);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_attention(tb_net_gpu *ctx, int q_index, int kv_index,
                            int batch, int query_tokens, int key_tokens,
                            int self_qkv) {
    if (!ctx || q_index < 0 || kv_index < 0 || q_index >= ctx->actor_node_count ||
        kv_index >= ctx->actor_node_count || key_tokens > TB_NET_SENSOR_TOKENS) return -1;
    TbActorNode *q = &ctx->actor_nodes[q_index], *kv = &ctx->actor_nodes[kv_index];
    if (q->rows != batch * query_tokens || kv->rows != batch * key_tokens ||
        q->cols != (self_qkv ? 3 : 1) * TB_NET_D_MODEL ||
        kv->cols != (self_qkv ? 3 : 2) * TB_NET_D_MODEL) return -1;
    TbActorNode *n = new_actor_node(ctx, self_qkv ? TB_NODE_SELF_ATTN : TB_NODE_CROSS_ATTN,
                                    batch * query_tokens, TB_NET_D_MODEL,
                                    q_index, kv_index);
    if (!n) return -1;
    n->aux[0] = batch; n->aux[1] = query_tokens; n->aux[2] = key_tokens;
    n->mean = trace_alloc(ctx, &ctx->actor_trace,
        (size_t)batch * TB_NET_HEADS * query_tokens * key_tokens);
    if (!n->mean) return -1;
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(ctx->ops);
    int rows = batch * TB_NET_HEADS * query_tokens;
    attention_scores<<<(unsigned)((rows + 127) / 128), 128, 0, stream>>>(
        q->v, kv->v, n->mean, batch, query_tokens, key_tokens, self_qkv);
    size_t count = (size_t)batch * query_tokens * TB_NET_D_MODEL;
    attention_values<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
        kv->v, n->mean, n->v, batch, query_tokens, key_tokens, self_qkv);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}

static int actor_pack(tb_net_gpu *ctx, int p, int m, int l, int c, int batch) {
    if (p < 0 || m < 0 || l < 0 || c < 0) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_PACK, batch * 10, TB_NET_D_MODEL, p, m);
    if (!n) return -1;
    n->aux[0] = l; n->aux[1] = c;
    size_t count = (size_t)batch * 10 * TB_NET_D_MODEL;
    pack_tokens_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                          (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        ctx->actor_nodes[p].v, ctx->actor_nodes[m].v,
        ctx->actor_nodes[l].v, ctx->actor_nodes[c].v, n->v, batch);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_query(tb_net_gpu *ctx, int batch) {
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, "action_expert.action_queries");
    if (!w) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_QUERY,
                                    batch * TB_NET_ACTION_HORIZON, TB_NET_D_MODEL, -1, -1);
    if (!n) return -1;
    n->weight = w;
    size_t count = (size_t)n->rows * n->cols;
    query_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                    (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        ctx->parameters + w->offset, n->v, batch, TB_NET_ACTION_HORIZON, TB_NET_D_MODEL);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_broadcast_add(tb_net_gpu *ctx, int x, int v, int tokens) {
    if (x < 0 || v < 0) return -1;
    TbActorNode *a = &ctx->actor_nodes[x], *b = &ctx->actor_nodes[v];
    if (a->rows != b->rows * tokens || a->cols != b->cols) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_BROADCAST_ADD, a->rows, a->cols, x, v);
    if (!n) return -1;
    n->aux[0] = tokens;
    size_t count = (size_t)n->rows * n->cols;
    broadcast_add_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                            (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        a->v, b->v, n->v, b->rows, tokens, b->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_swiglu(tb_net_gpu *ctx, int input) {
    if (input < 0) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    if (x->cols != 2 * TB_NET_MOE_EXPERT_DIM) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_SWIGLU,
                                    x->rows, TB_NET_MOE_EXPERT_DIM, input, -1);
    if (!n) return -1;
    size_t count = (size_t)n->rows * n->cols;
    swiglu_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                     (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(x->v, n->v, n->rows, n->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_softmax(tb_net_gpu *ctx, int input) {
    if (input < 0) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_SOFTMAX, x->rows, x->cols, input, -1);
    if (!n || !tb_gpu_softmax_fwd(ctx->ops, x->v, n->v, n->rows, n->cols)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_moe_mix(tb_net_gpu *ctx, int prob, const int experts[6]) {
    if (prob < 0) return -1;
    TbActorNode *p = &ctx->actor_nodes[prob];
    if (p->cols != 6) return -1;
    for (int e = 0; e < 6; ++e)
        if (experts[e] < 0 || ctx->actor_nodes[experts[e]].rows != p->rows ||
            ctx->actor_nodes[experts[e]].cols != TB_NET_D_MODEL) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_MOE_MIX,
                                    p->rows, TB_NET_D_MODEL, prob, -1);
    if (!n) return -1;
    for (int e = 0; e < 6; ++e) n->aux[e] = experts[e];
    size_t count = (size_t)n->rows * n->cols;
    moe_mix_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                      (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        p->v, ctx->actor_nodes[experts[0]].v, ctx->actor_nodes[experts[1]].v,
        ctx->actor_nodes[experts[2]].v, ctx->actor_nodes[experts[3]].v,
        ctx->actor_nodes[experts[4]].v, ctx->actor_nodes[experts[5]].v,
        n->v, n->rows, n->cols);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_action(tb_net_gpu *ctx, int raw, int prior, int batch) {
    if (raw < 0 || prior < 0) return -1;
    TbActorNode *x = &ctx->actor_nodes[raw], *p = &ctx->actor_nodes[prior];
    if (x->rows != batch || x->cols != TB_NET_ACTION_DIM || p->rows != batch ||
        p->cols != TB_NET_ACTION_DIM) return -1;
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, "residual_gate_logit");
    if (!w) return -1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_ACTION,
                                    batch, TB_NET_ACTION_DIM, raw, prior);
    if (!n) return -1;
    n->weight = w;
    size_t count = (size_t)batch * TB_NET_ACTION_DIM;
    action_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                     (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, p->v, ctx->actor_mission, ctx->parameters + w->offset, n->v, batch);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_circular_pad(tb_net_gpu *ctx, int input,
                              int channels, int height, int width, int ph, int pw) {
    if (input < 0 || ctx->actor_nodes[input].cols != channels * height * width) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    int oh = height + 2 * ph, ow = width + 2 * pw;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_CIRC_PAD,
                                    x->rows, channels * oh * ow, input, -1);
    if (!n) return -1;
    n->aux[0] = channels; n->aux[1] = height; n->aux[2] = width;
    n->pad_h = ph; n->pad_w = pw;
    size_t count = (size_t)n->rows * n->cols;
    circular_pad_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                           (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        x->v, n->v, n->rows, channels, height, width, ph, pw);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_conv(tb_net_gpu *ctx, int input, const char *name,
                      int in_channels, int out_channels, int height, int width,
                      int kh, int kw, int sh, int sw, int ph, int pw) {
    if (input < 0 || ctx->actor_nodes[input].cols != in_channels * height * width) return -1;
    char pname[96];
    if (snprintf(pname, sizeof(pname), "%s.weight", name) >= (int)sizeof(pname)) return -1;
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, pname);
    if (!w || w->rank != 4 || w->shape[0] != out_channels || w->shape[1] != in_channels ||
        w->shape[2] != kh || w->shape[3] != kw) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    int oh = (height + 2 * ph - kh) / sh + 1, ow = (width + 2 * pw - kw) / sw + 1;
    TbActorNode *n = new_actor_node(ctx, TB_NODE_CONV,
                                    x->rows, out_channels * oh * ow, input, -1);
    if (!n) return -1;
    n->weight = w;
    n->aux[0] = in_channels; n->aux[1] = out_channels;
    n->aux[2] = height; n->aux[3] = width;
    n->aux[4] = kh; n->aux[5] = kw;
    n->aux[6] = sh; n->aux[7] = sw;
    n->pad_h = ph; n->pad_w = pw;
    if (!tb_gpu_conv2d_fwd(ctx->ops, x->v, ctx->parameters + w->offset,
                            NULL, n->v, x->rows, in_channels, out_channels,
                            height, width, kh, kw, sh, sw, ph, pw)) return -1;
    return ctx->actor_node_count - 1;
}
static int actor_groupnorm(tb_net_gpu *ctx, int input, const char *name,
                           int channels, int height, int width, int groups) {
    if (input < 0 || channels % groups ||
        ctx->actor_nodes[input].cols != channels * height * width) return -1;
    char pname[96];
    snprintf(pname, sizeof(pname), "%s.weight", name);
    const tb_net_tensor_info *w = tb_net_find_tensor(ctx->layout, pname);
    snprintf(pname, sizeof(pname), "%s.bias", name);
    const tb_net_tensor_info *bias = tb_net_find_tensor(ctx->layout, pname);
    if (!w || !bias || w->count != (uint64_t)channels || bias->count != (uint64_t)channels) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_GN, x->rows, x->cols, input, -1);
    if (!n) return -1;
    n->weight = w; n->bias = bias;
    n->aux[0] = channels; n->aux[1] = height * width; n->aux[2] = groups;
    n->mean = trace_alloc(ctx, &ctx->actor_trace, (size_t)x->rows * groups);
    n->inv_std = trace_alloc(ctx, &ctx->actor_trace, (size_t)x->rows * groups);
    if (!n->mean || !n->inv_std) return -1;
    int rows = x->rows * groups;
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(ctx->ops);
    groupnorm_stats<<<(unsigned)((rows + 127) / 128), 128, 0, stream>>>(
        x->v, n->mean, n->inv_std, x->rows, channels, height * width, groups);
    size_t count = (size_t)n->rows * n->cols;
    groupnorm_apply<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
        x->v, ctx->parameters + w->offset, ctx->parameters + bias->offset,
        n->mean, n->inv_std, n->v, x->rows, channels, height * width, groups);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_camera_summary(tb_net_gpu *ctx, int input) {
    if (input < 0 || ctx->actor_nodes[input].cols != 192 * 16 * 16) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_CAM_SUMMARY, x->rows, 576, input, -1);
    if (!n) return -1;
    int count = x->rows * 192;
    camera_summary_forward<<<(unsigned)((count + 127) / 128), 128, 0,
                             (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(x->v, n->v, x->rows, 192);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}
static int actor_lidar_sector(tb_net_gpu *ctx, int input) {
    if (input < 0 || ctx->actor_nodes[input].cols != 288 * 2 * 16) return -1;
    TbActorNode *x = &ctx->actor_nodes[input];
    TbActorNode *n = new_actor_node(ctx, TB_NODE_LIDAR_SECTOR, x->rows * 4, 2304, input, -1);
    if (!n) return -1;
    size_t count = (size_t)n->rows * n->cols;
    lidar_sector_forward<<<(unsigned)((count + 255) / 256), 256, 0,
                           (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(x->v, n->v, x->rows);
    return cudaGetLastError() == cudaSuccess ? ctx->actor_node_count - 1 : -1;
}

static int actor_backward_node(tb_net_gpu *ctx, int index) {
    TbActorNode *n = &ctx->actor_nodes[index];
    TbActorNode *a = n->a >= 0 ? &ctx->actor_nodes[n->a] : NULL;
    TbActorNode *b = n->b >= 0 ? &ctx->actor_nodes[n->b] : NULL;
    size_t count = (size_t)n->rows * n->cols;
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(ctx->ops);
    switch (n->op) {
    case TB_NODE_INPUT: return 1;
    case TB_NODE_LINEAR:
        return a && tb_gpu_linear_bwd(ctx->ops, a->v,
            ctx->parameters + n->weight->offset, n->g, a->g,
            ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            ctx->gradients && n->bias ? ctx->gradients + n->bias->offset : NULL,
            n->rows, a->cols, n->cols, 1);
    case TB_NODE_NORM:
        return a && tb_gpu_layernorm_bwd(ctx->ops, a->v,
            ctx->parameters + n->weight->offset, n->g, n->mean, n->inv_std,
            a->g, ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            ctx->gradients ? ctx->gradients + n->bias->offset : NULL,
            n->rows, n->cols, 1);
    case TB_NODE_SILU:
        return a && tb_gpu_silu_bwd(ctx->ops, a->v, n->g, a->g, count, 1);
    case TB_NODE_ADD:
        return a && b && tb_gpu_axpy(ctx->ops, a->g, n->g, 1.0f, count) &&
               tb_gpu_axpy(ctx->ops, b->g, n->g, 1.0f, count);
    case TB_NODE_CONCAT:
        if (!a || !b) return 0;
        concat_rows_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, b->g, n->rows, a->cols, b->cols);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_EMBED:
        if (!a) return 0;
        embed_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            n->rows / n->aux[0], n->aux[0], n->cols);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_POOL:
        if (!a) return 0;
        pool_backward<<<(unsigned)(((size_t)a->rows * a->cols + 255) / 256),
                        256, 0, stream>>>(n->g, a->g, n->rows, n->aux[0], n->cols);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_PICK:
        if (!a) return 0;
        pick_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, n->rows, n->aux[0], n->cols, n->aux[1]);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_CLAMP:
        if (!a) return 0;
        clamp_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            a->v, n->g, a->g, count,
            n->aux[0] * 0.001f, n->aux[1] * 0.001f);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_SELF_ATTN:
    case TB_NODE_CROSS_ATTN:
        if (!a || !b) return 0;
        attention_backward<<<(unsigned)((n->aux[0] * TB_NET_HEADS * n->aux[1] + 127) / 128),
                             128, 0, stream>>>(
            a->v, b->v, n->mean, n->g, a->g, b->g,
            n->aux[0], n->aux[1], n->aux[2], n->op == TB_NODE_SELF_ATTN);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_PACK:
        if (!a || !b) return 0;
        pack_tokens_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, b->g,
            ctx->actor_nodes[n->aux[0]].g, ctx->actor_nodes[n->aux[1]].g,
            n->rows / 10);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_QUERY:
        if (ctx->gradients)
            query_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
                n->g, ctx->gradients + n->weight->offset,
                n->rows / TB_NET_ACTION_HORIZON,
                TB_NET_ACTION_HORIZON, TB_NET_D_MODEL);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_BROADCAST_ADD:
        if (!a || !b) return 0;
        broadcast_add_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, b->g, b->rows, n->aux[0], n->cols);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_SWIGLU:
        if (!a) return 0;
        swiglu_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            a->v, n->g, a->g, n->rows, n->cols);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_SOFTMAX:
        return a && tb_gpu_softmax_bwd(ctx->ops, n->v, n->g, a->g,
                                        n->rows, n->cols, 1);
    case TB_NODE_MOE_MIX: {
        if (!a) return 0;
        TbActorNode *e0 = &ctx->actor_nodes[n->aux[0]], *e1 = &ctx->actor_nodes[n->aux[1]];
        TbActorNode *e2 = &ctx->actor_nodes[n->aux[2]], *e3 = &ctx->actor_nodes[n->aux[3]];
        TbActorNode *e4 = &ctx->actor_nodes[n->aux[4]], *e5 = &ctx->actor_nodes[n->aux[5]];
        moe_mix_backward<<<(unsigned)((n->rows + 127) / 128), 128, 0, stream>>>(
            a->v, e0->v,e1->v,e2->v,e3->v,e4->v,e5->v, n->g, a->g,
            e0->g,e1->g,e2->g,e3->g,e4->g,e5->g, n->rows, n->cols);
        return cudaGetLastError() == cudaSuccess;
    }
    case TB_NODE_ACTION:
        if (!a || !b) return 0;
        action_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            a->v, b->v, ctx->actor_mission,
            ctx->parameters + n->weight->offset, n->g, a->g,
            ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            n->rows);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_CIRC_PAD:
        if (!a) return 0;
        circular_pad_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, n->rows, n->aux[0], n->aux[1], n->aux[2], n->pad_h, n->pad_w);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_CONV:
        return a && tb_gpu_conv2d_bwd(ctx->ops, a->v,
            ctx->parameters + n->weight->offset, n->g, a->g,
            ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            NULL, n->rows, n->aux[0], n->aux[1], n->aux[2], n->aux[3],
            n->aux[4], n->aux[5], n->aux[6], n->aux[7], n->pad_h, n->pad_w, 1);
    case TB_NODE_GN:
        if (!a) return 0;
        groupnorm_backward<<<(unsigned)((n->rows * n->aux[2] + 127) / 128), 128, 0, stream>>>(
            a->v, ctx->parameters + n->weight->offset, n->g,
            n->mean, n->inv_std, a->g,
            ctx->gradients ? ctx->gradients + n->weight->offset : NULL,
            ctx->gradients ? ctx->gradients + n->bias->offset : NULL,
            n->rows, n->aux[0], n->aux[1], n->aux[2]);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_CAM_SUMMARY:
        if (!a) return 0;
        camera_summary_backward<<<(unsigned)((n->rows * 192 + 127) / 128), 128, 0, stream>>>(
            a->v, n->v, n->g, a->g, n->rows, 192);
        return cudaGetLastError() == cudaSuccess;
    case TB_NODE_LIDAR_SECTOR:
        if (!a) return 0;
        lidar_sector_backward<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
            n->g, a->g, a->rows);
        return cudaGetLastError() == cudaSuccess;
    default: return net_error(ctx, "actor tape operation unavailable");
    }
}

__global__ static void join_critic_input(const float *summary,
                                          const float *privileged,
                                          const float *action,
                                          float *output, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * TB_CRITIC_INPUT) return;
    int row = (int)(i / TB_CRITIC_INPUT);
    int col = (int)(i % TB_CRITIC_INPUT);
    if (col < 2 * TB_NET_D_MODEL)
        output[i] = summary[(size_t)row * 2 * TB_NET_D_MODEL + col];
    else if (col < 2 * TB_NET_D_MODEL + 64)
        output[i] = privileged[(size_t)row * 64 + col - 2 * TB_NET_D_MODEL];
    else output[i] = action[(size_t)row * TB_NET_ACTION_DIM +
                            col - 2 * TB_NET_D_MODEL - 64];
}

__global__ static void split_critic_gradient(const float *q1, const float *q2,
                                              float *d_summary, float *d_action,
                                              int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * TB_CRITIC_INPUT) return;
    int row = (int)(i / TB_CRITIC_INPUT);
    int col = (int)(i % TB_CRITIC_INPUT);
    float value = q1[i] + q2[i];
    if (col < 2 * TB_NET_D_MODEL) {
        if (d_summary) d_summary[(size_t)row * 2 * TB_NET_D_MODEL + col] = value;
    } else if (col >= 2 * TB_NET_D_MODEL + 64) {
        if (d_action)
            d_action[(size_t)row * TB_NET_ACTION_DIM + col - 2 * TB_NET_D_MODEL - 64] = value;
    }
}

tb_net_gpu *tb_net_gpu_create(const tb_net_model *layout, tb_gpu_ops *ops,
                              float *device_parameters, float *device_gradients,
                              int max_batch) {
    if (!layout || !tb_net_validate_layout(layout) || !ops || !device_parameters ||
        max_batch < 1 || max_batch > 4096) return NULL;
    tb_net_gpu *ctx = (tb_net_gpu *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->layout = layout;
    ctx->ops = ops;
    ctx->parameters = device_parameters;
    ctx->gradients = device_gradients;
    ctx->max_batch = max_batch;
    return ctx;
}

void tb_net_gpu_destroy(tb_net_gpu *ctx) {
    if (!ctx) return;
    trace_clear(ctx, &ctx->actor_trace);
    trace_clear(ctx, &ctx->critic_trace);
    free(ctx);
}

const char *tb_net_gpu_last_error(const tb_net_gpu *ctx) {
    return ctx ? ctx->error : "invalid network context";
}

static int forward_critic_branch(tb_net_gpu *ctx, TbCriticBranch *q,
                                 int index, int batch, float *out_logits) {
    char name[96];
    int in_dim = TB_CRITIC_INPUT;
    float *input = q->input;
    for (int layer = 0; layer < TB_NET_CRITIC_LAYERS; ++layer) {
        q->linear[layer] = trace_alloc(ctx, &ctx->critic_trace,
                                      (size_t)batch * TB_NET_CRITIC_HIDDEN);
        q->norm[layer] = trace_alloc(ctx, &ctx->critic_trace,
                                    (size_t)batch * TB_NET_CRITIC_HIDDEN);
        q->activation[layer] = trace_alloc(ctx, &ctx->critic_trace,
                                          (size_t)batch * TB_NET_CRITIC_HIDDEN);
        q->mean[layer] = trace_alloc(ctx, &ctx->critic_trace, batch);
        q->inv_std[layer] = trace_alloc(ctx, &ctx->critic_trace, batch);
        if (!q->linear[layer] || !q->norm[layer] || !q->activation[layer] ||
            !q->mean[layer] || !q->inv_std[layer])
            return net_error(ctx, "critic scratch allocation failed");
        float *w, *bias, *gamma, *beta;
        if (!named_parameter(ctx, name, sizeof(name), index, layer * 3, "weight", &w, NULL) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3, "bias", &bias, NULL) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3 + 1, "weight", &gamma, NULL) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3 + 1, "bias", &beta, NULL))
            return 0;
        if (!tb_gpu_linear_fwd(ctx->ops, input, w, bias, q->linear[layer],
                               batch, in_dim, TB_NET_CRITIC_HIDDEN) ||
            !tb_gpu_layernorm_fwd(ctx->ops, q->linear[layer], gamma, beta,
                                  q->norm[layer], q->mean[layer], q->inv_std[layer],
                                  batch, TB_NET_CRITIC_HIDDEN, 1.0e-5f) ||
            !tb_gpu_silu_fwd(ctx->ops, q->norm[layer], q->activation[layer],
                             (size_t)batch * TB_NET_CRITIC_HIDDEN))
            return net_error(ctx, "critic hidden forward failed");
        input = q->activation[layer];
        in_dim = TB_NET_CRITIC_HIDDEN;
    }
    float *w, *bias;
    if (!named_parameter(ctx, name, sizeof(name), index, TB_NET_CRITIC_LAYERS * 3,
                         "weight", &w, NULL) ||
        !named_parameter(ctx, name, sizeof(name), index, TB_NET_CRITIC_LAYERS * 3,
                         "bias", &bias, NULL)) return 0;
    if (!tb_gpu_linear_fwd(ctx->ops, input, w, bias, out_logits,
                           batch, TB_NET_CRITIC_HIDDEN, TB_NET_ATOMS))
        return net_error(ctx, "critic logits forward failed");
    q->logits = out_logits;
    return 1;
}

int tb_net_gpu_critic_forward(tb_net_gpu *ctx,
                              const float *summary, const float *privileged,
                              const float *action, int batch,
                              float *logits_q1, float *logits_q2) {
    if (!ctx || !summary || !privileged || !action || !logits_q1 || !logits_q2 ||
        batch < 1 || batch > ctx->max_batch) return 0;
    trace_clear(ctx, &ctx->critic_trace);
    memset(ctx->q, 0, sizeof(ctx->q));
    ctx->critic_batch = batch;
    float *joined = trace_alloc(ctx, &ctx->critic_trace,
                                (size_t)batch * TB_CRITIC_INPUT);
    if (!joined) return net_error(ctx, "critic input allocation failed");
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(ctx->ops);
    size_t count = (size_t)batch * TB_CRITIC_INPUT;
    join_critic_input<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(
        summary, privileged, action, joined, batch);
    if (cudaGetLastError() != cudaSuccess) return net_error(ctx, "critic input join failed");
    ctx->q[0].input = joined;
    ctx->q[1].input = joined;
    return forward_critic_branch(ctx, &ctx->q[0], 0, batch, logits_q1) &&
           forward_critic_branch(ctx, &ctx->q[1], 1, batch, logits_q2);
}

static int backward_critic_branch(tb_net_gpu *ctx, TbCriticBranch *q, int index,
                                  const float *d_logits, float *d_input,
                                  int accumulate_weights) {
    char name[96];
    int batch = ctx->critic_batch;
    float *w, *bias, *dw, *db;
    int output_layer = TB_NET_CRITIC_LAYERS * 3;
    if (!named_parameter(ctx, name, sizeof(name), index, output_layer,
                         "weight", &w, &dw) ||
        !named_parameter(ctx, name, sizeof(name), index, output_layer,
                         "bias", &bias, &db)) return 0;
    float *d_activation = trace_alloc(ctx, &ctx->critic_trace,
                                      (size_t)batch * TB_NET_CRITIC_HIDDEN);
    if (!d_activation || !tb_gpu_zero(ctx->ops, d_activation,
                                      (size_t)batch * TB_NET_CRITIC_HIDDEN)) return 0;
    if (!tb_gpu_linear_bwd(ctx->ops, q->activation[TB_NET_CRITIC_LAYERS - 1], w,
                           d_logits, d_activation,
                           accumulate_weights ? dw : NULL,
                           accumulate_weights ? db : NULL,
                           batch, TB_NET_CRITIC_HIDDEN, TB_NET_ATOMS,
                           accumulate_weights))
        return net_error(ctx, "critic logits backward failed");
    for (int layer = TB_NET_CRITIC_LAYERS - 1; layer >= 0; --layer) {
        float *d_norm = trace_alloc(ctx, &ctx->critic_trace,
                                    (size_t)batch * TB_NET_CRITIC_HIDDEN);
        float *d_linear = trace_alloc(ctx, &ctx->critic_trace,
                                      (size_t)batch * TB_NET_CRITIC_HIDDEN);
        float *d_previous = layer ? trace_alloc(ctx, &ctx->critic_trace,
                                                (size_t)batch * TB_NET_CRITIC_HIDDEN) : d_input;
        if (!d_norm || !d_linear || !d_previous) return net_error(ctx, "critic backward allocation failed");
        float *gamma, *dgamma, *beta, *dbeta;
        if (!named_parameter(ctx, name, sizeof(name), index, layer * 3 + 1,
                             "weight", &gamma, &dgamma) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3 + 1,
                             "bias", &beta, &dbeta) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3,
                             "weight", &w, &dw) ||
            !named_parameter(ctx, name, sizeof(name), index, layer * 3,
                             "bias", &bias, &db)) return 0;
        size_t hidden_count = (size_t)batch * TB_NET_CRITIC_HIDDEN;
        if (!tb_gpu_zero(ctx->ops, d_norm, hidden_count) ||
            !tb_gpu_zero(ctx->ops, d_linear, hidden_count) ||
            !tb_gpu_zero(ctx->ops, d_previous,
                         (size_t)batch * (layer ? TB_NET_CRITIC_HIDDEN : TB_CRITIC_INPUT)) ||
            !tb_gpu_silu_bwd(ctx->ops, q->norm[layer], d_activation,
                             d_norm, hidden_count, 0) ||
            !tb_gpu_layernorm_bwd(ctx->ops, q->linear[layer], gamma, d_norm,
                                  q->mean[layer], q->inv_std[layer], d_linear,
                                  accumulate_weights ? dgamma : NULL,
                                  accumulate_weights ? dbeta : NULL,
                                  batch, TB_NET_CRITIC_HIDDEN, accumulate_weights) ||
            !tb_gpu_linear_bwd(ctx->ops,
                               layer ? q->activation[layer - 1] : q->input,
                               w, d_linear, d_previous,
                               accumulate_weights ? dw : NULL,
                               accumulate_weights ? db : NULL,
                               batch,
                               layer ? TB_NET_CRITIC_HIDDEN : TB_CRITIC_INPUT,
                               TB_NET_CRITIC_HIDDEN, accumulate_weights))
            return net_error(ctx, "critic hidden backward failed");
        d_activation = d_previous;
    }
    return 1;
}

int tb_net_gpu_critic_backward(tb_net_gpu *ctx,
                               const float *d_logits_q1,
                               const float *d_logits_q2,
                               float *d_summary, float *d_action,
                               int accumulate_parameter_gradients) {
    if (!ctx || !d_logits_q1 || !d_logits_q2 || ctx->critic_batch < 1 ||
        (accumulate_parameter_gradients && !ctx->gradients)) return 0;
    int batch = ctx->critic_batch;
    float *d1 = trace_alloc(ctx, &ctx->critic_trace,
                            (size_t)batch * TB_CRITIC_INPUT);
    float *d2 = trace_alloc(ctx, &ctx->critic_trace,
                            (size_t)batch * TB_CRITIC_INPUT);
    if (!d1 || !d2) return net_error(ctx, "critic input gradient allocation failed");
    if (!backward_critic_branch(ctx, &ctx->q[0], 0, d_logits_q1,
                                d1, accumulate_parameter_gradients) ||
        !backward_critic_branch(ctx, &ctx->q[1], 1, d_logits_q2,
                                d2, accumulate_parameter_gradients)) return 0;
    size_t count = (size_t)batch * TB_CRITIC_INPUT;
    split_critic_gradient<<<(unsigned)((count + 255) / 256), 256, 0,
                            (cudaStream_t)tb_gpu_stream(ctx->ops)>>>(
        d1, d2, d_summary, d_action, batch);
    if (cudaGetLastError() != cudaSuccess) return net_error(ctx, "critic gradient split failed");
    return 1;
}

static int actor_stem(tb_net_gpu *ctx, int x, const char *stem) {
    char name[96];
    snprintf(name, sizeof(name), "%s.0", stem);
    x = actor_linear(ctx, x, name, TB_NET_D_MODEL);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "%s.1", stem);
    x = actor_norm(ctx, x, name);
    if (x < 0) return -1;
    x = actor_silu(ctx, x);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "%s.3", stem);
    x = actor_linear(ctx, x, name, TB_NET_D_MODEL);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "%s.4", stem);
    x = actor_norm(ctx, x, name);
    return x < 0 ? -1 : actor_silu(ctx, x);
}
static int actor_block(tb_net_gpu *ctx, int x, int layer, int batch) {
    char name[96];
    int skip = x;
    snprintf(name, sizeof(name), "blocks.%d.norm1", layer);
    x = actor_norm(ctx, x, name);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "blocks.%d.qkv", layer);
    x = actor_linear(ctx, x, name, 3 * TB_NET_D_MODEL);
    if (x < 0) return -1;
    x = actor_attention(ctx, x, x, batch, TB_NET_SENSOR_TOKENS,
                        TB_NET_SENSOR_TOKENS, 1);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "blocks.%d.out_proj", layer);
    x = actor_linear(ctx, x, name, TB_NET_D_MODEL);
    if (x < 0) return -1;
    x = actor_add(ctx, skip, x);
    if (x < 0) return -1;
    skip = x;
    snprintf(name, sizeof(name), "blocks.%d.norm2", layer);
    x = actor_norm(ctx, x, name);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "blocks.%d.ffn.0", layer);
    x = actor_linear(ctx, x, name, TB_NET_FFN_DIM);
    if (x < 0) return -1;
    x = actor_silu(ctx, x);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "blocks.%d.ffn.2", layer);
    x = actor_linear(ctx, x, name, TB_NET_D_MODEL);
    return x < 0 ? -1 : actor_add(ctx, skip, x);
}
static int actor_moe(tb_net_gpu *ctx, int x, int module) {
    char name[96];
    int skip = x;
    snprintf(name, sizeof(name), "aero_energy_moe.%d.norm", module);
    x = actor_norm(ctx, x, name);
    if (x < 0) return -1;
    int expert_input = x;
    snprintf(name, sizeof(name), "aero_energy_moe.%d.router.0", module);
    x = actor_linear(ctx, x, name, TB_NET_D_MODEL / 2);
    if (x < 0) return -1;
    x = actor_silu(ctx, x);
    if (x < 0) return -1;
    snprintf(name, sizeof(name), "aero_energy_moe.%d.router.2", module);
    x = actor_linear(ctx, x, name, TB_NET_MOE_EXPERTS);
    if (x < 0) return -1;
    int prob = actor_softmax(ctx, x);
    if (prob < 0) return -1;
    int expert[TB_NET_MOE_EXPERTS];
    for (int e = 0; e < TB_NET_MOE_EXPERTS; ++e) {
        snprintf(name, sizeof(name), "aero_energy_moe.%d.experts_gate_up.%d", module, e);
        int gu = actor_linear(ctx, expert_input, name, 2 * TB_NET_MOE_EXPERT_DIM);
        if (gu < 0) return -1;
        gu = actor_swiglu(ctx, gu);
        if (gu < 0) return -1;
        snprintf(name, sizeof(name), "aero_energy_moe.%d.experts_down.%d", module, e);
        expert[e] = actor_linear(ctx, gu, name, TB_NET_D_MODEL);
        if (expert[e] < 0) return -1;
    }
    x = actor_moe_mix(ctx, prob, expert);
    return x < 0 ? -1 : actor_add(ctx, skip, x);
}
static int actor_project(tb_net_gpu *ctx, int x, const char *linear,
                         const char *norm, int dim) {
    x = actor_linear(ctx, x, linear, dim);
    if (x < 0) return -1;
    x = actor_norm(ctx, x, norm);
    return x < 0 ? -1 : actor_silu(ctx, x);
}

int tb_net_gpu_actor_forward(tb_net_gpu *ctx, const float *obs, const float *mission,
                             const uint8_t *cameras, int batch, float *action,
                             float *log_std, float *summary, float *aux, float *world) {
    if (!ctx || !obs || !mission || !cameras || !action || !log_std || !summary ||
        batch < 1 || batch > ctx->max_batch) return 0;
    trace_clear(ctx, &ctx->actor_trace);
    ctx->actor_node_count = 0;
    ctx->actor_batch = batch;
    ctx->actor_mission = mission;
    ctx->actor_aux_node = ctx->actor_world_node = -1;
    float *p = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * 91);
    float *m = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * 49);
    float *t = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * 24);
    float *prior = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * TB_NET_ACTION_DIM);
    float *lidar = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * 6 * 16 * 128);
    float *cam = trace_alloc(ctx, &ctx->actor_trace, (size_t)batch * 4 * 64 * 64);
    if (!p || !m || !t || !prior || !lidar || !cam)
        return net_error(ctx, "actor sensor allocation failed");
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(ctx->ops);
    scalar_inputs<<<(unsigned)((batch + 127) / 128), 128, 0, stream>>>(
        obs, mission, p, m, t, prior, batch);
    size_t lidar_count = (size_t)batch * 6 * 16 * 128;
    size_t cam_count = (size_t)batch * 4 * 64 * 64;
    lidar_channels<<<(unsigned)((lidar_count + 255) / 256), 256, 0, stream>>>(obs, lidar, lidar_count);
    camera_luma<<<(unsigned)((cam_count + 255) / 256), 256, 0, stream>>>(cameras, cam, cam_count);
    if (cudaGetLastError() != cudaSuccess) return net_error(ctx, "actor sensor preprocessing failed");
    int p0 = actor_input(ctx, p, batch, 91);
    int m0 = actor_input(ctx, m, batch, 49);
    int t0 = actor_input(ctx, t, batch, 24);
    int prior0 = actor_input(ctx, prior, batch, TB_NET_ACTION_DIM);
    int l0 = actor_input(ctx, lidar, batch, 6 * 16 * 128);
    int c0 = actor_input(ctx, cam, batch * 4, 64 * 64);
    if (p0 < 0 || m0 < 0 || t0 < 0 || prior0 < 0 || l0 < 0 || c0 < 0)
        return net_error(ctx, "actor input tape failed");
    int pp = actor_stem(ctx, p0, "proprio_stem");
    int mm = actor_stem(ctx, m0, "mission_stem");
    int l = actor_circular_pad(ctx, l0, 6, 16, 128, 1, 2);
    l = actor_conv(ctx, l, "lidar_stem.conv1", 6, 144, 18, 132, 3, 5, 2, 2, 0, 0);
    l = actor_groupnorm(ctx, l, "lidar_stem.gn1", 144, 8, 64, 6);
    l = actor_silu(ctx, l);
    l = actor_circular_pad(ctx, l, 144, 8, 64, 1, 2);
    l = actor_conv(ctx, l, "lidar_stem.conv2", 144, 288, 10, 68, 3, 5, 2, 2, 0, 0);
    l = actor_groupnorm(ctx, l, "lidar_stem.gn2", 288, 4, 32, 8);
    l = actor_silu(ctx, l);
    l = actor_circular_pad(ctx, l, 288, 4, 32, 0, 1);
    l = actor_conv(ctx, l, "lidar_stem.conv3", 288, 288, 4, 34, 2, 3, 2, 2, 0, 0);
    l = actor_groupnorm(ctx, l, "lidar_stem.gn3", 288, 2, 16, 8);
    l = actor_silu(ctx, l);
    l = actor_lidar_sector(ctx, l);
    int c = actor_conv(ctx, c0, "camera_stem.conv.0", 1, 96, 64, 64, 5, 5, 2, 2, 2, 2);
    c = actor_groupnorm(ctx, c, "camera_stem.conv.1", 96, 32, 32, 4);
    c = actor_silu(ctx, c);
    c = actor_conv(ctx, c, "camera_stem.conv.3", 96, 192, 32, 32, 3, 3, 2, 2, 1, 1);
    c = actor_groupnorm(ctx, c, "camera_stem.conv.4", 192, 16, 16, 8);
    c = actor_silu(ctx, c);
    c = actor_conv(ctx, c, "camera_stem.conv.6", 192, 192, 16, 16, 3, 3, 1, 1, 1, 1);
    c = actor_groupnorm(ctx, c, "camera_stem.conv.7", 192, 16, 16, 8);
    c = actor_silu(ctx, c);
    c = actor_camera_summary(ctx, c);
    if (l < 0 || c < 0) return net_error(ctx, "actor convolutional stem failed");
    int ll = actor_project(ctx, l, "lidar_stem.sector_proj.0",
                           "lidar_stem.sector_proj.1", TB_NET_D_MODEL);
    int cc = actor_project(ctx, c, "camera_stem.proj.0",
                           "camera_stem.proj.1", TB_NET_D_MODEL);
    if (pp < 0 || mm < 0 || ll < 0 || cc < 0)
        return net_error(ctx, "actor sensor stem failed");
    ll = actor_embed(ctx, ll, "lidar_stem.sector_emb", 4);
    cc = actor_embed(ctx, cc, "camera_stem.cam_id_emb", 4);
    int x = actor_pack(ctx, pp, mm, ll, cc, batch);
    if (x < 0) return net_error(ctx, "actor token packing failed");
    x = actor_embed(ctx, x, "modality_emb", TB_NET_SENSOR_TOKENS);
    if (x < 0) return net_error(ctx, "actor modality embedding failed");
    for (int layer = 0; layer < TB_NET_FUSION_LAYERS; ++layer) {
        x = actor_block(ctx, x, layer, batch);
        if (x < 0) return net_error(ctx, "actor fusion block failed");
        if (layer == 5 || layer == 11) {
            x = actor_moe(ctx, x, layer == 5 ? 0 : 1);
            if (x < 0) return net_error(ctx, "actor expert routing failed");
        }
    }
    x = actor_norm(ctx, x, "trunk_norm");
    if (x < 0) return net_error(ctx, "actor trunk normalization failed");
    int token0 = actor_pick(ctx, x, TB_NET_SENSOR_TOKENS, 0);
    int token1 = actor_pick(ctx, x, TB_NET_SENSOR_TOKENS, 1);
    int reflex = actor_concat(ctx, token0, token1);
    reflex = actor_concat(ctx, reflex, t0);
    reflex = actor_concat(ctx, reflex, prior0);
    reflex = actor_stem(ctx, reflex, "reflex_skip");
    int pooled = actor_pool(ctx, x, TB_NET_SENSOR_TOKENS);
    int sum = actor_concat(ctx, reflex, pooled);
    if (sum < 0) return net_error(ctx, "actor summary failed");
    ctx->actor_summary_node = sum;
    if (!tb_gpu_copy(ctx->ops, summary, ctx->actor_nodes[sum].v, (size_t)batch * 2 * TB_NET_D_MODEL))
        return net_error(ctx, "actor summary copy failed");
    if (aux) {
        int a = actor_project(ctx, sum, "aux_perception_head.0",
                              "aux_perception_head.1", TB_NET_D_MODEL);
        a = actor_linear(ctx, a, "aux_perception_head.3", 9);
        if (a < 0) return net_error(ctx, "actor auxiliary head failed");
        ctx->actor_aux_node = a;
        if (!tb_gpu_copy(ctx->ops, aux, ctx->actor_nodes[a].v, (size_t)batch * 9)) return 0;
    }
    if (world) {
        int w = actor_project(ctx, sum, "world_model_energy_head.net.0",
                              "world_model_energy_head.net.1", TB_NET_WORLD_DIM);
        w = actor_project(ctx, w, "world_model_energy_head.net.3",
                          "world_model_energy_head.net.4", TB_NET_WORLD_DIM);
        w = actor_linear(ctx, w, "world_model_energy_head.net.6", TB_NET_D_MODEL + 16);
        if (w < 0) return net_error(ctx, "actor world head failed");
        ctx->actor_world_node = w;
        if (!tb_gpu_copy(ctx->ops, world, ctx->actor_nodes[w].v,
                         (size_t)batch * (TB_NET_D_MODEL + 16))) return 0;
    }
    int query = actor_query(ctx, batch);
    query = actor_norm(ctx, query, "action_expert.q_norm");
    int q_proj = actor_linear(ctx, query, "action_expert.q_proj", TB_NET_D_MODEL);
    int kv_norm = actor_norm(ctx, x, "action_expert.kv_norm");
    int kv_proj = actor_linear(ctx, kv_norm, "action_expert.kv_proj", 2 * TB_NET_D_MODEL);
    int attention = actor_attention(ctx, q_proj, kv_proj, batch,
                                    TB_NET_ACTION_HORIZON, TB_NET_SENSOR_TOKENS, 0);
    attention = actor_linear(ctx, attention, "action_expert.out_proj", TB_NET_D_MODEL);
    int expert = actor_add(ctx, query, attention);
    int ff = actor_norm(ctx, expert, "action_expert.ffn_norm");
    ff = actor_linear(ctx, ff, "action_expert.ffn.0", 2 * TB_NET_D_MODEL);
    ff = actor_silu(ctx, ff);
    ff = actor_linear(ctx, ff, "action_expert.ffn.2", TB_NET_D_MODEL);
    expert = actor_add(ctx, expert, ff);
    expert = actor_broadcast_add(ctx, expert, reflex, TB_NET_ACTION_HORIZON);
    int raw = actor_linear(ctx, expert, "mu_head", TB_NET_ACTION_DIM);
    raw = actor_pick(ctx, raw, TB_NET_ACTION_HORIZON, 0);
    int ls = actor_linear(ctx, expert, "log_std_head", TB_NET_ACTION_DIM);
    ls = actor_pick(ctx, ls, TB_NET_ACTION_HORIZON, 0);
    ls = actor_clamp(ctx, ls, -5.0f, 2.0f);
    int act = actor_action(ctx, raw, prior0, batch);
    if (act < 0 || ls < 0) return net_error(ctx, "actor action head failed");
    ctx->actor_action_node = act;
    ctx->actor_logstd_node = ls;
    if (!tb_gpu_copy(ctx->ops, action, ctx->actor_nodes[act].v,
                     (size_t)batch * TB_NET_ACTION_DIM) ||
        !tb_gpu_copy(ctx->ops, log_std, ctx->actor_nodes[ls].v,
                     (size_t)batch * TB_NET_ACTION_DIM))
        return net_error(ctx, "actor output copy failed");
    return 1;
}
int tb_net_gpu_actor_backward(tb_net_gpu *ctx, const float *d_action,
                              const float *d_log_std, const float *d_summary,
                              const float *d_aux, const float *d_world) {
    if (!ctx || ctx->actor_batch < 1 || !ctx->gradients) return 0;
    int batch = ctx->actor_batch;
    if ((d_action && !tb_gpu_axpy(ctx->ops, ctx->actor_nodes[ctx->actor_action_node].g,
                                  d_action, 1.0f, (size_t)batch * TB_NET_ACTION_DIM)) ||
        (d_log_std && !tb_gpu_axpy(ctx->ops, ctx->actor_nodes[ctx->actor_logstd_node].g,
                                   d_log_std, 1.0f, (size_t)batch * TB_NET_ACTION_DIM)) ||
        (d_summary && !tb_gpu_axpy(ctx->ops, ctx->actor_nodes[ctx->actor_summary_node].g,
                                   d_summary, 1.0f, (size_t)batch * 2 * TB_NET_D_MODEL)) ||
        (d_aux && ctx->actor_aux_node >= 0 &&
         !tb_gpu_axpy(ctx->ops, ctx->actor_nodes[ctx->actor_aux_node].g,
                      d_aux, 1.0f, (size_t)batch * 9)) ||
        (d_world && ctx->actor_world_node >= 0 &&
         !tb_gpu_axpy(ctx->ops, ctx->actor_nodes[ctx->actor_world_node].g,
                      d_world, 1.0f, (size_t)batch * (TB_NET_D_MODEL + 16))))
        return net_error(ctx, "actor output gradient failed");
    for (int i = ctx->actor_node_count - 1; i >= 0; --i)
        if (!actor_backward_node(ctx, i)) return net_error(ctx, "actor backward failed");
    return 1;
}
