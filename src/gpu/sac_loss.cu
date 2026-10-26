#include "turbinator/gpu/sac_loss.h"

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>

#define TB_ATOMS 64
#define TB_ACTIONS 15
#define TB_WORLD 1552
#define TB_LATENT 1536
#define TB_OBS 4148
#define TB_PRIV 64

static cudaStream_t tb_stream(tb_gpu_ops *ops) { return (cudaStream_t)tb_gpu_stream(ops); }
static int tb_prepare(tb_gpu_ops *ops) {
    return ops && cudaSetDevice(tb_gpu_device(ops)) == cudaSuccess;
}

extern "C" void tb_gpu_sac_loss_config_default(tb_gpu_sac_loss_config *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    out->gamma = 0.992f;
    out->alpha = 0.05f;
    out->value_min = -45.0f;
    out->value_max = 65.0f;
    out->gaussian_sigma_ratio = 0.75f;
    out->q_weight = 0.05f;
    out->q_scale = 2.0f;
    out->reference_weight = 0.5f;
    out->aux_weight = 0.3f;
    out->world_weight = 0.15f;
}

__device__ static float tb_expected_q(const float *logits, float vmin, float step) {
    float maxlogit = -INFINITY;
    for (int j = 0; j < TB_ATOMS; ++j) maxlogit = fmaxf(maxlogit, logits[j]);
    float denom = 0.0f, numer = 0.0f;
    for (int j = 0; j < TB_ATOMS; ++j) {
        float e = expf(logits[j] - maxlogit);
        denom += e;
        numer += e * (vmin + j * step);
    }
    return numer / fmaxf(denom, 1.0e-30f);
}
__device__ static float tb_entropy(const float *action, const float *logstd) {
    float h = 0.0f;
    for (int i = 0; i < 14; ++i) h += logstd[i] + 1.4189385332046727f;
    float p = fminf(0.99999f, fmaxf(0.00001f, 0.5f * (action[14] + 1.0f)));
    h -= p * logf(p) + (1.0f - p) * logf(1.0f - p);
    return h;
}

__global__ static void tb_target_kernel(tb_gpu_sac_loss_config cfg,
                                        const float *l1, const float *l2,
                                        const float *action, const float *logstd,
                                        const float *reward, const float *done,
                                        float *target, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    float atom_step = (cfg.value_max - cfg.value_min) / (TB_ATOMS - 1);
    float q1 = tb_expected_q(l1 + (size_t)b * TB_ATOMS, cfg.value_min, atom_step);
    float q2 = tb_expected_q(l2 + (size_t)b * TB_ATOMS, cfg.value_min, atom_step);
    float entropy = tb_entropy(action + (size_t)b * TB_ACTIONS, logstd + (size_t)b * TB_ACTIONS);
    float backup = reward[b] + (1.0f - done[b]) * cfg.gamma * (fminf(q1, q2) + cfg.alpha * entropy);
    backup = fmaxf(cfg.value_min, fminf(cfg.value_max, backup));
    float sigma = fmaxf(atom_step * cfg.gaussian_sigma_ratio,
                        atom_step * cfg.gaussian_sigma_ratio + 0.1f * fabsf(q1 - q2));
    float denominator = 0.0f;
    for (int j = 0; j < TB_ATOMS; ++j) {
        float z = cfg.value_min + j * atom_step;
        float d = (z - backup) / sigma;
        float value = expf(-0.5f * d * d);
        target[(size_t)b * TB_ATOMS + j] = value;
        denominator += value;
    }
    float inv = 1.0f / fmaxf(denominator, 1.0e-30f);
    for (int j = 0; j < TB_ATOMS; ++j) target[(size_t)b * TB_ATOMS + j] *= inv;
}
extern "C" int tb_gpu_sac_build_targets(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg, const float *q1, const float *q2,
    const float *action, const float *logstd, const float *rewards,
    const float *dones, int batch, float *target) {
    if (!tb_prepare(ops) || !cfg || !q1 || !q2 || !action || !logstd || !rewards ||
        !dones || !target || batch <= 0) return 0;
    tb_target_kernel<<<(batch + 127) / 128, 128, 0, tb_stream(ops)>>>(
        *cfg, q1, q2, action, logstd, rewards, dones, target, batch);
    return cudaGetLastError() == cudaSuccess;
}

__global__ static void tb_critic_grad_kernel(tb_gpu_sac_loss_config cfg,
    const float *l1, const float *l2, const float *target, float *dl1, float *dl2,
    float *loss, float *qmean, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float *logits[2] = {l1 + (size_t)b * TB_ATOMS, l2 + (size_t)b * TB_ATOMS};
    float *grads[2] = {dl1 + (size_t)b * TB_ATOMS, dl2 + (size_t)b * TB_ATOMS};
    float local_loss = 0.0f;
    float qs[2] = {0.0f, 0.0f};
    float atom_step = (cfg.value_max - cfg.value_min) / (TB_ATOMS - 1);
    for (int critic = 0; critic < 2; ++critic) {
        float maxlogit = -INFINITY;
        for (int j = 0; j < TB_ATOMS; ++j) maxlogit = fmaxf(maxlogit, logits[critic][j]);
        float denominator = 0.0f;
        for (int j = 0; j < TB_ATOMS; ++j) denominator += expf(logits[critic][j] - maxlogit);
        float logdenom = maxlogit + logf(fmaxf(denominator, 1.0e-30f));
        for (int j = 0; j < TB_ATOMS; ++j) {
            float probability = expf(logits[critic][j] - logdenom);
            float target_p = target[(size_t)b * TB_ATOMS + j];
            grads[critic][j] = (probability - target_p) / batch;
            local_loss -= target_p * (logits[critic][j] - logdenom) / batch;
            qs[critic] += probability * (cfg.value_min + j * atom_step);
        }
    }
    if (loss) atomicAdd(loss, local_loss);
    if (qmean) atomicAdd(qmean, 0.5f * (qs[0] + qs[1]) / batch);
}
extern "C" int tb_gpu_sac_critic_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg, const float *q1, const float *q2,
    const float *target, int batch, float *dq1, float *dq2,
    float *loss, float *qmean) {
    if (!tb_prepare(ops) || !cfg || !q1 || !q2 || !target || !dq1 || !dq2 || batch <= 0) return 0;
    tb_critic_grad_kernel<<<(batch + 127) / 128, 128, 0, tb_stream(ops)>>>(
        *cfg, q1, q2, target, dq1, dq2, loss, qmean, batch);
    return cudaGetLastError() == cudaSuccess;
}

__global__ static void tb_policy_grad_kernel(tb_gpu_sac_loss_config cfg,
    const float *l1, const float *l2, const float *action, const float *logstd,
    const float *reference, float *dl1, float *dl2, float *da, float *dlogstd,
    float *loss, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    float atom_step = (cfg.value_max - cfg.value_min) / (TB_ATOMS - 1);
    float q1 = tb_expected_q(l1 + (size_t)b * TB_ATOMS, cfg.value_min, atom_step);
    float q2 = tb_expected_q(l2 + (size_t)b * TB_ATOMS, cfg.value_min, atom_step);
    int chosen = q1 <= q2 ? 0 : 1;
    float q = chosen ? q2 : q1;
    const float *logits = chosen ? l2 + (size_t)b * TB_ATOMS : l1 + (size_t)b * TB_ATOMS;
    float maxlogit = -INFINITY;
    for (int j = 0; j < TB_ATOMS; ++j) maxlogit = fmaxf(maxlogit, logits[j]);
    float denominator = 0.0f;
    for (int j = 0; j < TB_ATOMS; ++j) denominator += expf(logits[j] - maxlogit);
    for (int j = 0; j < TB_ATOMS; ++j) {
        float probability = expf(logits[j] - maxlogit) / fmaxf(denominator, 1.0e-30f);
        float gradient = -cfg.q_weight * probability * ((cfg.value_min + j * atom_step) - q) /
                         (batch * fmaxf(cfg.q_scale, 1.0f));
        dl1[(size_t)b * TB_ATOMS + j] = chosen ? 0.0f : gradient;
        dl2[(size_t)b * TB_ATOMS + j] = chosen ? gradient : 0.0f;
    }
    float bc = 0.0f;
    for (int k = 0; k < TB_ACTIONS; ++k) {
        size_t i = (size_t)b * TB_ACTIONS + k;
        float diff = action[i] - reference[i];
        bc += diff * diff / TB_ACTIONS;
        da[i] = 2.0f * cfg.reference_weight * diff / (batch * TB_ACTIONS);
        dlogstd[i] = k < 14 ? -cfg.q_weight * cfg.alpha /
            (batch * fmaxf(cfg.q_scale, 1.0f)) : 0.0f;
    }
    float p = fminf(0.99999f, fmaxf(0.00001f, 0.5f * (action[(size_t)b * TB_ACTIONS + 14] + 1.0f)));
    da[(size_t)b * TB_ACTIONS + 14] += cfg.q_weight * cfg.alpha *
        0.5f * logf(p / (1.0f - p)) / (batch * fmaxf(cfg.q_scale, 1.0f));
    if (loss) {
        float entropy = tb_entropy(action + (size_t)b * TB_ACTIONS, logstd + (size_t)b * TB_ACTIONS);
        atomicAdd(loss, (cfg.q_weight * (-q - cfg.alpha * entropy) /
                        fmaxf(cfg.q_scale, 1.0f) + cfg.reference_weight * bc) / batch);
    }
}
extern "C" int tb_gpu_sac_policy_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg, const float *q1, const float *q2,
    const float *action, const float *logstd, const float *reference,
    int batch, float *dq1, float *dq2, float *da, float *dlogstd, float *loss) {
    if (!tb_prepare(ops) || !cfg || !q1 || !q2 || !action || !logstd || !reference ||
        !dq1 || !dq2 || !da || !dlogstd || batch <= 0) return 0;
    tb_policy_grad_kernel<<<(batch + 127) / 128, 128, 0, tb_stream(ops)>>>(
        *cfg, q1, q2, action, logstd, reference, dq1, dq2, da, dlogstd, loss, batch);
    return cudaGetLastError() == cudaSuccess;
}

__device__ static float tb_aux_target(const float *priv, int k) {
    int arena = fabsf(priv[35]) > 1.0e-4f;
    int projectile = priv[41] * priv[41] + priv[42] * priv[42] + priv[43] * priv[43] > 1.0e-8f;
    if (k < 3) return priv[(arena ? 29 : 13) + k];
    if (k < 6) return priv[16 + k - 3];
    return priv[(projectile ? 41 : 53) + k - 6];
}
__device__ static float tb_huber(float diff) {
    float a = fabsf(diff);
    return a <= 1.0f ? 0.5f * diff * diff : a - 0.5f;
}
__device__ static float tb_huber_grad(float diff) { return fmaxf(-1.0f, fminf(1.0f, diff)); }

__global__ static void tb_aux_kernel(tb_gpu_sac_loss_config cfg, const float *priv,
    const float *obs, const float *next_obs, const float *next_summary,
    const float *reward, const float *done, const float *aux, const float *world,
    float *daux, float *dworld, float *loss, int batch) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)batch * TB_WORLD) return;
    int b = (int)(i / TB_WORLD), k = (int)(i % TB_WORLD);
    const float *p = priv + (size_t)b * TB_PRIV;
    if (k < 9) {
        float diff = aux[(size_t)b * 9 + k] - tb_aux_target(p, k);
        daux[(size_t)b * 9 + k] = cfg.aux_weight * tb_huber_grad(diff) / (batch * 9);
        if (loss) atomicAdd(loss, cfg.aux_weight * tb_huber(diff) / (batch * 9));
    }
    float target;
    if (k < TB_LATENT) target = next_summary[(size_t)b * 3072 + k];
    else {
        int j = k - TB_LATENT;
        const float *o = obs + (size_t)b * TB_OBS;
        const float *no = next_obs + (size_t)b * TB_OBS;
        if (j < 3) target = (no[3 + j] - o[3 + j]) * 0.25f;
        else if (j < 6) target = (no[10 + j - 3] - o[10 + j - 3]) * 0.5f;
        else if (j < 10) target = o[16 + j - 6];
        else if (j < 12) target = o[40 + j - 10];
        else if (j < 15) target = tb_aux_target(p, j - 12);
        else target = reward[b];
    }
    float mask = 1.0f - done[b];
    float diff = world[i] - target;
    dworld[i] = cfg.world_weight * mask * tb_huber_grad(diff) / (batch * TB_WORLD);
    if (loss && mask > 0) atomicAdd(loss, cfg.world_weight * tb_huber(diff) / (batch * TB_WORLD));
}
extern "C" int tb_gpu_sac_aux_grad(tb_gpu_ops *ops,
    const tb_gpu_sac_loss_config *cfg, const float *priv, const float *obs,
    const float *next_obs, const float *next_summary, const float *rewards,
    const float *dones, const float *aux, const float *world, int batch,
    float *daux, float *dworld, float *loss) {
    if (!tb_prepare(ops) || !cfg || !priv || !obs || !next_obs || !next_summary ||
        !rewards || !dones || !aux || !world || !daux || !dworld || batch <= 0) return 0;
    size_t count = (size_t)batch * TB_WORLD;
    tb_aux_kernel<<<(unsigned)((count + 255) / 256), 256, 0, tb_stream(ops)>>>(
        *cfg, priv, obs, next_obs, next_summary, rewards, dones,
        aux, world, daux, dworld, loss, batch);
    return cudaGetLastError() == cudaSuccess;
}

__global__ static void tb_entropy_mean_kernel(const float *action, const float *logstd,
                                                float *result, int batch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) atomicAdd(result, tb_entropy(action + (size_t)b * TB_ACTIONS,
                                              logstd + (size_t)b * TB_ACTIONS) / batch);
}
extern "C" int tb_gpu_sac_mean_entropy(tb_gpu_ops *ops, const float *action,
                                        const float *logstd, int batch, float *result) {
    if (!tb_prepare(ops) || !action || !logstd || !result || batch <= 0 ||
        !tb_gpu_zero(ops, result, 1)) return 0;
    tb_entropy_mean_kernel<<<(batch + 127) / 128, 128, 0, tb_stream(ops)>>>(
        action, logstd, result, batch);
    return cudaGetLastError() == cudaSuccess;
}
