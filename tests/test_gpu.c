#include "turbinator/gpu/ops.h"
#include "turbinator/gpu/optimizer.h"
#include "turbinator/gpu/replay.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(expr) do { if (!(expr)) { \
    fprintf(stderr, "GPU test failed at line %d: %s\n", __LINE__, #expr); \
    return 0; \
} } while (0)

static int near(float a, float b, float tolerance) {
    return isfinite(a) && isfinite(b) && fabsf(a - b) <= tolerance;
}

static int test_linear(tb_gpu_ops *ops) {
    const float x[6] = {1, 2, 3, 4, 5, 6};
    const float w[6] = {1, -1, 2, 0.5f, 2, -0.5f};
    const float bias[2] = {0.5f, -1};
    const float dy[4] = {1, 2, 3, 4};
    float result[4], dx[6], dw[6], db[2];
    float *d_x = tb_gpu_malloc(ops, sizeof(x));
    float *d_w = tb_gpu_malloc(ops, sizeof(w));
    float *d_bias = tb_gpu_malloc(ops, sizeof(bias));
    float *d_dy = tb_gpu_malloc(ops, sizeof(dy));
    float *d_y = tb_gpu_malloc(ops, sizeof(result));
    float *d_dx = tb_gpu_malloc(ops, sizeof(dx));
    float *d_dw = tb_gpu_malloc(ops, sizeof(dw));
    float *d_db = tb_gpu_malloc(ops, sizeof(db));
    REQUIRE(d_x && d_w && d_bias && d_dy && d_y && d_dx && d_dw && d_db);
    REQUIRE(tb_gpu_h2d(ops, d_x, x, sizeof(x)));
    REQUIRE(tb_gpu_h2d(ops, d_w, w, sizeof(w)));
    REQUIRE(tb_gpu_h2d(ops, d_bias, bias, sizeof(bias)));
    REQUIRE(tb_gpu_h2d(ops, d_dy, dy, sizeof(dy)));
    REQUIRE(tb_gpu_linear_fwd(ops, d_x, d_w, d_bias, d_y, 2, 3, 2));
    REQUIRE(tb_gpu_linear_bwd(ops, d_x, d_w, d_dy, d_dx, d_dw, d_db,
                              2, 3, 2, 0));
    REQUIRE(tb_gpu_d2h(ops, result, d_y, sizeof(result)));
    REQUIRE(tb_gpu_d2h(ops, dx, d_dx, sizeof(dx)));
    REQUIRE(tb_gpu_d2h(ops, dw, d_dw, sizeof(dw)));
    REQUIRE(tb_gpu_d2h(ops, db, d_db, sizeof(db)));
    const float expected_y[4] = {5.5f, 2, 11.5f, 8};
    const float expected_dx[6] = {2, 3, 1, 5, 5, 4};
    const float expected_dw[6] = {13, 17, 21, 18, 24, 30};
    for (int i = 0; i < 4; ++i) REQUIRE(near(result[i], expected_y[i], 1e-4f));
    for (int i = 0; i < 6; ++i) {
        REQUIRE(near(dx[i], expected_dx[i], 1e-4f));
        REQUIRE(near(dw[i], expected_dw[i], 1e-4f));
    }
    REQUIRE(near(db[0], 4, 1e-4f) && near(db[1], 6, 1e-4f));
    tb_gpu_free(ops, d_x); tb_gpu_free(ops, d_w); tb_gpu_free(ops, d_bias);
    tb_gpu_free(ops, d_dy); tb_gpu_free(ops, d_y); tb_gpu_free(ops, d_dx);
    tb_gpu_free(ops, d_dw); tb_gpu_free(ops, d_db);
    return 1;
}

static int test_normalization(tb_gpu_ops *ops) {
    const float x[8] = {-2, 0, 1, 3, 4, -1, 2, 0.5f};
    const float gamma[4] = {1, 1.5f, -0.5f, 2};
    const float dy[8] = {1, -2, 0.5f, 0.25f, 0.75f, 1, -1, 2};
    float dx[8];
    float *d_x = tb_gpu_malloc(ops, sizeof(x));
    float *d_gamma = tb_gpu_malloc(ops, sizeof(gamma));
    float *d_dy = tb_gpu_malloc(ops, sizeof(dy));
    float *d_y = tb_gpu_malloc(ops, sizeof(x));
    float *d_mean = tb_gpu_malloc(ops, 2 * sizeof(float));
    float *d_inv = tb_gpu_malloc(ops, 2 * sizeof(float));
    float *d_dx = tb_gpu_malloc(ops, sizeof(dx));
    REQUIRE(d_x && d_gamma && d_dy && d_y && d_mean && d_inv && d_dx);
    REQUIRE(tb_gpu_h2d(ops, d_x, x, sizeof(x)));
    REQUIRE(tb_gpu_h2d(ops, d_gamma, gamma, sizeof(gamma)));
    REQUIRE(tb_gpu_h2d(ops, d_dy, dy, sizeof(dy)));
    REQUIRE(tb_gpu_layernorm_fwd(ops, d_x, d_gamma, NULL, d_y,
                                 d_mean, d_inv, 2, 4, 1e-5f));
    REQUIRE(tb_gpu_layernorm_bwd(ops, d_x, d_gamma, d_dy, d_mean, d_inv,
                                 d_dx, NULL, NULL, 2, 4, 0));
    REQUIRE(tb_gpu_d2h(ops, dx, d_dx, sizeof(dx)));
    const float epsilon = 1e-3f;
    for (int index = 0; index < 8; ++index) {
        float perturbed[8], y[8], plus = 0, minus = 0;
        memcpy(perturbed, x, sizeof(x));
        perturbed[index] += epsilon;
        REQUIRE(tb_gpu_h2d(ops, d_x, perturbed, sizeof(x)));
        REQUIRE(tb_gpu_layernorm_fwd(ops, d_x, d_gamma, NULL, d_y,
                                     NULL, NULL, 2, 4, 1e-5f));
        REQUIRE(tb_gpu_d2h(ops, y, d_y, sizeof(y)));
        for (int j = 0; j < 8; ++j) plus += y[j] * dy[j];
        perturbed[index] -= 2 * epsilon;
        REQUIRE(tb_gpu_h2d(ops, d_x, perturbed, sizeof(x)));
        REQUIRE(tb_gpu_layernorm_fwd(ops, d_x, d_gamma, NULL, d_y,
                                     NULL, NULL, 2, 4, 1e-5f));
        REQUIRE(tb_gpu_d2h(ops, y, d_y, sizeof(y)));
        for (int j = 0; j < 8; ++j) minus += y[j] * dy[j];
        REQUIRE(near(dx[index], (plus - minus) / (2 * epsilon), 5e-3f));
    }
    tb_gpu_free(ops, d_x); tb_gpu_free(ops, d_gamma); tb_gpu_free(ops, d_dy);
    tb_gpu_free(ops, d_y); tb_gpu_free(ops, d_mean); tb_gpu_free(ops, d_inv);
    tb_gpu_free(ops, d_dx);
    return 1;
}

static int test_optimizer(tb_gpu_ops *ops) {
    const float params[4] = {1, -2, 3, -4};
    const float grads[4] = {0.5f, -1, 1.5f, -2};
    float updated[4], target[4];
    float *d_params = tb_gpu_malloc(ops, sizeof(params));
    float *d_grads = tb_gpu_malloc(ops, sizeof(grads));
    float *d_target = tb_gpu_malloc(ops, sizeof(params));
    tb_gpu_rmsprop *optimizer = tb_gpu_rmsprop_create(ops, 4, 0.01f, 0.99f,
                                                       1e-8f, 0, 1.0f);
    REQUIRE(d_params && d_grads && d_target && optimizer);
    REQUIRE(tb_gpu_h2d(ops, d_params, params, sizeof(params)));
    REQUIRE(tb_gpu_h2d(ops, d_grads, grads, sizeof(grads)));
    REQUIRE(tb_gpu_h2d(ops, d_target, params, sizeof(params)));
    float norm = 0;
    REQUIRE(tb_gpu_rmsprop_step(optimizer, d_params, d_grads, &norm));
    REQUIRE(near(norm, sqrtf(7.5f), 1e-4f));
    REQUIRE(tb_gpu_rmsprop_steps(optimizer) == 1);
    REQUIRE(tb_gpu_ema_update(ops, d_target, d_params, 4, 0.5f));
    REQUIRE(tb_gpu_d2h(ops, updated, d_params, sizeof(updated)));
    REQUIRE(tb_gpu_d2h(ops, target, d_target, sizeof(target)));
    for (int i = 0; i < 4; ++i) {
        REQUIRE(isfinite(updated[i]) && fabsf(updated[i] - params[i]) > 0.001f);
        REQUIRE(near(target[i], (updated[i] + params[i]) * 0.5f, 1e-5f));
    }
    tb_gpu_rmsprop_destroy(optimizer);
    tb_gpu_free(ops, d_params); tb_gpu_free(ops, d_grads);
    tb_gpu_free(ops, d_target);
    return 1;
}

static int test_replay(tb_gpu_ops *ops) {
    const size_t transitions = 3;
    float *obs = (float *)calloc(transitions * TB_TRAIN_OBSERVATION_DIM, sizeof(float));
    float *mission = (float *)calloc(transitions * TB_TRAIN_MISSION_DIM, sizeof(float));
    uint8_t *camera = (uint8_t *)calloc(transitions * TB_TRAIN_CAMERA_PIXELS, 1);
    float *priv = (float *)calloc(transitions * TB_TRAIN_PRIVILEGED_DIM, sizeof(float));
    float *actions = (float *)calloc(transitions * TB_TRAIN_ACTION_DIM, sizeof(float));
    float reward[3] = {10, 20, 30}, done[3] = {0, 1, 0};
    REQUIRE(obs && mission && camera && priv && actions);
    for (int i = 0; i < 3; ++i) {
        obs[(size_t)i * TB_TRAIN_OBSERVATION_DIM] = (float)(i + 1);
        mission[(size_t)i * TB_TRAIN_MISSION_DIM] = (float)(i + 11);
        camera[(size_t)i * TB_TRAIN_CAMERA_PIXELS] = (uint8_t)(i + 21);
        priv[(size_t)i * TB_TRAIN_PRIVILEGED_DIM] = (float)(i + 31);
        actions[(size_t)i * TB_TRAIN_ACTION_DIM] = (float)(i + 41);
    }
    tb_gpu_host_batch host = {obs, mission, camera, priv, actions, actions,
                              reward, obs, mission, camera, priv, done};
    tb_gpu_replay *replay = tb_gpu_replay_create(ops, 2, 7);
    tb_gpu_sample_batch *sample = tb_gpu_sample_batch_create(ops, 8);
    REQUIRE(replay && sample);
    REQUIRE(tb_gpu_replay_add(replay, &host, transitions));
    REQUIRE(tb_gpu_replay_size(replay) == 2);
    REQUIRE(tb_gpu_replay_sample(replay, sample));
    float sampled_reward[8], sampled_done[8];
    float sampled_mission[8 * TB_TRAIN_MISSION_DIM];
    uint8_t sampled_camera[8 * TB_TRAIN_CAMERA_PIXELS];
    /* Read strided fields through a complete row to check replay alignment. */
    float *full_obs = (float *)malloc(8 * TB_TRAIN_OBSERVATION_DIM * sizeof(float));
    REQUIRE(full_obs);
    REQUIRE(tb_gpu_d2h(ops, full_obs, sample->obs,
                       8 * TB_TRAIN_OBSERVATION_DIM * sizeof(float)));
    REQUIRE(tb_gpu_d2h(ops, sampled_reward, sample->rewards, sizeof(sampled_reward)));
    REQUIRE(tb_gpu_d2h(ops, sampled_done, sample->dones, sizeof(sampled_done)));
    REQUIRE(tb_gpu_d2h(ops, sampled_mission, sample->mission, sizeof(sampled_mission)));
    REQUIRE(tb_gpu_d2h(ops, sampled_camera, sample->cameras, sizeof(sampled_camera)));
    for (int i = 0; i < 8; ++i) {
        float id = full_obs[(size_t)i * TB_TRAIN_OBSERVATION_DIM];
        REQUIRE(id == 2 || id == 3);
        REQUIRE(near(sampled_reward[i], id * 10, 0));
        REQUIRE(near(sampled_done[i], id == 2 ? 1 : 0, 0));
        REQUIRE(near(sampled_mission[(size_t)i * TB_TRAIN_MISSION_DIM], id + 10, 0));
        REQUIRE(sampled_camera[(size_t)i * TB_TRAIN_CAMERA_PIXELS] == (uint8_t)(id + 20));
    }
    free(full_obs);
    tb_gpu_sample_batch_destroy(ops, sample);
    tb_gpu_replay_destroy(replay);
    free(obs); free(mission); free(camera); free(priv); free(actions);
    return 1;
}

int main(void) {
    if (tb_gpu_device_count() == 0) {
        puts("GPU tests skipped: no CUDA device");
        return 0;
    }
    tb_gpu_ops *ops = tb_gpu_ops_create(0);
    if (!ops) { fputs("Failed to create CUDA context\n", stderr); return 1; }
    int ok = test_linear(ops) && test_normalization(ops) &&
             test_optimizer(ops) && test_replay(ops);
    if (!ok) fprintf(stderr, "CUDA error: %s\n", tb_gpu_last_error(ops));
    tb_gpu_ops_destroy(ops);
    if (ok) puts("GPU primitive tests passed");
    return ok ? 0 : 1;
}
