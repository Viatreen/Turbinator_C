#include "turbinator/network/network.h"
#include "turbinator/network/network_gpu.h"
#include "turbinator/gpu/ops.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expr) do { if (!(expr)) { \
    fprintf(stderr, "GPU network smoke failed at %s:%d: %s\n", __FILE__, __LINE__, #expr); \
    goto cleanup; \
} } while (0)

static float *device_array(tb_gpu_ops *ops, size_t count) {
    return (float *)tb_gpu_malloc(ops, count * sizeof(float));
}

static int valid_values(const float *x, size_t count) {
    for (size_t i = 0; i < count; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}

static int gradient_norm(tb_gpu_ops *ops, const tb_net_model *layout,
                         const float *gradients, const char *name, float *out) {
    const tb_net_tensor_info *info = tb_net_find_tensor(layout, name);
    return info && tb_gpu_l2_norm(ops, gradients + info->offset,
                                   (size_t)info->count, out) && isfinite(*out);
}

int main(void) {
    int ok = 0;
    tb_net_model *layout = NULL;
    tb_gpu_ops *ops = NULL;
    tb_net_gpu *network = NULL;
    float *parameters = NULL, *gradients = NULL, *host_tile = NULL;
    float *obs = NULL, *mission = NULL, *action = NULL, *log_std = NULL;
    float *summary = NULL, *aux = NULL, *world = NULL, *privileged = NULL;
    float *logits1 = NULL, *logits2 = NULL, *dlogits1 = NULL, *dlogits2 = NULL;
    float *daction = NULL, *dlog_std = NULL, *dsummary = NULL;
    float *daux = NULL, *dworld = NULL;
    uint8_t *cameras = NULL;
    float host_obs[4148] = {0};
    float host_mission[10] = {0};
    uint8_t host_cameras[4 * 64 * 64];
    float host_privileged[64] = {0};
    float host_action[15], host_log_std[15], host_summary[3072];
    float host_aux[9], host_world[1552], host_q1[64], host_q2[64];
    float host_dlogits[64], host_ones_action[15], host_ones_logstd[15];
    float host_daux[9], host_dworld[1552];
    const size_t tile = (size_t)1 << 20;
    layout = tb_net_create_618m();
    CHECK(layout && tb_net_parameter_count(layout) == TB_NET_TOTAL_PARAMETERS);
    CHECK(tb_gpu_device_count() > 0);
    ops = tb_gpu_ops_create(0);
    CHECK(ops);
    parameters = device_array(ops, (size_t)tb_net_parameter_count(layout));
    gradients = device_array(ops, (size_t)tb_net_parameter_count(layout));
    host_tile = (float *)malloc(tile * sizeof(float));
    CHECK(parameters && gradients && host_tile);
    fprintf(stderr, "Initializing 617,973,042 network parameters on GPU 0...\n");
    for (uint64_t offset = 0; offset < tb_net_parameter_count(layout); offset += tile) {
        size_t n = tb_net_parameter_count(layout) - offset < tile ?
                   (size_t)(tb_net_parameter_count(layout) - offset) : tile;
        CHECK(tb_net_initialize_slice(layout, offset, host_tile, n, UINT64_C(6061)));
        CHECK(tb_gpu_h2d(ops, parameters + offset, host_tile, n * sizeof(float)));
    }
    CHECK(tb_gpu_zero(ops, gradients, (size_t)tb_net_parameter_count(layout)));
    network = tb_net_gpu_create(layout, ops, parameters, gradients, 1);
    CHECK(network);
    obs = device_array(ops, 4148);
    mission = device_array(ops, 10);
    cameras = (uint8_t *)tb_gpu_malloc(ops, sizeof(host_cameras));
    action = device_array(ops, 15);
    log_std = device_array(ops, 15);
    summary = device_array(ops, 3072);
    aux = device_array(ops, 9);
    world = device_array(ops, 1552);
    privileged = device_array(ops, 64);
    logits1 = device_array(ops, 64);
    logits2 = device_array(ops, 64);
    dlogits1 = device_array(ops, 64);
    dlogits2 = device_array(ops, 64);
    daction = device_array(ops, 15);
    dlog_std = device_array(ops, 15);
    dsummary = device_array(ops, 3072);
    daux = device_array(ops, 9);
    dworld = device_array(ops, 1552);
    CHECK(obs && mission && cameras && action && log_std && summary && aux && world &&
          privileged && logits1 && logits2 && dlogits1 && dlogits2 && daction &&
          dlog_std && dsummary && daux && dworld);
    for (int i = 0; i < 48; ++i) host_obs[i] = (float)((i % 9) - 4) * 0.08f;
    for (int i = 48; i < 4144; ++i) host_obs[i] = 0.2f + (float)(i % 137) * 0.1f;
    for (int i = 4144; i < 4148; ++i) host_obs[i] = 0.5f;
    for (int i = 0; i < 10; ++i) host_mission[i] = (float)(i + 1) * 0.03f;
    for (size_t i = 0; i < sizeof(host_cameras); ++i)
        host_cameras[i] = (uint8_t)((i * 13 + i / 64) % 256);
    for (int i = 0; i < 64; ++i) {
        host_privileged[i] = (float)(i % 11) * 0.05f;
        host_dlogits[i] = (float)((i % 7) - 3) * 0.02f;
    }
    for (int i = 0; i < 15; ++i) {
        host_ones_action[i] = 0.1f;
        host_ones_logstd[i] = 0.01f;
    }
    for (int i = 0; i < 9; ++i) host_daux[i] = 0.01f;
    for (int i = 0; i < 1552; ++i) host_dworld[i] = 0.001f;
    CHECK(tb_gpu_h2d(ops, obs, host_obs, sizeof(host_obs)));
    CHECK(tb_gpu_h2d(ops, mission, host_mission, sizeof(host_mission)));
    CHECK(tb_gpu_h2d(ops, cameras, host_cameras, sizeof(host_cameras)));
    CHECK(tb_gpu_h2d(ops, privileged, host_privileged, sizeof(host_privileged)));
    CHECK(tb_gpu_h2d(ops, dlogits1, host_dlogits, sizeof(host_dlogits)));
    CHECK(tb_gpu_h2d(ops, dlogits2, host_dlogits, sizeof(host_dlogits)));
    CHECK(tb_gpu_h2d(ops, daction, host_ones_action, sizeof(host_ones_action)));
    CHECK(tb_gpu_h2d(ops, dlog_std, host_ones_logstd, sizeof(host_ones_logstd)));
    CHECK(tb_gpu_h2d(ops, daux, host_daux, sizeof(host_daux)));
    CHECK(tb_gpu_h2d(ops, dworld, host_dworld, sizeof(host_dworld)));
    CHECK(tb_net_gpu_actor_forward(network, obs, mission, cameras, 1,
                                   action, log_std, summary, aux, world));
    CHECK(tb_gpu_sync(ops));
    CHECK(tb_gpu_d2h(ops, host_action, action, sizeof(host_action)));
    CHECK(tb_gpu_d2h(ops, host_log_std, log_std, sizeof(host_log_std)));
    CHECK(tb_gpu_d2h(ops, host_summary, summary, sizeof(host_summary)));
    CHECK(tb_gpu_d2h(ops, host_aux, aux, sizeof(host_aux)));
    CHECK(tb_gpu_d2h(ops, host_world, world, sizeof(host_world)));
    CHECK(valid_values(host_action, 15) && valid_values(host_log_std, 15) &&
          valid_values(host_summary, 3072) && valid_values(host_aux, 9) &&
          valid_values(host_world, 1552));
    for (int i = 0; i < 15; ++i)
        CHECK(host_action[i] >= -1.0f && host_action[i] <= 1.0f &&
              host_log_std[i] >= -5.0f && host_log_std[i] <= 2.0f);
    CHECK(tb_net_gpu_critic_forward(network, summary, privileged, action, 1,
                                    logits1, logits2));
    CHECK(tb_gpu_sync(ops));
    CHECK(tb_gpu_d2h(ops, host_q1, logits1, sizeof(host_q1)));
    CHECK(tb_gpu_d2h(ops, host_q2, logits2, sizeof(host_q2)));
    CHECK(valid_values(host_q1, 64) && valid_values(host_q2, 64));
    CHECK(tb_net_gpu_critic_backward(network, dlogits1, dlogits2,
                                     dsummary, daction, 1));
    CHECK(tb_gpu_sync(ops));
    CHECK(tb_net_gpu_actor_backward(network, daction, dlog_std,
                                    dsummary, daux, dworld));
    CHECK(tb_gpu_sync(ops));
    const char *required[] = {
        "lidar_stem.conv1.weight", "camera_stem.conv.0.weight",
        "blocks.0.qkv.weight", "blocks.11.ffn.0.weight",
        "aero_energy_moe.0.router.2.weight", "mu_head.weight",
        "world_model_energy_head.net.0.weight", "aux_perception_head.0.weight",
        "critic.q1_net.0.weight", "critic.q2_net.18.weight"
    };
    for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        float norm = 0.0f;
        CHECK(gradient_norm(ops, layout, gradients, required[i], &norm));
        fprintf(stderr, "%s gradient L2 = %.7g\n", required[i], norm);
        CHECK(norm > 0.0f);
    }
    int active_experts = 0;
    for (int e = 0; e < 6; ++e) {
        char name[96];
        float norm = 0.0f;
        snprintf(name, sizeof(name), "aero_energy_moe.0.experts_down.%d.weight", e);
        CHECK(gradient_norm(ops, layout, gradients, name, &norm));
        if (norm > 0.0f) ++active_experts;
    }
    CHECK(active_experts >= 2);
    printf("Full 617,973,042 parameter GPU forward/backward smoke passed; active experts=%d\n",
           active_experts);
    ok = 1;
cleanup:
    if (!ok) {
        if (network) fprintf(stderr, "network: %s\n", tb_net_gpu_last_error(network));
        if (ops) fprintf(stderr, "GPU: %s\n", tb_gpu_last_error(ops));
    }
    tb_net_gpu_destroy(network);
    if (ops) {
        tb_gpu_free(ops, parameters); tb_gpu_free(ops, gradients);
        tb_gpu_free(ops, obs); tb_gpu_free(ops, mission); tb_gpu_free(ops, cameras);
        tb_gpu_free(ops, action); tb_gpu_free(ops, log_std); tb_gpu_free(ops, summary);
        tb_gpu_free(ops, aux); tb_gpu_free(ops, world); tb_gpu_free(ops, privileged);
        tb_gpu_free(ops, logits1); tb_gpu_free(ops, logits2);
        tb_gpu_free(ops, dlogits1); tb_gpu_free(ops, dlogits2);
        tb_gpu_free(ops, daction); tb_gpu_free(ops, dlog_std);
        tb_gpu_free(ops, dsummary); tb_gpu_free(ops, daux); tb_gpu_free(ops, dworld);
    }
    tb_gpu_ops_destroy(ops);
    tb_net_destroy(layout);
    free(host_tile);
    return ok ? 0 : 1;
}
