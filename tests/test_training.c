#include "turbinator/training/training.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    tb_train_config cfg;
    tb_train_config_default(&cfg);
    cfg.hidden_dim = 32;
    cfg.replay_capacity = 64;
    tb_sac_agent *agent = tb_sac_create(&cfg);
    assert(agent);

    float obs[TB_TRAIN_OBSERVATION_DIM] = {0};
    float next_obs[TB_TRAIN_OBSERVATION_DIM] = {0};
    float mission[TB_TRAIN_MISSION_DIM] = {0};
    float priv[TB_TRAIN_PRIVILEGED_DIM] = {0};
    float action[TB_TRAIN_ACTION_DIM] = {0};
    float reference[TB_TRAIN_ACTION_DIM] = {0};
    float physical[TB_TRAIN_ACTION_DIM] = {0};
    float round_trip[TB_TRAIN_ACTION_DIM] = {0};
    float features[TB_TRAIN_FEATURE_DIM] = {0};
    uint8_t cameras[TB_TRAIN_CAMERA_PIXELS];
    memset(cameras, 128, sizeof(cameras));
    mission[0] = 1.0f;
    obs[2] = 2.0f;
    next_obs[2] = 2.1f;
    tb_train_extract_features_cpu(obs, mission, cameras, features);
    assert(features[2] > 0.0f && features[126] > 0.0f);
    mission[6] = 1.0f;
    mission[7] = 0.9f;
    tb_train_extract_features_cpu(obs, mission, cameras, features);
    assert(features[59] == 0.0f); /* Opponent truth must not enter actor features. */
    mission[6] = 0.0f;

    for (int i = 0; i < 96; ++i) {
        obs[0] = (float)i * 0.01f;
        next_obs[0] = obs[0] + 0.01f;
        reference[0] = 0.3f;
        assert(tb_sac_act(agent, obs, mission, cameras, 0, action, NULL));
        for (int j = 0; j < TB_TRAIN_ACTION_DIM; ++j) assert(isfinite(action[j]) && fabsf(action[j]) <= 1.0f);
        assert(tb_sac_add_transition(agent, obs, mission, cameras, priv, action, reference,
                                     1.0f, next_obs, mission, cameras, priv, i % 17 == 0));
    }
    assert(tb_sac_replay_size(agent) == 64);

    tb_train_metrics metrics;
    for (int i = 0; i < 4; ++i) {
        assert(tb_sac_update(agent, 8, &metrics));
        assert(isfinite(metrics.actor_loss) && isfinite(metrics.critic_loss));
        assert(metrics.critic_loss >= 0.0f);
    }
    assert(metrics.update_count == 4);
    assert(tb_sac_act(agent, obs, mission, cameras, 1, action, NULL));
    tb_train_action_to_physical(action, 0, physical);
    tb_train_action_from_physical(physical, 0, round_trip);
    for (int i = 0; i < 14; ++i) assert(fabsf(round_trip[i] - action[i]) < 1.0e-5f);

    const char *path = "tb_training_test.ckpt";
    assert(tb_sac_save(agent, path));
    tb_sac_agent *loaded = tb_sac_load(path);
    assert(loaded);
    float loaded_action[TB_TRAIN_ACTION_DIM];
    assert(tb_sac_act(loaded, obs, mission, cameras, 1, loaded_action, NULL));
    for (int i = 0; i < TB_TRAIN_ACTION_DIM; ++i)
        assert(fabsf(loaded_action[i] - action[i]) < 1.0e-6f);
    assert(tb_sac_update_count(loaded) == 4);
    tb_sac_destroy(loaded);
    tb_sac_destroy(agent);
    remove(path);
    puts("training tests passed");
    return 0;
}
