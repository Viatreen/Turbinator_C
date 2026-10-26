#ifndef TURBINATOR_TRAINING_H
#define TURBINATOR_TRAINING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matches the simulator's training ABI. */
enum {
    TB_TRAIN_OBSERVATION_DIM = 4148,
    TB_TRAIN_MISSION_DIM = 10,
    TB_TRAIN_CAMERA_PIXELS = 4 * 64 * 64,
    TB_TRAIN_PRIVILEGED_DIM = 64,
    TB_TRAIN_ACTION_DIM = 15,
    TB_TRAIN_FEATURE_DIM = 142
};

typedef struct tb_sac_agent tb_sac_agent;

typedef struct tb_train_config {
    uint64_t seed;
    size_t replay_capacity;
    int hidden_dim;
    float gamma;
    float tau;
    float actor_lr;
    float critic_lr;
    float alpha_lr;
    float initial_alpha;
    float target_entropy;
    float reference_weight;
    float max_grad_norm;
    float reward_scale;
} tb_train_config;

typedef struct tb_train_metrics {
    float critic_loss;
    float actor_loss;
    float reference_loss;
    float q_mean;
    float alpha;
    float entropy;
    float actor_grad_norm;
    float critic_grad_norm;
    uint64_t update_count;
} tb_train_metrics;

/* Uses 52 scalar proprioception values, 10 mission values, 64 LiDAR sector
 * summaries, and 16 camera quadrant summaries. Combat mission goal errors
 * are masked because the source simulator can fill them from opponent truth.
 * NULL cameras mean no image. */
void tb_train_extract_features_cpu(const float obs[TB_TRAIN_OBSERVATION_DIM],
                                   const float mission[TB_TRAIN_MISSION_DIM],
                                   const uint8_t *cameras,
                                   float out_features[TB_TRAIN_FEATURE_DIM]);

void tb_train_config_default(tb_train_config *out);
tb_sac_agent *tb_sac_create(const tb_train_config *config);
void tb_sac_destroy(tb_sac_agent *agent);
size_t tb_sac_replay_size(const tb_sac_agent *agent);
uint64_t tb_sac_update_count(const tb_sac_agent *agent);

/* Actions are normalized to [-1, 1], before conversion to actuator units. */
int tb_sac_act(tb_sac_agent *agent,
               const float obs[TB_TRAIN_OBSERVATION_DIM],
               const float mission[TB_TRAIN_MISSION_DIM],
               const uint8_t *cameras,
               int deterministic,
               float out_action[TB_TRAIN_ACTION_DIM],
               float *out_log_probability);

/* Precomputed 142-value features can be supplied by a batched CUDA sensor
 * extractor while the compact SAC networks and optimizer stay in C11. */
int tb_sac_act_features(tb_sac_agent *agent,
                        const float features[TB_TRAIN_FEATURE_DIM],
                        int deterministic,
                        float out_action[TB_TRAIN_ACTION_DIM],
                        float *out_log_probability);

int tb_sac_add_transition(tb_sac_agent *agent,
                          const float obs[TB_TRAIN_OBSERVATION_DIM],
                          const float mission[TB_TRAIN_MISSION_DIM],
                          const uint8_t *cameras,
                          const float privileged[TB_TRAIN_PRIVILEGED_DIM],
                          const float action[TB_TRAIN_ACTION_DIM],
                          const float reference_action[TB_TRAIN_ACTION_DIM],
                          float reward,
                          const float next_obs[TB_TRAIN_OBSERVATION_DIM],
                          const float next_mission[TB_TRAIN_MISSION_DIM],
                          const uint8_t *next_cameras,
                          const float next_privileged[TB_TRAIN_PRIVILEGED_DIM],
                          int done);
int tb_sac_add_transition_features(tb_sac_agent *agent,
                                   const float features[TB_TRAIN_FEATURE_DIM],
                                   const float privileged[TB_TRAIN_PRIVILEGED_DIM],
                                   const float action[TB_TRAIN_ACTION_DIM],
                                   const float reference_action[TB_TRAIN_ACTION_DIM],
                                   float reward,
                                   const float next_features[TB_TRAIN_FEATURE_DIM],
                                   const float next_privileged[TB_TRAIN_PRIVILEGED_DIM],
                                   int done);

/* Returns 1 after a gradient update, 0 if replay is empty or input invalid. */
int tb_sac_update(tb_sac_agent *agent, size_t batch_size, tb_train_metrics *out);

/* Checkpoints include parameters and optimizer state. Replay is ephemeral. */
int tb_sac_save(const tb_sac_agent *agent, const char *path);
tb_sac_agent *tb_sac_load(const char *path);

void tb_train_action_to_physical(const float normalized[TB_TRAIN_ACTION_DIM],
                                 int bwb_mode,
                                 float physical[TB_TRAIN_ACTION_DIM]);
void tb_train_action_from_physical(const float physical[TB_TRAIN_ACTION_DIM],
                                   int bwb_mode,
                                   float normalized[TB_TRAIN_ACTION_DIM]);

#ifdef __cplusplus
}
#endif

#endif
