#pragma once

#include <stdint.h>

/* Shape constants for the C and CUDA C training code. */
#define TB_ACTION_DIM 15
#define TB_OBSERVATION_DIM 4148
#define TB_LIDAR_BINS 4096
#define TB_CAMERA_PIXELS 16384
#define TB_CAMERA_RAW_PIXELS 147456
#define TB_MISSION_DIM 10
#define TB_PRIVILEGED_DIM 64
#define TB_TRUTH_DIM 85
#define TB_SCENE_DOUBLES 873

#ifdef _WIN32
#define TB_API __declspec(dllexport)
#else
#define TB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Primary Turbinator C ABI (tb_*) and backward-compatible aliases (ev_*)
TB_API const char* tb_get_build_info_json(void);
TB_API int tb_get_contract_versions(int* out_versions2);
TB_API const char* tb_last_error(void);

TB_API void* tb_create(int batch_size, uint64_t seed);
TB_API void* tb_create_with_config(int batch_size, uint64_t seed, const char* config_json);
TB_API void tb_destroy(void* handle);
TB_API const char* tb_get_config_json(void* handle);

TB_API int tb_reset_all(void* handle, int scenario, uint64_t seed);
TB_API int tb_reset_one(void* handle, int index, int scenario, uint64_t seed);
TB_API int tb_reset_subset(void* handle, int count, const int* indices, const int* scenarios, const uint64_t* seeds);

TB_API int tb_observe(void* handle, float* out_obs);
TB_API int tb_observe_opponents(void* handle, float* out_obs);
TB_API int tb_observe_training_batch(
    void* handle,
    float* out_obs,
    float* out_mission10,
    uint8_t* out_cameras,
    float* out_priv64,
    double* out_truth85);
TB_API int tb_observe_opponents_training_batch(
    void* handle,
    float* out_opp_obs,
    float* out_opp_mission10,
    uint8_t* out_opp_cameras);
TB_API int tb_set_opponent_actions(void* handle, const float* actions);
TB_API int tb_set_opponent_actions_masked(void* handle, const float* actions, const uint8_t* mask);

TB_API int tb_step(
    void* handle,
    const float* actions,
    float* out_obs,
    float* out_rewards,
    uint8_t* out_terminated,
    uint8_t* out_truncated);

TB_API int tb_get_cameras(void* handle, uint8_t* out_pixels);
TB_API int tb_get_opponent_cameras(void* handle, uint8_t* out_pixels);
TB_API int tb_get_lidar(void* handle, float* out_ranges_m, double* out_capture_times_s);
TB_API int tb_get_opponent_lidar(void* handle, float* out_ranges_m, double* out_capture_times_s);

TB_API int tb_set_raw_camera_capture(void* handle, int index, int enabled);
TB_API int tb_get_raw_cameras(void* handle, int index, int player, uint8_t* out_pixels, double* out_capture_time_s);

TB_API int tb_set_external_cameras(void* handle, int enabled);
TB_API int tb_get_external_cameras(void* handle, int* out_flags);
TB_API int tb_get_camera_scenes(void* handle, int player, double* out_scenes, int16_t* out_q8_noise);

TB_API int tb_get_truth(void* handle, int index, double* out_truth85);
TB_API int tb_get_truth_batch(void* handle, double* out_truth85_batch);
TB_API int tb_get_privileged_critic_features(void* handle, float* out_features64_batch);
TB_API int tb_get_joint_telemetry(void* handle, int index, int player, double* out_joint24);
TB_API int tb_get_balls(void* handle, int index, double* out_balls, int max_balls);
TB_API int tb_get_ball_events(void* handle, int index, double* out_events, int max_events);

TB_API int tb_reference_action(void* handle, int index, const double* target_enu3, float* out_action15);
TB_API int tb_reference_action_batch(void* handle, float* out_actions15_batch);
TB_API int tb_get_scenario_status(void* handle, int index, int* out_status4);

TB_API int64_t tb_state_size(void* handle);
TB_API int64_t tb_save_state(void* handle, uint8_t* out_bytes, int64_t capacity);
TB_API int tb_load_state(void* handle, const uint8_t* bytes, int64_t size);

// Native C TUI Dashboard ABI (tb_tui_*)
TB_API void* tb_tui_create(const char* season_dir, int tui_enabled, double max_duration_minutes, int log_capacity);
TB_API void tb_tui_destroy(void* handle);
TB_API void tb_tui_log(void* handle, const char* message, int also_print_if_not_tui);
TB_API void tb_tui_poll_controls(void* handle, int write_to_control_file);
TB_API void tb_tui_sync_from_disk(void* handle);
TB_API void tb_tui_record_status_json(void* handle, const char* status_json);
TB_API void tb_tui_render_if_due(void* handle, double session_elapsed_min, const char* mobile_api_summary, int force);
TB_API const char* tb_tui_format_frame(void* handle, double session_elapsed_min, const char* mobile_api_summary);
TB_API int tb_tui_get_state_flags(void* handle, int clear_requests);
TB_API double tb_tui_get_max_duration_minutes(void* handle);
TB_API void tb_tui_set_max_duration_minutes(void* handle, double max_duration_minutes);
TB_API int tb_tui_is_indefinite_training(void* handle);
TB_API void tb_tui_set_indefinite_training(void* handle, int enabled, int write_to_control_file);
TB_API void tb_tui_switch_vehicle_mode(void* handle);
TB_API void tb_tui_toggle_desktop_viewer(void* handle);
TB_API const char* tb_tui_get_season_dir(void* handle);
TB_API const char* tb_tui_format_mst_now(void);

// Compatibility symbols (ev_*)
TB_API const char* ev_get_build_info_json(void);
TB_API int ev_get_contract_versions(int* out_versions2);
TB_API const char* ev_last_error(void);
TB_API void* ev_create(int batch_size, uint64_t seed);
TB_API void* ev_create_with_config(int batch_size, uint64_t seed, const char* config_json);
TB_API void ev_destroy(void* handle);
TB_API const char* ev_get_config_json(void* handle);
TB_API int ev_reset_all(void* handle, int scenario, uint64_t seed);
TB_API int ev_reset_one(void* handle, int index, int scenario, uint64_t seed);
TB_API int ev_observe(void* handle, float* out_obs);
TB_API int ev_observe_opponents(void* handle, float* out_obs);
TB_API int ev_set_opponent_actions(void* handle, const float* actions);
TB_API int ev_set_opponent_actions_masked(void* handle, const float* actions, const uint8_t* mask);
TB_API int ev_step(void* handle, const float* actions, float* out_obs, float* out_rewards, uint8_t* out_terminated, uint8_t* out_truncated);
TB_API int ev_get_cameras(void* handle, uint8_t* out_pixels);
TB_API int ev_get_opponent_cameras(void* handle, uint8_t* out_pixels);
TB_API int ev_get_lidar(void* handle, float* out_ranges_m, double* out_capture_times_s);
TB_API int ev_get_opponent_lidar(void* handle, float* out_ranges_m, double* out_capture_times_s);
TB_API int ev_set_raw_camera_capture(void* handle, int index, int enabled);
TB_API int ev_get_raw_cameras(void* handle, int index, int player, uint8_t* out_pixels, double* out_capture_time_s);
TB_API int ev_set_external_cameras(void* handle, int enabled);
TB_API int ev_get_external_cameras(void* handle, int* out_flags);
TB_API int ev_get_camera_scenes(void* handle, int player, double* out_scenes, int16_t* out_q8_noise);
TB_API int ev_get_truth(void* handle, int index, double* out_truth85);
TB_API int ev_get_truth_batch(void* handle, double* out_truth85_batch);
TB_API int ev_get_privileged_critic_features(void* handle, float* out_features64_batch);
TB_API int ev_get_joint_telemetry(void* handle, int index, int player, double* out_joint24);
TB_API int ev_get_balls(void* handle, int index, double* out_balls, int max_balls);
TB_API int ev_get_ball_events(void* handle, int index, double* out_events, int max_events);
TB_API int ev_reference_action(void* handle, int index, const double* target_enu3, float* out_action15);
TB_API int ev_reference_action_batch(void* handle, float* out_actions15_batch);
TB_API int ev_get_scenario_status(void* handle, int index, int* out_status4);
TB_API int64_t ev_state_size(void* handle);
TB_API int64_t ev_save_state(void* handle, uint8_t* out_bytes, int64_t capacity);
TB_API int ev_load_state(void* handle, const uint8_t* bytes, int64_t size);

#ifdef __cplusplus
}
#endif
