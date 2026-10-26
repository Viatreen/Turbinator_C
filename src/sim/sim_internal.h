#ifndef TB_SIM_INTERNAL_H
#define TB_SIM_INTERNAL_H

#include "turbinator/c_api.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define TB_PI 3.14159265358979323846
#define TB_GRAVITY 9.80665
#define TB_MAX_BALLS 64
#define TB_MAX_EVENTS 128
#define TB_MAX_OBSTACLES 24

typedef struct { double x, y, z; } TbVec3;
typedef struct { double w, x, y, z; } TbQuat;
typedef struct { TbVec3 center; double radius; } TbObstacle;
typedef struct {
    TbVec3 pos, vel;
    int owner, id, ricochets, scored;
    double age;
} TbBall;
typedef struct {
    int kind, ball_id, owner, other_player, ricochets;
    double time;
} TbBallEvent;
typedef struct {
    TbVec3 pos, vel, omega, specific_force;
    TbQuat quat;
    double spool[4], outer[4], inner[4], outer_rate[4], inner_rate[4];
    double outer_torque[4], inner_torque[4], outer_thermal[4];
    double fuel, cannon_az, cannon_el, cooldown;
    int ammo, stuck, hits, misses, self_hits, boundary_contacts, shots;
    int payload_attached, payload_delivered, jammed_arm, degraded_engine;
    double degraded_factor;
} TbAircraft;
typedef struct {
    double dt, pivot_radius, pivot_height, dry_mass, payload_mass, fuel_capacity;
    double max_thrust, thrust_factor, spool_up95, spool_down95;
    double outer_rate_limit, inner_rate_limit, gust_sigma, dome_radius;
    double ball_mass, impact_mass, ball_speed, shot_cooldown, match_duration;
    double lidar_max, initial_forward_velocity, randomized_start_probability;
    int starting_ammunition, full_length_arena_mode, bwb_mode, bright_lidar;
    TbVec3 wind;
} TbConfig;
typedef struct {
    int scenario, step_index, terminated, truncated, outcome, winner;
    int stable_steps, landing_seen, landing_hard, loop_stage;
    int obstacle_count, ball_count, event_count, next_ball_id;
    int external_cameras, capture_raw;
    uint64_t rng;
    double time, previous_goal_dist, previous_opp_dist;
    double lidar_capture[2][2], camera_capture[2], raw_capture[2];
    float lidar_valid[2][2], reward[2];
    TbVec3 gust, drag;
    TbAircraft aircraft[2];
    TbObstacle obstacles[TB_MAX_OBSTACLES];
    TbBall balls[TB_MAX_BALLS];
    TbBallEvent events[TB_MAX_EVENTS];
    float lidar[2][TB_LIDAR_BINS];
    uint8_t camera[2][TB_CAMERA_PIXELS];
    uint8_t raw_camera[2][TB_CAMERA_RAW_PIXELS];
} TbEnv;
typedef struct {
    int batch_size, has_opponent_actions;
    TbConfig config;
    TbEnv *envs;
    float *opponent_actions;
    uint8_t *opponent_mask;
    char config_json[1024];
} TbBatch;

static inline double tb_clamp(double v, double lo, double hi) { return fmax(lo, fmin(hi, v)); }
static inline TbVec3 tb_v(double x, double y, double z) { TbVec3 a = {x,y,z}; return a; }
static inline TbVec3 tb_add(TbVec3 a, TbVec3 b) { return tb_v(a.x+b.x,a.y+b.y,a.z+b.z); }
static inline TbVec3 tb_sub(TbVec3 a, TbVec3 b) { return tb_v(a.x-b.x,a.y-b.y,a.z-b.z); }
static inline TbVec3 tb_scale(TbVec3 a, double s) { return tb_v(a.x*s,a.y*s,a.z*s); }
static inline double tb_dot(TbVec3 a, TbVec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline TbVec3 tb_cross(TbVec3 a, TbVec3 b) { return tb_v(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x); }
static inline double tb_norm(TbVec3 a) { return sqrt(tb_dot(a,a)); }
static inline TbVec3 tb_unit(TbVec3 a) { double n=tb_norm(a); return n>1e-12?tb_scale(a,1/n):tb_v(0,0,1); }
static inline TbQuat tb_q(double w,double x,double y,double z) { TbQuat q={w,x,y,z}; return q; }
static inline TbQuat tb_qmul(TbQuat a,TbQuat b) { return tb_q(a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w); }
static inline TbQuat tb_qnorm(TbQuat q) { double n=sqrt(q.w*q.w+q.x*q.x+q.y*q.y+q.z*q.z); if(n<1e-12)return tb_q(1,0,0,0); double s=(q.w<0?-1.0:1.0)/n; return tb_q(q.w*s,q.x*s,q.y*s,q.z*s); }
static inline TbVec3 tb_rotate(TbQuat q,TbVec3 v) { TbVec3 u=tb_v(q.x,q.y,q.z),t=tb_scale(tb_cross(u,v),2); return tb_add(tb_add(v,tb_scale(t,q.w)),tb_cross(u,t)); }
static inline TbVec3 tb_inverse_rotate(TbQuat q,TbVec3 v) { return tb_rotate(tb_q(q.w,-q.x,-q.y,-q.z),v); }
static inline int tb_is_bwb(int s) { return s>=22 && s<=27; }
static inline int tb_is_arena(int s) { return s==3||s==14||s==15||s==16||s==18||s==22||s==24||s==27; }

uint64_t tb_rand64(uint64_t *state);
double tb_uniform(uint64_t *state);
double tb_normal(uint64_t *state);
void tb_config_defaults(TbConfig *c);
void tb_sim_reset_env(TbEnv *e, const TbConfig *c, int scenario, uint64_t seed);
void tb_sim_step_env(TbEnv *e, const TbConfig *c, const float *action, const float *opp_action);
void tb_sim_observe(const TbEnv *e, const TbConfig *c, int player, float *out);
void tb_sim_truth(const TbEnv *e, const TbConfig *c, double *out);
void tb_sim_privileged(const TbEnv *e, const TbConfig *c, float *out);
void tb_sim_reference(const TbEnv *e, const TbConfig *c, const double *target, float *out);
TbVec3 tb_sim_target(const TbEnv *e, int player);
void tb_sim_render_lidar(TbEnv *e, const TbConfig *c, int player);
void tb_sim_render_camera(TbEnv *e, const TbConfig *c, int player);
void tb_sim_scene(const TbEnv *e, const TbConfig *c, int player, double *out, int16_t *noise);

#endif
