#include "sim_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#define TB_TLS __declspec(thread)
#else
#define TB_TLS _Thread_local
#endif
static TB_TLS char last_error[256];
static void error_text(const char *s){snprintf(last_error,sizeof(last_error),"%s",s);}
static TbBatch *batch(void *h){if(!h)error_text("null simulator handle");return (TbBatch*)h;}

static int json_number(const char *s,const char *key,double *out){
    if(!s||!key)return 0;
    char pattern[80];snprintf(pattern,sizeof(pattern),"\"%s\"",key);
    const char *p=strstr(s,pattern);if(!p)return 0;p=strchr(p+strlen(pattern),':');if(!p)return 0;
    char *end;double v=strtod(p+1,&end);if(end==p+1||!isfinite(v))return 0;*out=v;return 1;
}
static int json_bool(const char *s,const char *key,int *out){
    if(!s)return 0;
    char pattern[80];snprintf(pattern,sizeof(pattern),"\"%s\"",key);
    const char *p=strstr(s,pattern);if(!p)return 0;p=strchr(p+strlen(pattern),':');if(!p)return 0;p++;
    while(*p==' '||*p=='\t')p++;
    if(!strncmp(p,"true",4)||*p=='1'){*out=1;return 1;}
    if(!strncmp(p,"false",5)||*p=='0'){*out=0;return 1;}return 0;
}
static void parse_config(TbConfig *c,const char *s){
    double v;
#define GET(name,field) if(json_number(s,name,&v))c->field=v
    GET("dt",dt);GET("pivot_radius_m",pivot_radius);GET("pivot_height_m",pivot_height);
    GET("dry_mass_kg",dry_mass);GET("payload_mass_kg",payload_mass);GET("fuel_capacity_kg",fuel_capacity);
    GET("max_thrust_per_engine_n",max_thrust);GET("installed_thrust_factor",thrust_factor);
    GET("spool_up_time95_s",spool_up95);GET("spool_down_time95_s",spool_down95);
    GET("outer_rate_limit_rad_s",outer_rate_limit);GET("inner_rate_limit_rad_s",inner_rate_limit);
    GET("gust_sigma_m_s",gust_sigma);GET("dome_radius_m",dome_radius);
    GET("ball_actual_mass_kg",ball_mass);GET("ball_exaggerated_impact_mass_kg",impact_mass);
    GET("ball_launch_speed_m_s",ball_speed);GET("shot_cooldown_s",shot_cooldown);
    GET("match_duration_s",match_duration);GET("arena_randomized_start_probability",randomized_start_probability);
    GET("lidar_max_range_m",lidar_max);GET("initial_forward_velocity_m_s",initial_forward_velocity);
    if(json_number(s,"starting_ammunition",&v))c->starting_ammunition=(int)v;
    json_bool(s,"full_length_arena_mode",&c->full_length_arena_mode);
    json_bool(s,"bwb_mode",&c->bwb_mode);
    json_bool(s,"arena_bright_opponent_lidar_surface",&c->bright_lidar);
#undef GET
    c->dt=tb_clamp(c->dt,0.001,0.1);c->fuel_capacity=fmax(0.1,c->fuel_capacity);
    c->max_thrust=fmax(1,c->max_thrust);c->dome_radius=fmax(10,c->dome_radius);
    c->starting_ammunition=c->starting_ammunition<0?0:c->starting_ammunition;
}
static void format_config(TbBatch *b){const TbConfig *c=&b->config;
    snprintf(b->config_json,sizeof(b->config_json),
        "{\"config_version\":5,\"dt\":%.8g,\"pivot_radius_m\":%.8g,\"pivot_height_m\":%.8g,"
        "\"dry_mass_kg\":%.8g,\"payload_mass_kg\":%.8g,\"fuel_capacity_kg\":%.8g,"
        "\"max_thrust_per_engine_n\":%.8g,\"installed_thrust_factor\":%.8g,"
        "\"spool_up_time95_s\":%.8g,\"spool_down_time95_s\":%.8g,"
        "\"dome_radius_m\":%.8g,\"shot_cooldown_s\":%.8g,\"match_duration_s\":%.8g,"
        "\"starting_ammunition\":%d,\"lidar_max_range_m\":%.8g,\"full_length_arena_mode\":%s}",
        c->dt,c->pivot_radius,c->pivot_height,c->dry_mass,c->payload_mass,c->fuel_capacity,
        c->max_thrust,c->thrust_factor,c->spool_up95,c->spool_down95,c->dome_radius,c->shot_cooldown,
        c->match_duration,c->starting_ammunition,c->lidar_max,c->full_length_arena_mode?"true":"false");
}

TB_API const char *tb_get_build_info_json(void){return "{\"project\":\"turbinator-c\",\"action_version\":2,\"physics_version\":7,\"observation_version\":3,\"observation_dim\":4148,\"privileged_critic_dim\":64,\"camera_count\":4,\"camera_pixels\":16384,\"action_dim\":15}";}
TB_API int tb_get_contract_versions(int *v){if(!v)return -1;v[0]=2;v[1]=7;return 0;}
TB_API const char *tb_last_error(void){return last_error;}
TB_API void *tb_create(int batch_size,uint64_t seed){return tb_create_with_config(batch_size,seed,NULL);}
TB_API void *tb_create_with_config(int batch_size,uint64_t seed,const char *json){
    if(batch_size<1||batch_size>4096){error_text("batch size must be between 1 and 4096");return NULL;}
    TbBatch *b=(TbBatch*)calloc(1,sizeof(*b));if(!b){error_text("allocation failed");return NULL;}
    b->batch_size=batch_size;tb_config_defaults(&b->config);parse_config(&b->config,json);format_config(b);
    b->envs=(TbEnv*)calloc((size_t)batch_size,sizeof(TbEnv));
    b->opponent_actions=(float*)calloc((size_t)batch_size*TB_ACTION_DIM,sizeof(float));
    b->opponent_mask=(uint8_t*)calloc((size_t)batch_size,sizeof(uint8_t));
    if(!b->envs||!b->opponent_actions||!b->opponent_mask){tb_destroy(b);error_text("allocation failed");return NULL;}
    for(int i=0;i<batch_size;i++)tb_sim_reset_env(&b->envs[i],&b->config,0,seed+(uint64_t)i*10007);
    last_error[0]=0;return b;
}
TB_API void tb_destroy(void *h){TbBatch *b=(TbBatch*)h;if(!b)return;free(b->envs);free(b->opponent_actions);free(b->opponent_mask);free(b);}
TB_API const char *tb_get_config_json(void *h){TbBatch *b=batch(h);return b?b->config_json:"";}

TB_API int tb_reset_all(void *h,int scenario,uint64_t seed){TbBatch *b=batch(h);if(!b)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_reset_env(&b->envs[i],&b->config,scenario,seed+(uint64_t)i*10007);b->has_opponent_actions=0;return 0;}
TB_API int tb_reset_one(void *h,int i,int scenario,uint64_t seed){TbBatch *b=batch(h);if(!b||i<0||i>=b->batch_size)return -1;tb_sim_reset_env(&b->envs[i],&b->config,scenario,seed);b->opponent_mask[i]=0;return 0;}
TB_API int tb_reset_subset(void *h,int count,const int *indices,const int *scenarios,const uint64_t *seeds){TbBatch *b=batch(h);if(!b||count<0||!indices||!scenarios||!seeds)return -1;for(int k=0;k<count;k++)if(tb_reset_one(b,indices[k],scenarios[k],seeds[k])!=0)return -1;return 0;}
TB_API int tb_observe(void *h,float *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_observe(&b->envs[i],&b->config,0,out+(size_t)i*TB_OBSERVATION_DIM);return 0;}
TB_API int tb_observe_opponents(void *h,float *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_observe(&b->envs[i],&b->config,1,out+(size_t)i*TB_OBSERVATION_DIM);return 0;}

static void mission(const TbEnv *e,int player,const float *obs,float *dst){
    memset(dst,0,TB_MISSION_DIM*sizeof(float));int s=e->scenario;
    if(s==0||s==7)dst[0]=1;
    else if(s==1)dst[1]=1;
    else if(s==6)dst[2]=1;
    else if(s==4||s==5||s==17)dst[3]=1;
    else if(tb_is_arena(s))dst[6]=1;
    else if(s==19)dst[5]=1;
    else dst[4]=1;
    /* Arena actor inputs contain only observable data. The exact opponent truth
       remains available through the privileged critic and truth interfaces. */
    if(!tb_is_arena(s)){
        TbVec3 g=tb_sim_target(e,player);
        dst[7]=(float)tanh((g.x-obs[0])*0.05);dst[8]=(float)tanh((g.y-obs[1])*0.05);dst[9]=(float)tanh((g.z-obs[2])*0.05);
    }
}
TB_API int tb_observe_training_batch(void *h,float *obs,float *missions,uint8_t *cameras,float *priv,double *truth){
    TbBatch *b=batch(h);if(!b||!obs)return -1;
    for(int i=0;i<b->batch_size;i++){TbEnv *e=&b->envs[i];float *o=obs+(size_t)i*TB_OBSERVATION_DIM;tb_sim_observe(e,&b->config,0,o);
        if(missions)mission(e,0,o,missions+(size_t)i*TB_MISSION_DIM);
        if(cameras)memcpy(cameras+(size_t)i*TB_CAMERA_PIXELS,e->camera[0],TB_CAMERA_PIXELS);
        if(priv)tb_sim_privileged(e,&b->config,priv+(size_t)i*TB_PRIVILEGED_DIM);
        if(truth)tb_sim_truth(e,&b->config,truth+(size_t)i*TB_TRUTH_DIM);
    }return 0;
}
TB_API int tb_observe_opponents_training_batch(void *h,float *obs,float *missions,uint8_t *cameras){
    TbBatch *b=batch(h);if(!b||!obs)return -1;
    for(int i=0;i<b->batch_size;i++){TbEnv *e=&b->envs[i];float *o=obs+(size_t)i*TB_OBSERVATION_DIM;tb_sim_observe(e,&b->config,1,o);
        if(missions)mission(e,1,o,missions+(size_t)i*TB_MISSION_DIM);
        if(cameras)memcpy(cameras+(size_t)i*TB_CAMERA_PIXELS,e->camera[1],TB_CAMERA_PIXELS);
    }return 0;
}
TB_API int tb_set_opponent_actions(void *h,const float *actions){return tb_set_opponent_actions_masked(h,actions,NULL);}
TB_API int tb_set_opponent_actions_masked(void *h,const float *actions,const uint8_t *mask){
    TbBatch *b=batch(h);if(!b||!actions)return -1;
    for(size_t i=0;i<(size_t)b->batch_size*TB_ACTION_DIM;i++)if(!isfinite(actions[i])){error_text("nonfinite opponent action");return -1;}
    memcpy(b->opponent_actions,actions,(size_t)b->batch_size*TB_ACTION_DIM*sizeof(float));
    for(int i=0;i<b->batch_size;i++)b->opponent_mask[i]=mask?mask[i]!=0:1;
    b->has_opponent_actions=1;return 0;
}
TB_API int tb_step(void *h,const float *actions,float *obs,float *rewards,uint8_t *terminated,uint8_t *truncated){
    TbBatch *b=batch(h);if(!b||!actions)return -1;
    for(size_t i=0;i<(size_t)b->batch_size*TB_ACTION_DIM;i++)if(!isfinite(actions[i])){error_text("nonfinite action");return -1;}
    for(int i=0;i<b->batch_size;i++){TbEnv *e=&b->envs[i];const float *opp=(b->has_opponent_actions&&b->opponent_mask[i])?b->opponent_actions+(size_t)i*TB_ACTION_DIM:NULL;
        tb_sim_step_env(e,&b->config,actions+(size_t)i*TB_ACTION_DIM,opp);
        if(obs)tb_sim_observe(e,&b->config,0,obs+(size_t)i*TB_OBSERVATION_DIM);
        if(rewards)rewards[i]=e->reward[0];
        if(terminated)terminated[i]=(uint8_t)e->terminated;
        if(truncated)truncated[i]=(uint8_t)e->truncated;
    }return 0;
}
TB_API int tb_get_cameras(void *h,uint8_t *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)memcpy(out+(size_t)i*TB_CAMERA_PIXELS,b->envs[i].camera[0],TB_CAMERA_PIXELS);return 0;}
TB_API int tb_get_opponent_cameras(void *h,uint8_t *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)memcpy(out+(size_t)i*TB_CAMERA_PIXELS,b->envs[i].camera[1],TB_CAMERA_PIXELS);return 0;}
TB_API int tb_get_lidar(void *h,float *range,double *time){TbBatch *b=batch(h);if(!b)return -1;for(int i=0;i<b->batch_size;i++){if(range)memcpy(range+(size_t)i*TB_LIDAR_BINS,b->envs[i].lidar[0],TB_LIDAR_BINS*sizeof(float));if(time){time[2*i]=b->envs[i].lidar_capture[0][0];time[2*i+1]=b->envs[i].lidar_capture[0][1];}}return 0;}
TB_API int tb_get_opponent_lidar(void *h,float *range,double *time){TbBatch *b=batch(h);if(!b)return -1;for(int i=0;i<b->batch_size;i++){if(range)memcpy(range+(size_t)i*TB_LIDAR_BINS,b->envs[i].lidar[1],TB_LIDAR_BINS*sizeof(float));if(time){time[2*i]=b->envs[i].lidar_capture[1][0];time[2*i+1]=b->envs[i].lidar_capture[1][1];}}return 0;}
TB_API int tb_set_raw_camera_capture(void *h,int i,int enabled){TbBatch *b=batch(h);if(!b||i<0||i>=b->batch_size)return -1;b->envs[i].capture_raw=enabled!=0;return 0;}
TB_API int tb_get_raw_cameras(void *h,int i,int player,uint8_t *pixels,double *time){TbBatch *b=batch(h);if(!b||i<0||i>=b->batch_size||player<0||player>1)return -1;if(pixels)memcpy(pixels,b->envs[i].raw_camera[player],TB_CAMERA_RAW_PIXELS);if(time)*time=b->envs[i].raw_capture[player];return 0;}
TB_API int tb_set_external_cameras(void *h,int enabled){TbBatch *b=batch(h);if(!b)return -1;for(int i=0;i<b->batch_size;i++)b->envs[i].external_cameras=enabled!=0;return 0;}
TB_API int tb_get_external_cameras(void *h,int *flags){TbBatch *b=batch(h);if(!b||!flags)return -1;for(int i=0;i<b->batch_size;i++)flags[i]=b->envs[i].external_cameras;return 0;}
TB_API int tb_get_camera_scenes(void *h,int player,double *scenes,int16_t *noise){TbBatch *b=batch(h);if(!b||!scenes||player<0||player>1)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_scene(&b->envs[i],&b->config,player,scenes+(size_t)i*TB_SCENE_DOUBLES,noise?noise+(size_t)i*TB_CAMERA_PIXELS:NULL);return 0;}
TB_API int tb_get_truth(void *h,int i,double *out){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size)return -1;tb_sim_truth(&b->envs[i],&b->config,out);return 0;}
TB_API int tb_get_truth_batch(void *h,double *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_truth(&b->envs[i],&b->config,out+(size_t)i*TB_TRUTH_DIM);return 0;}
TB_API int tb_get_privileged_critic_features(void *h,float *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_privileged(&b->envs[i],&b->config,out+(size_t)i*TB_PRIVILEGED_DIM);return 0;}
TB_API int tb_get_joint_telemetry(void *h,int i,int player,double *out){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size||player<0||player>1)return -1;TbAircraft *a=&b->envs[i].aircraft[player];for(int j=0;j<4;j++){out[j]=a->outer[j];out[4+j]=a->inner[j];out[8+j]=a->outer_rate[j];out[12+j]=a->inner_rate[j];out[16+j]=a->outer_torque[j];out[20+j]=a->inner_torque[j];}return 0;}
TB_API int tb_get_balls(void *h,int i,double *out,int max_balls){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size||max_balls<0)return 0;TbEnv *e=&b->envs[i];int n=e->ball_count<max_balls?e->ball_count:max_balls;for(int j=0;j<n;j++){TbBall *v=&e->balls[j];double *d=out+(size_t)j*11;d[0]=v->pos.x;d[1]=v->pos.y;d[2]=v->pos.z;d[3]=v->vel.x;d[4]=v->vel.y;d[5]=v->vel.z;d[6]=v->owner;d[7]=v->age;d[8]=v->ricochets;d[9]=v->scored;d[10]=v->id;}return n;}
TB_API int tb_get_ball_events(void *h,int i,double *out,int max_events){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size||max_events<0)return 0;TbEnv *e=&b->envs[i];int n=e->event_count<max_events?e->event_count:max_events;for(int j=0;j<n;j++){TbBallEvent *v=&e->events[j];double *d=out+(size_t)j*6;d[0]=v->kind;d[1]=v->ball_id;d[2]=v->owner;d[3]=v->other_player;d[4]=v->time;d[5]=v->ricochets;}return n;}
TB_API int tb_reference_action(void *h,int i,const double *target,float *out){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size)return -1;tb_sim_reference(&b->envs[i],&b->config,target,out);return 0;}
TB_API int tb_reference_action_batch(void *h,float *out){TbBatch *b=batch(h);if(!b||!out)return -1;for(int i=0;i<b->batch_size;i++)tb_sim_reference(&b->envs[i],&b->config,NULL,out+(size_t)i*TB_ACTION_DIM);return 0;}
TB_API int tb_get_scenario_status(void *h,int i,int *out){TbBatch *b=batch(h);if(!b||!out||i<0||i>=b->batch_size)return -1;TbEnv *e=&b->envs[i];out[0]=e->stable_steps;out[1]=e->landing_seen;out[2]=e->landing_hard;out[3]=e->loop_stage;return 0;}

typedef struct {
    uint64_t magic, checksum;
    uint32_t count, env_size, action_dim, has_opponent_actions;
} TbStateHeader;
static int64_t state_bytes(const TbBatch *b){
    return (int64_t)sizeof(TbStateHeader)+(int64_t)b->batch_size*
        ((int64_t)sizeof(TbEnv)+(int64_t)TB_ACTION_DIM*sizeof(float)+sizeof(uint8_t));
}
static uint64_t state_checksum(const uint8_t *data,size_t n){
    uint64_t h=UINT64_C(14695981039346656037);
    for(size_t i=0;i<n;i++){h^=data[i];h*=UINT64_C(1099511628211);}
    return h;
}
static int valid_aircraft(const TbAircraft *a){
    return isfinite(a->pos.x)&&isfinite(a->pos.y)&&isfinite(a->pos.z)&&
        isfinite(a->vel.x)&&isfinite(a->vel.y)&&isfinite(a->vel.z)&&
        isfinite(a->quat.w)&&isfinite(a->quat.x)&&isfinite(a->quat.y)&&isfinite(a->quat.z)&&
        isfinite(a->fuel)&&a->ammo>=0&&a->ammo<=100000;
}
static int valid_env(const TbEnv *e){
    return e->scenario>=0&&e->scenario<=27&&e->step_index>=0&&
        e->ball_count>=0&&e->ball_count<=TB_MAX_BALLS&&
        e->event_count>=0&&e->event_count<=TB_MAX_EVENTS&&
        e->obstacle_count>=0&&e->obstacle_count<=TB_MAX_OBSTACLES&&
        e->terminated>=0&&e->terminated<=1&&e->truncated>=0&&e->truncated<=1&&
        e->winner>=-1&&e->winner<=1&&isfinite(e->time)&&e->time>=0&&
        valid_aircraft(&e->aircraft[0])&&valid_aircraft(&e->aircraft[1]);
}
TB_API int64_t tb_state_size(void *h){TbBatch *b=batch(h);return b?state_bytes(b):-1;}
TB_API int64_t tb_save_state(void *h,uint8_t *out,int64_t capacity){
    TbBatch *b=batch(h);if(!b)return -1;int64_t n=state_bytes(b);
    if(out&&capacity>=n){
        TbStateHeader hdr={UINT64_C(0x5442535441544532),0,(uint32_t)b->batch_size,(uint32_t)sizeof(TbEnv),TB_ACTION_DIM,(uint32_t)b->has_opponent_actions};
        uint8_t *p=out+sizeof(hdr);size_t env_bytes=(size_t)b->batch_size*sizeof(TbEnv),action_bytes=(size_t)b->batch_size*TB_ACTION_DIM*sizeof(float);
        memcpy(p,b->envs,env_bytes);p+=env_bytes;
        memcpy(p,b->opponent_actions,action_bytes);p+=action_bytes;
        memcpy(p,b->opponent_mask,(size_t)b->batch_size);
        hdr.checksum=state_checksum(out+sizeof(hdr),(size_t)(n-(int64_t)sizeof(hdr)));
        memcpy(out,&hdr,sizeof(hdr));
    }
    return n;
}
TB_API int tb_load_state(void *h,const uint8_t *bytes,int64_t size){
    TbBatch *b=batch(h);if(!b||!bytes||size<0||size!=state_bytes(b)){error_text("invalid state size");return -1;}
    TbStateHeader hdr;memcpy(&hdr,bytes,sizeof(hdr));
    if(hdr.magic!=UINT64_C(0x5442535441544532)||hdr.count!=(uint32_t)b->batch_size||
       hdr.env_size!=(uint32_t)sizeof(TbEnv)||hdr.action_dim!=TB_ACTION_DIM||hdr.has_opponent_actions>1){error_text("invalid state header");return -1;}
    if(hdr.checksum!=state_checksum(bytes+sizeof(hdr),(size_t)(size-(int64_t)sizeof(hdr)))){error_text("state checksum mismatch");return -1;}
    TbEnv *sample=(TbEnv*)malloc(sizeof(TbEnv));if(!sample){error_text("state validation allocation failed");return -1;}
    const uint8_t *p=bytes+sizeof(hdr);size_t env_bytes=(size_t)b->batch_size*sizeof(TbEnv),action_bytes=(size_t)b->batch_size*TB_ACTION_DIM*sizeof(float);
    for(int i=0;i<b->batch_size;i++){
        memcpy(sample,p+(size_t)i*sizeof(TbEnv),sizeof(TbEnv));
        if(!valid_env(sample)){free(sample);error_text("invalid simulator state payload");return -1;}
    }
    free(sample);
    const uint8_t *action_data=p+env_bytes;
    /* memcpy avoids alignment assumptions for a serialized byte buffer. */
    for(size_t i=0;i<(size_t)b->batch_size*TB_ACTION_DIM;i++){float value;memcpy(&value,action_data+i*sizeof(float),sizeof(value));if(!isfinite(value)){error_text("invalid opponent action in state");return -1;}}
    memcpy(b->envs,p,env_bytes);p+=env_bytes;
    memcpy(b->opponent_actions,p,action_bytes);p+=action_bytes;
    memcpy(b->opponent_mask,p,(size_t)b->batch_size);
    b->has_opponent_actions=(int)hdr.has_opponent_actions;
    return 0;
}

/* The legacy ev_* names remain linkable for existing C consumers. */
#define ALIAS0(ret,name) TB_API ret ev_##name(void){return tb_##name();}
ALIAS0(const char*,get_build_info_json)
TB_API int ev_get_contract_versions(int *v){return tb_get_contract_versions(v);}
ALIAS0(const char*,last_error)
TB_API void *ev_create(int n,uint64_t seed){return tb_create(n,seed);}
TB_API void *ev_create_with_config(int n,uint64_t seed,const char *cfg){return tb_create_with_config(n,seed,cfg);}
TB_API void ev_destroy(void *h){tb_destroy(h);}
TB_API const char *ev_get_config_json(void *h){return tb_get_config_json(h);}
TB_API int ev_reset_all(void *h,int s,uint64_t seed){return tb_reset_all(h,s,seed);}
TB_API int ev_reset_one(void *h,int i,int s,uint64_t seed){return tb_reset_one(h,i,s,seed);}
TB_API int ev_observe(void *h,float *o){return tb_observe(h,o);}
TB_API int ev_observe_opponents(void *h,float *o){return tb_observe_opponents(h,o);}
TB_API int ev_set_opponent_actions(void *h,const float *a){return tb_set_opponent_actions(h,a);}
TB_API int ev_set_opponent_actions_masked(void *h,const float *a,const uint8_t *m){return tb_set_opponent_actions_masked(h,a,m);}
TB_API int ev_step(void *h,const float *a,float *o,float *r,uint8_t *t,uint8_t *u){return tb_step(h,a,o,r,t,u);}
TB_API int ev_get_cameras(void *h,uint8_t *o){return tb_get_cameras(h,o);}
TB_API int ev_get_opponent_cameras(void *h,uint8_t *o){return tb_get_opponent_cameras(h,o);}
TB_API int ev_get_lidar(void *h,float *o,double *t){return tb_get_lidar(h,o,t);}
TB_API int ev_get_opponent_lidar(void *h,float *o,double *t){return tb_get_opponent_lidar(h,o,t);}
TB_API int ev_set_raw_camera_capture(void *h,int i,int enabled){return tb_set_raw_camera_capture(h,i,enabled);}
TB_API int ev_get_raw_cameras(void *h,int i,int p,uint8_t *o,double *t){return tb_get_raw_cameras(h,i,p,o,t);}
TB_API int ev_set_external_cameras(void *h,int en){return tb_set_external_cameras(h,en);}
TB_API int ev_get_external_cameras(void *h,int *o){return tb_get_external_cameras(h,o);}
TB_API int ev_get_camera_scenes(void *h,int p,double *s,int16_t *n){return tb_get_camera_scenes(h,p,s,n);}
TB_API int ev_get_truth(void *h,int i,double *o){return tb_get_truth(h,i,o);}
TB_API int ev_get_truth_batch(void *h,double *o){return tb_get_truth_batch(h,o);}
TB_API int ev_get_privileged_critic_features(void *h,float *o){return tb_get_privileged_critic_features(h,o);}
TB_API int ev_get_joint_telemetry(void *h,int i,int p,double *o){return tb_get_joint_telemetry(h,i,p,o);}
TB_API int ev_get_balls(void *h,int i,double *o,int n){return tb_get_balls(h,i,o,n);}
TB_API int ev_get_ball_events(void *h,int i,double *o,int n){return tb_get_ball_events(h,i,o,n);}
TB_API int ev_reference_action(void *h,int i,const double *t,float *o){return tb_reference_action(h,i,t,o);}
TB_API int ev_reference_action_batch(void *h,float *o){return tb_reference_action_batch(h,o);}
TB_API int ev_get_scenario_status(void *h,int i,int *o){return tb_get_scenario_status(h,i,o);}
TB_API int64_t ev_state_size(void *h){return tb_state_size(h);}
TB_API int64_t ev_save_state(void *h,uint8_t *o,int64_t n){return tb_save_state(h,o,n);}
TB_API int ev_load_state(void *h,const uint8_t *o,int64_t n){return tb_load_state(h,o,n);}
