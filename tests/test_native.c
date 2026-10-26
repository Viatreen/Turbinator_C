#include "turbinator/c_api.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(int pass,const char *what){if(!pass)fprintf(stderr,"FAIL: %s\n",what);return !pass;}
int main(void){
    int failed=0;
    void *sim=tb_create(2,1234);failed+=check(sim!=NULL,"create");if(!sim)return 1;
    float *obs=(float*)calloc(2*TB_OBSERVATION_DIM,sizeof(float));
    float *act=(float*)calloc(2*TB_ACTION_DIM,sizeof(float));
    float *mission=(float*)calloc(2*TB_MISSION_DIM,sizeof(float));
    uint8_t done[2],trunc[2];float rewards[2];double truth[TB_TRUTH_DIM];
    failed+=check(obs&&act&&mission,"allocate buffers");
    failed+=check(tb_reset_all(sim,3,2026)==0,"reset arena");
    failed+=check(tb_observe_training_batch(sim,obs,mission,NULL,NULL,NULL)==0,"observe training");
    failed+=check(mission[7]==0&&mission[8]==0&&mission[9]==0,"arena mission excludes opponent truth");
    failed+=check(isfinite(obs[0])&&isfinite(obs[4147]),"finite observation");
    failed+=check(tb_get_truth(sim,0,truth)==0,"truth");
    failed+=check(truth[69]==0&&truth[70]==0&&truth[71]==0&&truth[72]==0&&truth[73]==0&&truth[74]==0,"arena zero wind");
    for(int i=0;i<2;i++){for(int j=0;j<4;j++)act[i*TB_ACTION_DIM+j]=0.45f;act[i*TB_ACTION_DIM+12]=0.7853982f;act[i*TB_ACTION_DIM+14]=1;}
    failed+=check(tb_step(sim,act,obs,rewards,done,trunc)==0,"step arena");
    double balls[11*64];failed+=check(tb_get_balls(sim,0,balls,64)>0,"arena projectile");
    int64_t state_size=tb_state_size(sim);uint8_t *state=(uint8_t*)malloc((size_t)state_size);
    failed+=check(state&&tb_save_state(sim,state,state_size)==state_size,"save state");
    double original[TB_TRUTH_DIM];tb_get_truth(sim,0,original);
    if(state&&state_size>64){
        state[64]^=0x5a;
        failed+=check(tb_load_state(sim,state,state_size)==-1,"corrupt state rejected");
        tb_get_truth(sim,0,truth);
        failed+=check(memcmp(original,truth,sizeof(truth))==0,"rejected load leaves state unchanged");
        state[64]^=0x5a;
    }
    tb_step(sim,act,obs,rewards,done,trunc);
    failed+=check(tb_load_state(sim,state,state_size)==0,"load state");tb_get_truth(sim,0,truth);
    failed+=check(memcmp(original,truth,sizeof(truth))==0,"state round trip");
    act[0]=NAN;
    failed+=check(tb_step(sim,act,NULL,rewards,done,trunc)==-1,"nonfinite action rejected");
    tb_get_truth(sim,0,truth);
    failed+=check(memcmp(original,truth,sizeof(truth))==0,"rejected action leaves state unchanged");
    act[0]=0.45f;
    failed+=check(tb_reset_all(sim,3,4040)==0,"reset for action persistence");
    float zero_opponent[2*TB_ACTION_DIM]={0};
    failed+=check(tb_set_opponent_actions(sim,zero_opponent)==0,"set persistent opponent action");
    failed+=check(tb_save_state(sim,state,state_size)==state_size,"save opponent policy state");
    tb_step(sim,act,NULL,rewards,done,trunc);tb_step(sim,act,NULL,rewards,done,trunc);
    double held_action_truth[TB_TRUTH_DIM];tb_get_truth(sim,0,held_action_truth);
    failed+=check(tb_load_state(sim,state,state_size)==0,"restore opponent policy state");
    tb_step(sim,act,NULL,rewards,done,trunc);
    tb_set_opponent_actions(sim,zero_opponent);
    tb_step(sim,act,NULL,rewards,done,trunc);
    tb_get_truth(sim,0,truth);
    failed+=check(memcmp(held_action_truth,truth,sizeof(truth))==0,"opponent action persists across steps");
    for(int scenario=0;scenario<28;scenario++){
        failed+=check(tb_reset_all(sim,scenario,3030+(uint64_t)scenario)==0,"reset scenario");
        failed+=check(tb_reference_action_batch(sim,act)==0,"reference action");
        failed+=check(tb_step(sim,act,obs,rewards,done,trunc)==0,"step scenario");
        tb_get_truth(sim,0,truth);
        failed+=check(isfinite(truth[1])&&isfinite(truth[4])&&isfinite(truth[3])&&truth[3]>0,"finite dynamics");
        if(scenario==3||scenario==14||scenario==15||scenario==16||scenario==18||scenario==22||scenario==24||scenario==27)
            failed+=check(truth[69]==0&&truth[70]==0&&truth[71]==0&&truth[72]==0&&truth[73]==0&&truth[74]==0,"arena wind remains zero");
    }
    uint8_t *cameras=(uint8_t*)calloc(2*TB_CAMERA_PIXELS,1);
    failed+=check(cameras&&tb_get_cameras(sim,cameras)==0,"camera API");
    free(cameras);
    free(state);free(obs);free(act);free(mission);tb_destroy(sim);
    if(!failed)puts("simulator C smoke test passed");
    return failed?1:0;
}
