#include "sim_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
#include <windows.h>
#else
#include <stdatomic.h>
#endif

typedef struct {double cl,cd,cy,roll,pitch,yaw;} TbAero6;
static float *aero_table;
#ifdef _MSC_VER
static volatile LONG aero_init_state=0;
static int aero_claim_init(void){return InterlockedCompareExchange(&aero_init_state,1,0)==0;}
static int aero_state(void){return (int)InterlockedCompareExchange(&aero_init_state,0,0);}
static void aero_done(void){InterlockedExchange(&aero_init_state,2);}
#else
static atomic_int aero_init_state=ATOMIC_VAR_INIT(0);
static int aero_claim_init(void){int expected=0;return atomic_compare_exchange_strong(&aero_init_state,&expected,1);}
static int aero_state(void){return atomic_load(&aero_init_state);}
static void aero_done(void){atomic_store(&aero_init_state,2);}
#endif
static void load_aero_table(void){
    if(!aero_claim_init()){
        while(aero_state()!=2){}
        return;
    }
    const char *paths[]={"aero_5d_lookup_table.bin","data/aero_5d_lookup_table.bin","../data/aero_5d_lookup_table.bin","../../data/aero_5d_lookup_table.bin"};
    const size_t scalars=(size_t)21*11*9*9*9*6;
    for(size_t i=0;i<sizeof(paths)/sizeof(paths[0]);i++){
        FILE *f=fopen(paths[i],"rb");if(!f)continue;
        if(fseek(f,0,SEEK_END)!=0||(size_t)ftell(f)!=scalars*sizeof(float)){fclose(f);continue;}
        rewind(f);float *p=(float*)malloc(scalars*sizeof(float));
        if(p&&fread(p,sizeof(float),scalars,f)==scalars){aero_table=p;fclose(f);aero_done();return;}
        free(p);fclose(f);
    }
    aero_done();
}
static void axis_coord(double v,double min,double step,int n,int *index,double *frac){
    double u=tb_clamp((v-min)/step,0,n-1);*index=(int)u;if(*index>n-2)*index=n-2;*frac=u-*index;
}
static TbAero6 aero_coefficients(double alpha,double beta,double speed,double gp,double gy){
    load_aero_table();
    if(aero_table){
        int ix[5];double f[5];
        axis_coord(alpha*180/TB_PI,-16,2,21,&ix[0],&f[0]);
        axis_coord(beta*180/TB_PI,-20,4,11,&ix[1],&f[1]);
        axis_coord(speed,5,5,9,&ix[2],&f[2]);
        axis_coord(gp*180/TB_PI,-30,7.5,9,&ix[3],&f[3]);
        axis_coord(gy*180/TB_PI,-30,7.5,9,&ix[4],&f[4]);
        double out[6]={0};
        for(int mask=0;mask<32;mask++){
            int a=ix[0]+((mask>>0)&1),b=ix[1]+((mask>>1)&1),v=ix[2]+((mask>>2)&1),p=ix[3]+((mask>>3)&1),y=ix[4]+((mask>>4)&1);
            double w=1;for(int d=0;d<5;d++)w*=((mask>>d)&1)?f[d]:(1-f[d]);
            size_t base=(size_t)(((((a*11+b)*9+v)*9+p)*9+y)*6);
            for(int k=0;k<6;k++)out[k]+=w*aero_table[base+k];
        }
        return (TbAero6){out[0],out[1],out[2],out[3],out[4],out[5]};
    }
    /* Deterministic analytic fallback when the table file is unavailable. */
    double stall=1/(1+exp((fabs(alpha*180/TB_PI)-16.8)/2.6));
    double vortex=alpha>6*TB_PI/180?0.14*pow(sin(fmax(0,alpha-6*TB_PI/180)),1.6)*stall:0;
    double cosb2=cos(beta)*cos(beta),mach=speed/340.3;
    double cl_att=(0.088+3.22*alpha)*cosb2/sqrt(fmax(0.5,1-mach*mach))+vortex;
    double cl_sep=(alpha>=0?1:-1)*0.94*sin(2*fabs(alpha))*cosb2;
    double cl=stall*cl_att+(1-stall)*cl_sep+0.068*sin(gp)*15/fmax(10,speed);
    double cd=0.0228*pow(25/fmax(5,speed),0.14)+cl*cl/(TB_PI*2.86*0.862)+(1-stall)*1.12*sin(alpha)*sin(alpha)+0.135*sin(beta)*sin(beta)+0.0115*(sin(gp)*sin(gp)+sin(gy)*sin(gy));
    double da=alpha-3.6*TB_PI/180;
    return (TbAero6){cl,cd,-0.49*beta+0.029*sin(gy),-(0.098+0.066*fmax(0,cl))*beta+0.0085*sin(gy),-0.53*da-0.185*da*da*da-0.043*sin(gp),0.138*beta*stall-0.036*sin(gy)};
}

uint64_t tb_rand64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}
double tb_uniform(uint64_t *state) { return (double)(tb_rand64(state) >> 11) * (1.0/9007199254740992.0); }
double tb_normal(uint64_t *state) { double u=fmax(1e-12,tb_uniform(state)),v=tb_uniform(state); return sqrt(-2*log(u))*cos(2*TB_PI*v); }

void tb_config_defaults(TbConfig *c) {
    memset(c,0,sizeof(*c));
    c->dt=0.02; c->pivot_radius=1; c->pivot_height=0.1625;
    c->dry_mass=43; c->payload_mass=30; c->fuel_capacity=9;
    c->max_thrust=294.2; c->thrust_factor=0.85;
    c->spool_up95=3; c->spool_down95=4.5;
    c->outer_rate_limit=2; c->inner_rate_limit=2;
    c->gust_sigma=1.6; c->wind=tb_v(1.5,-0.8,0);
    c->dome_radius=40; c->ball_mass=0.2; c->impact_mass=4;
    c->ball_speed=20; c->shot_cooldown=3; c->match_duration=180;
    c->randomized_start_probability=0.25; c->starting_ammunition=30;
    c->lidar_max=30; c->initial_forward_velocity=22;
    c->bright_lidar=1;
}

static TbQuat yaw_quat(double yaw) { return tb_q(cos(0.5*yaw),0,0,sin(0.5*yaw)); }
static double quat_tilt(TbQuat q) { return acos(tb_clamp(tb_rotate(q,tb_v(0,0,1)).z,-1,1)); }
static void add_obstacle(TbEnv *e,TbVec3 p,double r) { if(e->obstacle_count<TB_MAX_OBSTACLES){e->obstacles[e->obstacle_count].center=p;e->obstacles[e->obstacle_count++].radius=r;} }

TbVec3 tb_sim_target(const TbEnv *e,int player) {
    if(tb_is_arena(e->scenario)) return e->aircraft[1-player].pos;
    switch(e->scenario) {
        case 1: return tb_v(0,0,0.60);
        case 4: return tb_v(0,0,14);
        case 6: return tb_v(15,0,e->aircraft[0].payload_attached?0.60:2.2);
        case 2: case 9: case 20: return tb_v(36,0,8);
        case 8: return tb_v(52,0,8);
        case 10: return tb_v(36,0,17);
        case 11: return tb_v(36,0,5);
        case 21: { static const double x[]={6,12.5,19,25},y[]={2.8,-2.8,2.8,0}; int i=e->loop_stage<3?e->loop_stage:3;return tb_v(x[i],y[i],8); }
        case 23: case 25: case 26: { static const double x[]={-45,50,145},y[]={18,-14,0};int i=e->loop_stage<2?e->loop_stage:2;return tb_v(x[i],y[i],65); }
        default: return tb_v(0,0,8);
    }
}

static double terrain(const TbEnv *e,double x) {
    if(e->scenario==10) {double t=tb_clamp((x-10)/16,0,1);return 9*t*t*(3-2*t);}
    if(e->scenario==20) {double t=tb_clamp((x-8)/30,0,1);return 2.5*sin(TB_PI*t);}
    return 0;
}

void tb_sim_reset_env(TbEnv *e,const TbConfig *c,int scenario,uint64_t seed) {
    int external=e->external_cameras,raw=e->capture_raw;
    memset(e,0,sizeof(*e)); e->external_cameras=external;e->capture_raw=raw;
    e->scenario=(scenario>=0&&scenario<=27)?scenario:0;e->rng=seed?seed:1;
    e->winner=-1;e->next_ball_id=1;
    const int arena=tb_is_arena(e->scenario),bwb=tb_is_bwb(e->scenario);
    if(!arena) e->gust=tb_v(tb_normal(&e->rng)*0.4*c->gust_sigma,tb_normal(&e->rng)*0.4*c->gust_sigma,0);
    for(int p=0;p<2;p++) {
        TbAircraft *a=&e->aircraft[p];a->quat=tb_q(1,0,0,0);
        a->pos=tb_v(0,0,bwb?65:8);a->specific_force=tb_v(0,0,TB_GRAVITY);
        a->fuel=bwb?2.85:c->fuel_capacity;a->jammed_arm=-1;a->degraded_engine=-1;a->degraded_factor=1;
        a->ammo=arena?(c->full_length_arena_mode&&c->starting_ammunition<60?60:c->starting_ammunition):0;
        a->payload_attached=bwb||(e->scenario==6&&p==0);
        a->cannon_az=bwb?0:TB_PI/4;
        double mass=(arena?c->dry_mass-3:c->dry_mass)+a->fuel+(a->payload_attached?c->payload_mass:0)+a->ammo*c->ball_mass;
        double hover=tb_clamp((mass*TB_GRAVITY/4-c->max_thrust*c->thrust_factor*0.08)/(c->max_thrust*c->thrust_factor*0.92),0,1);
        for(int j=0;j<4;j++) a->spool[j]=bwb?(e->scenario==25?0.36:0.54):hover;
    }
    if(arena) {
        double dome=bwb?300:c->dome_radius,ratio=bwb?0.45:0.88;
        if(scenario==14)ratio=0.30;
        if(scenario==15)ratio=0.40;
        if(scenario==16||scenario==18)ratio=0.60;
        if(scenario==24)ratio=0.32;
        double half=0.5*ratio*dome,angle=(tb_uniform(&e->rng)<c->randomized_start_probability&&!c->full_length_arena_mode)?(2*TB_PI*tb_uniform(&e->rng)):0;
        TbVec3 dir=tb_v(cos(angle),sin(angle),0);
        e->aircraft[0].pos=tb_v(-half*dir.x,-half*dir.y,bwb?65:8);
        e->aircraft[1].pos=tb_v(half*dir.x,half*dir.y,bwb?65:8);
        e->aircraft[0].quat=yaw_quat(angle);
        e->aircraft[1].quat=yaw_quat(angle+TB_PI);
        if(scenario==15)e->aircraft[1].stuck=1;
        if(bwb) {e->aircraft[0].vel=tb_scale(dir,c->initial_forward_velocity);e->aircraft[1].vel=tb_scale(dir,-c->initial_forward_velocity);}
        if(scenario==27)for(int p=0;p<2;p++)e->aircraft[p].fuel*=0.65;
    } else {
        TbAircraft *a=&e->aircraft[0];
        if(scenario==4){a->quat=tb_q(0,1,0,0);a->pos.z=14;}
        if(scenario==1)a->pos.z=5;
        if(scenario==5){a->jammed_arm=1;}
        if(scenario==17){a->degraded_engine=2;a->degraded_factor=0.25;}
        if(bwb){a->vel=tb_v(c->initial_forward_velocity,0,0);a->pos=tb_v(-140,0,65);}
        if(scenario==2||scenario==8){for(int i=0;i<5;i++)add_obstacle(e,tb_v(8+i*7,(i&1)?-1.5:1.5,8),1.25);}
        if(scenario==9||scenario==20){for(int i=0;i<5;i++)add_obstacle(e,tb_v(8+i*6,(i&1)?-2.2:2.2,8),1.8);}
        if(scenario==11){a->pos.z=5;for(int i=0;i<5;i++)add_obstacle(e,tb_v(8+i*6,0,9.2),2.6);}
        if(scenario==21){add_obstacle(e,tb_v(3.2,-1.2,8),1.05);add_obstacle(e,tb_v(9.5,1.2,8),1.05);add_obstacle(e,tb_v(16,-1.2,8),1.05);}
    }
    e->previous_goal_dist=tb_norm(tb_sub(tb_sim_target(e,0),e->aircraft[0].pos));
    e->previous_opp_dist=tb_norm(tb_sub(e->aircraft[1].pos,e->aircraft[0].pos));
    for(int p=0;p<(arena?2:1);p++){tb_sim_render_lidar(e,c,p);if(!e->external_cameras)tb_sim_render_camera(e,c,p);}
}

static void fire_ball(TbEnv *e,const TbConfig *c,int player) {
    TbAircraft *a=&e->aircraft[player];if(e->ball_count>=TB_MAX_BALLS||a->ammo<=0)return;
    TbBall *b=&e->balls[e->ball_count++];memset(b,0,sizeof(*b));
    b->owner=player;b->id=e->next_ball_id++;
    double ca=cos(a->cannon_az),sa=sin(a->cannon_az),ce=cos(a->cannon_el),se=sin(a->cannon_el);
    TbVec3 dir=tb_rotate(a->quat,tb_v(ca*ce,sa*ce,se));
    b->pos=tb_add(a->pos,tb_scale(dir,0.85));b->vel=tb_add(a->vel,tb_scale(dir,tb_is_bwb(e->scenario)?75:c->ball_speed));
    a->ammo--;a->shots++;a->cooldown=tb_is_bwb(e->scenario)?0.60:c->shot_cooldown;
    if(e->event_count<TB_MAX_EVENTS)e->events[e->event_count++]=(TbBallEvent){0,b->id,player,-1,0,e->time};
}

static void integrate_quat(TbAircraft *a,double dt) {
    TbQuat q=a->quat,om=tb_q(0,a->omega.x,a->omega.y,a->omega.z),dq=tb_qmul(q,om);
    a->quat=tb_qnorm(tb_q(q.w+0.5*dt*dq.w,q.x+0.5*dt*dq.x,q.y+0.5*dt*dq.y,q.z+0.5*dt*dq.z));
}

static void advance_quad(TbEnv *e,const TbConfig *c,int player,const float *u) {
    TbAircraft *a=&e->aircraft[player];if(a->stuck)return;
    double dt=c->dt/4;
    for(int sub=0;sub<4;sub++) {
        double mass=(tb_is_arena(e->scenario)?c->dry_mass-3:c->dry_mass)+a->fuel+(a->payload_attached?c->payload_mass:0)+a->ammo*c->ball_mass;
        TbVec3 force=tb_v(0,0,0),moment=tb_v(0,0,0);
        for(int j=0;j<4;j++) {
            double cmd=(a->fuel>1e-5)?tb_clamp(u[j],0,1):0;
            double tau=fmax(0.05,(cmd>a->spool[j]?c->spool_up95:c->spool_down95)/2.995732273553991);
            a->spool[j]+= (cmd-a->spool[j])*(1-exp(-dt/tau));
            double desired_outer=(j==a->jammed_arm)?0:tb_clamp(u[4+j],-c->outer_rate_limit,c->outer_rate_limit);
            double ar=tb_clamp((desired_outer-a->outer_rate[j])*14,-10,10);
            a->outer_rate[j]=tb_clamp(a->outer_rate[j]+ar*dt,-c->outer_rate_limit,c->outer_rate_limit);
            if(j!=a->jammed_arm)a->outer[j]=tb_clamp(a->outer[j]+a->outer_rate[j]*dt,-0.65,0.65);
            double target_inner=tb_clamp(u[8+j],-TB_PI/4,TB_PI/4);
            double ir=tb_clamp((target_inner-a->inner[j])*8,-c->inner_rate_limit,c->inner_rate_limit);
            double ia=tb_clamp((ir-a->inner_rate[j])*14,-12,12);
            a->inner_rate[j]=tb_clamp(a->inner_rate[j]+ia*dt,-c->inner_rate_limit,c->inner_rate_limit);
            a->inner[j]=tb_clamp(a->inner[j]+a->inner_rate[j]*dt,-TB_PI/4,TB_PI/4);
            a->outer_torque[j]=tb_clamp(ar*0.25+fabs(a->outer_rate[j])*1.0,-72,72);
            a->inner_torque[j]=tb_clamp(ia*0.25+fabs(a->inner_rate[j])*1.0,-72,72);
            a->outer_thermal[j]+=dt*(fabs(a->outer_torque[j])/24-a->outer_thermal[j])/18;
            double thrust=c->max_thrust*c->thrust_factor*(0.08+0.92*a->spool[j]);
            if(j==a->degraded_engine)thrust*=a->degraded_factor;
            double rad=(j==0?0:(j==1?TB_PI/2:(j==2?TB_PI:3*TB_PI/2)));
            TbVec3 radial=tb_v(cos(rad),sin(rad),0),tangent=tb_v(-sin(rad),cos(rad),0);
            TbVec3 dir=tb_add(tb_add(tb_scale(radial,sin(a->inner[j])),tb_scale(tangent,sin(a->outer[j])*cos(a->inner[j]))),tb_v(0,0,cos(a->outer[j])*cos(a->inner[j])));
            TbVec3 f=tb_scale(dir,thrust),pivot=tb_v(c->pivot_radius*radial.x,c->pivot_radius*radial.y,c->pivot_height);
            force=tb_add(force,f);moment=tb_add(moment,tb_cross(pivot,f));
            moment.z+=(j&1?-1:1)*0.008*thrust;
            a->fuel=fmax(0,a->fuel-dt*(0.00367+0.010*a->spool[j]));
        }
        TbVec3 wind=tb_is_arena(e->scenario)?tb_v(0,0,0):tb_add(c->wind,e->gust);
        TbVec3 vair=tb_sub(a->vel,wind);
        double v=tb_norm(vair),rho=1.225*exp(-fmax(0,a->pos.z)/8500);
        TbVec3 drag=tb_scale(vair,-0.5*rho*0.7*v); if(player==0)e->drag=drag;
        TbVec3 fw=tb_add(tb_rotate(a->quat,force),drag);
        a->specific_force=tb_inverse_rotate(a->quat,tb_scale(fw,1/fmax(1,mass)));
        TbVec3 acc=tb_add(tb_scale(fw,1/fmax(1,mass)),tb_v(0,0,-TB_GRAVITY));
        a->vel=tb_add(a->vel,tb_scale(acc,dt));a->pos=tb_add(a->pos,tb_scale(a->vel,dt));
        moment=tb_sub(moment,tb_add(tb_scale(a->omega,1.2),tb_scale(a->omega,0.35*tb_norm(a->omega))));
        a->omega=tb_add(a->omega,tb_v(dt*moment.x/11.5,dt*moment.y/11.5,dt*moment.z/19));
        integrate_quat(a,dt);
        double floor_z=terrain(e,a->pos.x)+0.64;
        if(a->pos.z<floor_z){if(a->vel.z< -0.75||hypot(a->vel.x,a->vel.y)>0.6)e->landing_hard=1;e->landing_seen=1;a->pos.z=floor_z;if(a->vel.z<0)a->vel.z=0;a->vel.x*=0.90;a->vel.y*=0.90;}
    }
}

static void advance_bwb(TbEnv *e,const TbConfig *c,int player,const float *u) {
    TbAircraft *a=&e->aircraft[player];if(a->stuck)return;
    double dt=c->dt/4;
    for(int sub=0;sub<4;sub++) {
        double thrust=0;
        for(int j=0;j<2;j++){double cmd=a->fuel>0?tb_clamp(u[j],0,1):0;double tau=(cmd>a->spool[j]?0.08:0.10)/2.995732273553991;a->spool[j]+=(cmd-a->spool[j])*(1-exp(-dt/tau));thrust+=48*(0.02+0.98*a->spool[j]);}
        a->spool[2]=a->spool[0];a->spool[3]=a->spool[1];
        for(int j=0;j<2;j++){double target=tb_clamp(u[8+j]+0.08*u[4+j],-TB_PI/6,TB_PI/6);double desired=tb_clamp(12*(target-a->inner[j]),-6,6);double accel=tb_clamp(22*(desired-a->inner_rate[j]),-45,45);a->inner_rate[j]=tb_clamp(a->inner_rate[j]+accel*dt,-6,6);a->inner[j]=tb_clamp(a->inner[j]+a->inner_rate[j]*dt,-TB_PI/6,TB_PI/6);a->outer[j]=a->inner[j];a->outer_rate[j]=a->inner_rate[j];a->inner_torque[j]=tb_clamp(0.06*accel+0.015*thrust*sin(a->inner[j]),-72,72);a->outer_torque[j]=a->inner_torque[j];}
        TbVec3 thrust_dir=tb_unit(tb_v(cos(a->inner[0])*cos(a->inner[1]),cos(a->inner[0])*sin(a->inner[1]),-sin(a->inner[0])));
        TbVec3 fb=tb_scale(thrust_dir,thrust);
        TbVec3 moment=tb_add(tb_cross(tb_v(-0.465,0,0),fb),tb_scale(thrust_dir,14*(a->spool[0]-a->spool[1])));
        TbVec3 wind=tb_is_arena(e->scenario)?tb_v(0,0,0):tb_add(c->wind,e->gust);
        TbVec3 vbody=tb_inverse_rotate(a->quat,tb_sub(a->vel,wind));
        double speed=fmax(0.5,tb_norm(vbody)),alpha=atan2(-vbody.z,fmax(0.5,vbody.x)),beta=asin(tb_clamp(-vbody.y/speed,-1,1));
        double q=0.5*1.225*exp(-fmax(0,a->pos.z)/8500)*speed*speed;
        TbAero6 coeff=aero_coefficients(alpha,beta,speed,a->inner[0],a->inner[1]);
        double cl=coeff.cl,cd=coeff.cd,cy=coeff.cy;
        TbVec3 lift=tb_v(sin(alpha),0,cos(alpha));
        TbVec3 aero=tb_add(tb_scale(lift,q*2.55*0.685*cl),tb_scale(tb_unit(vbody),-q*1.35*0.685*(cd+0.024)));
        aero.y+=-q*1.60*0.685*cy;
        TbVec3 fworld=tb_rotate(a->quat,tb_add(fb,aero));
        if(player==0)e->drag=tb_rotate(a->quat,tb_scale(tb_unit(vbody),-q*1.35*0.685*(cd+0.024)));
        double mass=10.4640763+(a->payload_attached?4.5359237:0);
        a->specific_force=tb_scale(tb_add(fb,aero),1/mass);
        TbVec3 acc=tb_add(tb_scale(fworld,1/mass),tb_v(0,0,-TB_GRAVITY));
        a->vel=tb_add(a->vel,tb_scale(acc,dt));a->pos=tb_add(a->pos,tb_scale(a->vel,dt));
        double inv2v=1/(2*fmax(6,speed));
        double phat=a->omega.x*1.26*inv2v,qhat=a->omega.y*0.565*inv2v,rhat=a->omega.z*1.26*inv2v;
        double cm=coeff.pitch+0.042*sin(a->inner[0]),cn=coeff.yaw+0.035*sin(a->inner[1]);
        TbVec3 aero_moment=tb_v(q*0.685*1.26*(0.14*coeff.roll+0.35*(-0.42)*phat),
            q*0.685*0.565*(-0.18*cm-0.015*sin(a->inner[0])+0.85*(-1.85)*qhat),
            q*0.685*1.26*(-0.16*cn-0.012*sin(a->inner[1])+0.65*(-0.18)*rhat));
        moment=tb_add(moment,aero_moment);
        a->omega=tb_add(a->omega,tb_v(moment.x*dt/0.82,moment.y*dt/0.64,moment.z*dt/1.38));integrate_quat(a,dt);
        double gp=a->inner[0],gy=a->inner[1],tp=a->inner_torque[0],ty=a->inner_torque[1],rp=a->inner_rate[0],ry=a->inner_rate[1];
        double computer=tb_is_arena(e->scenario)?62:(e->scenario==25?32:45);
        double gimbal=12+18*(sin(gp)*sin(gp)+sin(gy)*sin(gy))+0.45*(tp*tp+ty*ty)+(fabs(tp*rp)+fabs(ty*ry))/0.82+28*(rp*rp+ry*ry);
        double prop=3200*0.5*(pow(a->spool[0],3)+pow(a->spool[1],3))+420*pow(a->spool[0]-a->spool[1],2);
        a->fuel=fmax(0,a->fuel-dt*(prop+gimbal+34+computer)*2.85/(2301.7*1000));
        if(a->pos.z<0.64){a->pos.z=0.64;if(a->vel.z<0)a->vel.z=0;}
    }
}

static void update_balls(TbEnv *e,const TbConfig *c) {
    double dome=tb_is_bwb(e->scenario)?300:c->dome_radius;
    for(int i=0;i<e->ball_count;){TbBall *b=&e->balls[i];TbVec3 prev=b->pos;
        b->age+=c->dt;b->vel.z-=TB_GRAVITY*c->dt;b->pos=tb_add(b->pos,tb_scale(b->vel,c->dt));
        int remove=b->age>6;
        for(int p=0;p<2&&!remove;p++){
            if(b->age<0.04&&p==b->owner)continue;
            TbVec3 target=e->aircraft[p].pos,segment=tb_sub(b->pos,prev),to=tb_sub(target,prev);
            double t=tb_clamp(tb_dot(to,segment)/fmax(1e-9,tb_dot(segment,segment)),0,1);
            if(tb_norm(tb_sub(target,tb_add(prev,tb_scale(segment,t))))<0.45+(tb_is_bwb(e->scenario)?0.40:0.15)){
                TbAircraft *a=&e->aircraft[b->owner];if(p==b->owner)a->self_hits++;else a->hits++;
                if(e->event_count<TB_MAX_EVENTS)e->events[e->event_count++]=(TbBallEvent){p==b->owner?2:1,b->id,b->owner,p,b->ricochets,e->time};
                remove=1;b->scored=1;
            }
        }
        if(!remove&&(b->pos.z<0.25||hypot(b->pos.x,b->pos.y)>dome)){
            if(b->ricochets<3){if(b->pos.z<0.25){b->pos.z=0.25;b->vel.z=fabs(b->vel.z)*0.82;}else {double r=hypot(b->pos.x,b->pos.y),nx=b->pos.x/r,ny=b->pos.y/r,vn=b->vel.x*nx+b->vel.y*ny;b->vel.x-=1.82*vn*nx;b->vel.y-=1.82*vn*ny;b->pos.x=nx*(dome-0.25);b->pos.y=ny*(dome-0.25);}b->ricochets++;if(e->event_count<TB_MAX_EVENTS)e->events[e->event_count++]=(TbBallEvent){4,b->id,b->owner,-1,b->ricochets,e->time};}
            else remove=1;
        }
        if(remove){if(!b->scored){e->aircraft[b->owner].misses++;if(e->event_count<TB_MAX_EVENTS)e->events[e->event_count++]=(TbBallEvent){3,b->id,b->owner,-1,b->ricochets,e->time};}e->balls[i]=e->balls[--e->ball_count];}else i++;
    }
}

static void score_env(TbEnv *e,const TbConfig *c){
    TbAircraft *a=&e->aircraft[0],*b=&e->aircraft[1];
    if(tb_is_arena(e->scenario)){
        double dist=tb_norm(tb_sub(b->pos,a->pos)),desired=tb_is_bwb(e->scenario)?68:13;
        double progress=fabs(e->previous_opp_dist-desired)-fabs(dist-desired);e->previous_opp_dist=dist;
        TbVec3 dir=tb_rotate(a->quat,tb_v(cos(a->cannon_az)*cos(a->cannon_el),sin(a->cannon_az)*cos(a->cannon_el),sin(a->cannon_el)));
        double aim=tb_dot(tb_unit(tb_sub(b->pos,a->pos)),dir);
        double reward=0.02+0.045*fmax(0,(aim-0.75)/0.25)+0.03*tb_clamp(progress,-0.5,0.5);
        for(int i=0;i<e->event_count;i++)if(fabs(e->events[i].time-e->time)<1e-8){TbBallEvent *ev=&e->events[i];if(ev->kind==1)reward+=ev->owner==0?12:-8;if(ev->kind==2)reward+=ev->owner==0?-10:4;}
        if(!c->full_length_arena_mode && ((a->hits>=3&&a->hits>b->hits)||(e->scenario==14&&a->hits>=1))){e->terminated=1;e->winner=0;e->outcome=6;reward+=20;}
        else if(!c->full_length_arena_mode&&b->hits>=3&&b->hits>a->hits){e->terminated=1;e->winner=1;e->outcome=7;reward-=15;}
        else if(e->time>=(c->full_length_arena_mode?c->match_duration:24)){e->truncated=1;int s0=3*a->hits-4*a->self_hits-2*a->stuck,s1=3*b->hits-4*b->self_hits-2*b->stuck;e->winner=s0>s1?0:(s1>s0?1:-1);e->outcome=e->winner==0?6:(e->winner==1?7:8);}
        e->reward[0]=(float)reward;e->reward[1]=(float)-reward;return;
    }
    TbVec3 goal=tb_sim_target(e,0);double dist=tb_norm(tb_sub(goal,a->pos)),speed=tb_norm(a->vel),tilt=quat_tilt(a->quat);
    for(int i=0;i<e->obstacle_count;i++)if(tb_norm(tb_sub(a->pos,e->obstacles[i].center))<e->obstacles[i].radius+0.45){e->terminated=1;e->outcome=3;e->reward[0]=-20;return;}
    if(a->pos.z<terrain(e,a->pos.x)+0.65&&e->scenario!=1&&e->scenario!=6){e->terminated=1;e->outcome=2;e->reward[0]=-20;return;}
    if(fabs(a->pos.x)>(tb_is_bwb(e->scenario)?450:120)||fabs(a->pos.y)>(tb_is_bwb(e->scenario)?450:120)||a->pos.z>(tb_is_bwb(e->scenario)?360:85)||tb_norm(a->omega)>18){e->terminated=1;e->outcome=9;e->reward[0]=-20;return;}
    if(a->fuel<=1e-4){e->terminated=1;e->outcome=4;e->reward[0]=-10;return;}
    double progress=e->previous_goal_dist-dist;e->previous_goal_dist=dist;
    double reward=1.8*tb_clamp(progress,-0.5,0.5)+0.08*exp(-0.35*dist)+0.05*cos(tilt)-0.008*tb_norm(a->omega);
    if(tb_is_bwb(e->scenario)){double v=tb_norm(a->vel);reward+=0.06*exp(-pow((v-(e->scenario==25?18.5:24))/7.5,2));if(dist<26&&v>13&&e->loop_stage<2){e->loop_stage++;reward+=8;}}
    if(e->scenario==6&&a->payload_attached&&tb_norm(tb_sub(a->pos,tb_v(15,0,0.6)))<1.2){a->payload_attached=0;a->payload_delivered=1;reward+=5;}
    int stable=0,required=25;
    if(e->scenario==0||e->scenario==7){stable=dist<0.75&&speed<0.65&&tilt<0.18;required=60;}
    else if(e->scenario==1){stable=e->landing_seen&&!e->landing_hard&&a->pos.z<0.72&&speed<0.25&&tilt<0.14;required=35;}
    else if(e->scenario==6){stable=a->payload_delivered&&dist<1.15&&speed<0.8&&tilt<0.20;required=35;}
    else if(tb_is_bwb(e->scenario)){stable=e->loop_stage>=2&&dist<28&&speed>14.5&&tilt<0.78;required=4;}
    else stable=dist<1.5&&speed<1.5&&tilt<0.28;
    e->stable_steps=stable?e->stable_steps+1:(e->stable_steps>0?e->stable_steps-1:0);
    if(stable)reward+=0.12;
    if(e->stable_steps>=required){e->terminated=1;e->outcome=1;reward+=25;}
    else if(e->time>=((e->scenario==4||e->scenario==6)?30:24)){e->truncated=1;e->outcome=5;}
    e->reward[0]=(float)reward;
}

void tb_sim_step_env(TbEnv *e,const TbConfig *c,const float *action,const float *opp_action) {
    if(e->terminated||e->truncated){e->reward[0]=0;e->reward[1]=0;return;}
    e->event_count=0;
    int arena=tb_is_arena(e->scenario),bwb=tb_is_bwb(e->scenario);
    float fallback[TB_ACTION_DIM];
    if(arena&&!opp_action){
        const TbAircraft *opp=&e->aircraft[1];
        memset(fallback,0,sizeof(fallback));
        if(bwb){fallback[0]=fallback[1]=0.54f;}
        else {
            double mass=c->dry_mass-3+opp->fuel+opp->ammo*c->ball_mass;
            double hover=(mass*TB_GRAVITY/4-c->max_thrust*c->thrust_factor*0.08)/(c->max_thrust*c->thrust_factor*0.92);
            TbVec3 err=tb_sub(e->aircraft[0].pos,opp->pos);
            double thr=tb_clamp(hover+0.025*err.z-0.035*opp->vel.z,0,1);
            for(int j=0;j<4;j++)fallback[j]=(float)thr;
            double vx=tb_clamp(0.05*err.x-0.08*opp->vel.x,-0.35,0.35);
            double vy=tb_clamp(0.05*err.y-0.08*opp->vel.y,-0.35,0.35);
            fallback[8]=(float)vx;fallback[9]=(float)vy;fallback[10]=(float)-vx;fallback[11]=(float)-vy;
        }
        fallback[12]=(float)(bwb?0:TB_PI/4);
        opp_action=fallback;
    }
    for(int p=0;p<(arena?2:1);p++){
        TbAircraft *a=&e->aircraft[p];const float *u=p?opp_action:action;
        a->cooldown=fmax(0,a->cooldown-c->dt);
        double az_center=bwb?0:TB_PI/4;
        double az_target=tb_clamp(u[12],az_center-TB_PI/6,az_center+TB_PI/6),el_target=tb_clamp(u[13],-TB_PI/6,TB_PI/6);
        a->cannon_az+=tb_clamp(az_target-a->cannon_az,-(bwb?3.2:1)*c->dt,(bwb?3.2:1)*c->dt);
        a->cannon_el+=tb_clamp(el_target-a->cannon_el,-(bwb?2.4:0.8)*c->dt,(bwb?2.4:0.8)*c->dt);
        if(arena&&u[14]>0.5&&a->cooldown<=0)fire_ball(e,c,p);
        if(bwb)advance_bwb(e,c,p,u);else advance_quad(e,c,p,u);
        if(arena){double dome=bwb?300:c->dome_radius,r=hypot(a->pos.x,a->pos.y);if(r>dome-0.5||a->pos.z>2*dome){a->stuck=1;a->boundary_contacts++;a->vel=tb_v(0,0,0);a->omega=tb_v(0,0,0);a->pos.x*=fmin(1,(dome-0.5)/fmax(r,1e-6));a->pos.y*=fmin(1,(dome-0.5)/fmax(r,1e-6));}}
    }
    e->time+=c->dt;e->step_index++;
    if(arena)update_balls(e,c);
    if(!arena){double alpha=1-exp(-c->dt/3.5);e->gust.x+=(tb_normal(&e->rng)*c->gust_sigma-e->gust.x)*alpha;e->gust.y+=(tb_normal(&e->rng)*c->gust_sigma-e->gust.y)*alpha;}
    score_env(e,c);
    int players=arena?2:1;
    for(int p=0;p<players;p++){
        if(e->step_index%9==0)tb_sim_render_lidar(e,c,p);
        if(!e->external_cameras&&e->step_index%2==0)tb_sim_render_camera(e,c,p);
    }
}
