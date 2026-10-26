#include "sim_internal.h"
#include <string.h>

static double sphere_hit(TbVec3 o,TbVec3 d,TbVec3 c,double r) {
    TbVec3 oc=tb_sub(o,c);double b=tb_dot(oc,d),disc=b*b-tb_dot(oc,oc)+r*r;
    if(disc<0)return INFINITY;
    double s=sqrt(disc),t=-b-s;
    if(t<=0.05)t=-b+s;
    return t>0.05?t:INFINITY;
}
static double trace(const TbEnv *e,const TbConfig *c,int player,TbVec3 o,TbVec3 d,int *kind) {
    double best=tb_is_bwb(e->scenario)?fmax(c->lidar_max,120):c->lidar_max;*kind=-1;
    if(d.z<-1e-5){double t=(0-o.z)/d.z;if(t>0.05&&t<best){best=t;*kind=0;}}
    if(e->scenario==10&&d.z<-1e-5){double t=(9-o.z)/d.z;if(t>0.05&&t<best&&o.x+t*d.x>=20){best=t;*kind=0;}}
    for(int i=0;i<e->obstacle_count;i++){double t=sphere_hit(o,d,e->obstacles[i].center,e->obstacles[i].radius);if(t<best){best=t;*kind=1;}}
    if(tb_is_arena(e->scenario)){
        const TbAircraft *opp=&e->aircraft[1-player];
        double t=sphere_hit(o,d,opp->pos,tb_is_bwb(e->scenario)?0.75:0.45);if(t<best){best=t;*kind=2;}
        double dome=tb_is_bwb(e->scenario)?300:c->dome_radius;
        double a=d.x*d.x+d.y*d.y,b=2*(o.x*d.x+o.y*d.y),cc=o.x*o.x+o.y*o.y-dome*dome;
        double disc=b*b-4*a*cc;
        if(a>1e-9&&disc>=0){t=(-b+sqrt(disc))/(2*a);if(t>0.05&&t<best){best=t;*kind=3;}}
    }
    for(int i=0;i<e->ball_count;i++){double t=sphere_hit(o,d,e->balls[i].pos,0.25);if(t<best){best=t;*kind=4;}}
    return best;
}

void tb_sim_render_lidar(TbEnv *e,const TbConfig *c,int player) {
    TbAircraft *a=&e->aircraft[player];double max_range=tb_is_bwb(e->scenario)?fmax(120,c->lidar_max):c->lidar_max;
    uint64_t seed=e->rng^(uint64_t)(e->step_index*UINT64_C(0x9e3779b97f4a7c15))^(uint64_t)(player*257);
    for(int h=0;h<2;h++){
        TbVec3 origin=tb_add(a->pos,tb_rotate(a->quat,tb_v(0.4,0.4,h? -0.4:0.4)));
        int valid=0;
        for(int el=0;el<16;el++){
            double elev=((h?-1:1)*90.0*el/15.0)*TB_PI/180.0,ce=cos(elev),se=sin(elev);
            for(int az=0;az<128;az++){
                int bin=h*2048+el*128+az;double angle=(45.0+360.0*az/128.0)*TB_PI/180.0;
                TbVec3 rb=tb_v(ce*cos(angle),ce*sin(angle),se);
                uint64_t hash=tb_rand64(&seed);
                if((double)(hash&0xffff)/65536.0<0.005||(rb.x< -0.15&&fabs(rb.z)<0.28&&fabs(rb.y)<0.28)){e->lidar[player][bin]=-1;continue;}
                TbVec3 rw=tb_rotate(a->quat,rb);int kind=0;double range=trace(e,c,player,origin,rw,&kind);
                double n=((double)((hash>>16)&0xffff)/32768.0-1.0)*0.02;
                range=tb_clamp(range+n,0.05,max_range);
                e->lidar[player][bin]=(float)range;valid++;
            }
        }
        e->lidar_capture[player][h]=e->time;
        e->lidar_valid[player][h]=(float)valid/2048;
    }
}

static TbVec3 camera_ray(int view,double u,double v) {
    TbVec3 f=tb_unit(tb_v(1,1,0)),r=tb_unit(tb_v(1,-1,0));
    if(view==0){double t=tan(45.0*TB_PI/360.0);return tb_unit(tb_add(f,tb_add(tb_scale(r,u*t),tb_v(0,0,-v*t))));}
    double radius=hypot(u,v),theta=tb_clamp(radius,0,1.4)*140.0*TB_PI/360.0;
    if(view==1){if(radius<1e-12)return f;TbVec3 tangent=tb_add(tb_scale(r,u/radius),tb_v(0,0,-v/radius));return tb_unit(tb_add(tb_scale(f,cos(theta)),tb_scale(tangent,sin(theta))));}
    double phi=atan2(v,u);return tb_unit(tb_v(sin(theta)*cos(phi),-sin(theta)*sin(phi),view==3?cos(theta):-cos(theta)));
}
static uint8_t shade(double distance,int kind,TbVec3 dir) {
    double luma=0.22+0.15*tb_clamp(dir.z,-1,1);
    if(kind==0)luma=0.28+0.17*tb_clamp(-dir.z,0,1);
    if(kind==1)luma=0.65;
    if(kind==2)luma=0.90;
    if(kind==3)luma=0.46;
    if(kind==4)luma=1.0;
    luma*=0.75+0.25*exp(-distance/80);
    return (uint8_t)tb_clamp(round(luma*255),0,255);
}
void tb_sim_render_camera(TbEnv *e,const TbConfig *c,int player) {
    TbAircraft *a=&e->aircraft[player];
    for(int view=0;view<4;view++){
        TbVec3 mount=view<2?tb_v(0.42,0.42,view==0?0.13:0.08):tb_v(0,0,view==2? -0.5:0.5);
        TbVec3 origin=tb_add(a->pos,tb_rotate(a->quat,mount));
        for(int y=0;y<64;y++)for(int x=0;x<64;x++){
            double u=(x+0.5-32)/32,v=(y+0.5-32)/32;
            TbVec3 dir=tb_rotate(a->quat,camera_ray(view,u,v));int kind=0;double d=trace(e,c,player,origin,dir,&kind);
            uint8_t px=shade(d,kind,dir);e->camera[player][view*4096+y*64+x]=px;
            if(e->capture_raw){for(int dy=0;dy<3;dy++)for(int dx=0;dx<3;dx++)e->raw_camera[player][view*36864+(3*y+dy)*192+3*x+dx]=px;}
        }
    }
    e->camera_capture[player]=e->time;if(e->capture_raw)e->raw_capture[player]=e->time;
}

void tb_sim_scene(const TbEnv *e,const TbConfig *c,int player,double *out,int16_t *noise) {
    memset(out,0,TB_SCENE_DOUBLES*sizeof(double));
    const TbAircraft *a=&e->aircraft[player];TbVec3 goal=tb_sim_target(e,player);
    out[0]=e->scenario;out[1]=e->time;out[2]=goal.x;out[3]=goal.y;out[4]=goal.z;
    out[5]=a->pos.x;out[6]=a->pos.y;out[7]=a->pos.z;
    out[8]=a->quat.w;out[9]=a->quat.x;out[10]=a->quat.y;out[11]=a->quat.z;
    out[12]=45;out[13]=140;out[14]=tb_is_bwb(e->scenario)?300:c->dome_radius;
    int count=0;
    for(int i=0;i<e->obstacle_count&&count<160;i++,count++){
        double *s=out+17+5*count;s[0]=e->obstacles[i].center.x;s[1]=e->obstacles[i].center.y;s[2]=e->obstacles[i].center.z;s[3]=e->obstacles[i].radius;s[4]=0;
    }
    if(tb_is_arena(e->scenario)&&count<160){const TbAircraft *b=&e->aircraft[1-player];double *s=out+17+5*count++;s[0]=b->pos.x;s[1]=b->pos.y;s[2]=b->pos.z;s[3]=0.45;s[4]=1;}
    for(int i=0;i<e->ball_count&&count<160;i++,count++){double *s=out+17+5*count;s[0]=e->balls[i].pos.x;s[1]=e->balls[i].pos.y;s[2]=e->balls[i].pos.z;s[3]=0.25;s[4]=2;}
    out[15]=count;
    if(noise){uint64_t state=e->rng^(uint64_t)e->step_index;for(int i=0;i<TB_CAMERA_PIXELS;i++)noise[i]=(int16_t)((int)(tb_rand64(&state)&255)-128);}
}

void tb_sim_observe(const TbEnv *e,const TbConfig *c,int player,float *out) {
    memset(out,0,TB_OBSERVATION_DIM*sizeof(float));const TbAircraft *a=&e->aircraft[player];
    uint64_t state=e->rng^(uint64_t)(e->step_index*UINT64_C(0xbf58476d1ce4e5b9))^(uint64_t)player;
    out[0]=(float)(a->pos.x+0.15*tb_normal(&state));out[1]=(float)(a->pos.y+0.15*tb_normal(&state));out[2]=(float)(a->pos.z+0.08*tb_normal(&state));
    out[3]=(float)(a->vel.x+0.08*tb_normal(&state));out[4]=(float)(a->vel.y+0.08*tb_normal(&state));out[5]=(float)(a->vel.z+0.08*tb_normal(&state));
    out[6]=(float)a->quat.w;out[7]=(float)a->quat.x;out[8]=(float)a->quat.y;out[9]=(float)a->quat.z;
    out[10]=(float)(a->omega.x+0.004*tb_normal(&state));out[11]=(float)(a->omega.y+0.004*tb_normal(&state));out[12]=(float)(a->omega.z+0.004*tb_normal(&state));
    out[13]=(float)(a->specific_force.x+0.06*tb_normal(&state));out[14]=(float)(a->specific_force.y+0.06*tb_normal(&state));out[15]=(float)(a->specific_force.z+0.06*tb_normal(&state));
    for(int i=0;i<4;i++){
        out[16+i]=(float)(a->spool[i]*(a->degraded_engine==i?a->degraded_factor:1));
        out[20+i]=(float)sin(a->outer[i]);out[24+i]=(float)cos(a->outer[i]);out[28+i]=(float)a->inner[i];
        out[32+i]=(float)a->outer_rate[i];out[36+i]=(float)a->inner_rate[i];
    }
    out[40]=(float)(a->fuel/fmax(1e-6,tb_is_bwb(e->scenario)?2.85:c->fuel_capacity));out[41]=(float)a->payload_attached;
    out[42]=tb_uniform(&state)<0.01?0:1;out[43]=1;
    out[44]=(float)a->cannon_az;out[45]=(float)a->cannon_el;out[46]=(float)(a->cooldown/fmax(1e-6,c->shot_cooldown));out[47]=(float)a->ammo/(float)(c->starting_ammunition>0?c->starting_ammunition:1);
    memcpy(out+48,e->lidar[player],TB_LIDAR_BINS*sizeof(float));
    out[4144]=(float)fmax(0,e->time-e->lidar_capture[player][0]);out[4145]=(float)fmax(0,e->time-e->lidar_capture[player][1]);
    out[4146]=e->lidar_valid[player][0];out[4147]=e->lidar_valid[player][1];
}

void tb_sim_truth(const TbEnv *e,const TbConfig *c,double *out) {
    memset(out,0,TB_TRUTH_DIM*sizeof(double));const TbAircraft *a=&e->aircraft[0],*b=&e->aircraft[1];
    out[0]=e->time;out[1]=a->pos.x;out[2]=a->pos.y;out[3]=a->pos.z;out[4]=a->vel.x;out[5]=a->vel.y;out[6]=a->vel.z;
    out[7]=a->quat.w;out[8]=a->quat.x;out[9]=a->quat.y;out[10]=a->quat.z;out[11]=a->omega.x;out[12]=a->omega.y;out[13]=a->omega.z;
    for(int i=0;i<4;i++){out[14+i]=a->spool[i];out[18+i]=a->outer[i];out[22+i]=a->inner[i];out[26+i]=a->outer_rate[i];out[30+i]=a->inner_rate[i];}
    out[34]=a->fuel;out[35]=a->payload_attached;out[36]=a->payload_delivered;out[37]=a->jammed_arm;
    out[38]=a->cannon_az;out[39]=a->cannon_el;out[40]=a->cooldown;out[41]=a->ammo;out[42]=a->stuck;out[43]=a->hits;out[44]=a->misses;out[45]=a->self_hits;out[46]=a->shots;
    out[47]=b->pos.x;out[48]=b->pos.y;out[49]=b->pos.z;out[50]=b->vel.x;out[51]=b->vel.y;out[52]=b->vel.z;
    out[53]=b->quat.w;out[54]=b->quat.x;out[55]=b->quat.y;out[56]=b->quat.z;out[57]=b->omega.x;out[58]=b->omega.y;out[59]=b->omega.z;
    out[60]=b->cannon_az;out[61]=b->cannon_el;out[62]=b->cooldown;out[63]=b->ammo;out[64]=b->stuck;out[65]=b->hits;out[66]=b->misses;out[67]=b->self_hits;out[68]=b->shots;
    TbVec3 wind=tb_is_arena(e->scenario)?tb_v(0,0,0):c->wind,gust=tb_is_arena(e->scenario)?tb_v(0,0,0):e->gust;
    out[69]=wind.x;out[70]=wind.y;out[71]=wind.z;out[72]=gust.x;out[73]=gust.y;out[74]=gust.z;
    out[75]=e->drag.x;out[76]=e->drag.y;out[77]=e->drag.z;out[78]=e->ball_count;out[79]=e->winner;out[80]=e->outcome;out[81]=e->terminated;
    TbVec3 goal=tb_sim_target(e,0);out[82]=goal.x;out[83]=goal.y;out[84]=goal.z;
}

void tb_sim_privileged(const TbEnv *e,const TbConfig *c,float *out) {
    memset(out,0,TB_PRIVILEGED_DIM*sizeof(float));const TbAircraft *a=&e->aircraft[0],*b=&e->aircraft[1];TbVec3 goal=tb_sim_target(e,0),d=tb_sub(goal,a->pos);
    out[0]=(float)tanh(a->pos.x/25);out[1]=(float)tanh(a->pos.y/25);out[2]=(float)tanh((a->pos.z-8)/10);
    out[3]=(float)tanh(a->vel.x/10);out[4]=(float)tanh(a->vel.y/10);out[5]=(float)tanh(a->vel.z/10);
    out[6]=(float)a->quat.w;out[7]=(float)a->quat.x;out[8]=(float)a->quat.y;out[9]=(float)a->quat.z;
    out[10]=(float)tanh(a->omega.x/4);out[11]=(float)tanh(a->omega.y/4);out[12]=(float)tanh(a->omega.z/4);
    out[13]=(float)tanh(d.x/15);out[14]=(float)tanh(d.y/15);out[15]=(float)tanh(d.z/10);out[16]=(float)tanh(a->pos.z/8);
    if(!tb_is_arena(e->scenario)){out[17]=(float)tanh((c->wind.x+e->gust.x)/8);out[18]=(float)tanh((c->wind.y+e->gust.y)/8);}
    for(int i=0;i<4;i++){out[19+i]=(float)(a->jammed_arm==i);out[23+i]=(float)(a->degraded_engine==i);out[56+i]=(float)(a->outer_torque[i]/72);out[60+i]=(float)a->outer_thermal[i];}
    out[27]=(float)a->degraded_factor;
    if(tb_is_arena(e->scenario)){TbVec3 p=tb_inverse_rotate(a->quat,tb_sub(b->pos,a->pos)),v=tb_inverse_rotate(a->quat,tb_sub(b->vel,a->vel));out[29]=(float)tanh(p.x/20);out[30]=(float)tanh(p.y/20);out[31]=(float)tanh(p.z/12);out[32]=(float)tanh(v.x/10);out[33]=(float)tanh(v.y/10);out[34]=(float)tanh(v.z/10);out[35]=(float)tanh(tb_norm(tb_sub(b->pos,a->pos))/30);out[36]=(float)b->stuck;out[37]=(float)a->stuck;out[38]=(float)(a->hits-b->hits)*0.25f;out[39]=(float)(b->cooldown/fmax(1e-6,c->shot_cooldown));out[40]=(float)b->ammo/fmax(1,c->starting_ammunition);}
    for(int i=0;i<2&&i<e->ball_count;i++){TbVec3 p=tb_inverse_rotate(a->quat,tb_sub(e->balls[i].pos,a->pos)),v=tb_inverse_rotate(a->quat,tb_sub(e->balls[i].vel,a->vel));int k=41+i*6;out[k]=(float)tanh(p.x/15);out[k+1]=(float)tanh(p.y/15);out[k+2]=(float)tanh(p.z/15);out[k+3]=(float)tanh(v.x/25);out[k+4]=(float)tanh(v.y/25);out[k+5]=(float)tanh(v.z/25);}
    double near=25;TbVec3 nearest=tb_v(25,0,0);for(int i=0;i<e->obstacle_count;i++){TbVec3 p=tb_sub(e->obstacles[i].center,a->pos);double d0=tb_norm(p)-e->obstacles[i].radius;if(d0<near){near=d0;nearest=tb_inverse_rotate(a->quat,p);}}
    out[53]=(float)tanh(near/10);out[54]=(float)tanh(nearest.x/12);out[55]=(float)tanh(nearest.y/12);
}

void tb_sim_reference(const TbEnv *e,const TbConfig *c,const double *target,float *out) {
    memset(out,0,TB_ACTION_DIM*sizeof(float));const TbAircraft *a=&e->aircraft[0];
    TbVec3 g=target?tb_v(target[0],target[1],target[2]):tb_sim_target(e,0),err=tb_sub(g,a->pos);
    if(tb_is_bwb(e->scenario)){
        const int arena=tb_is_arena(e->scenario);
        TbVec3 bx=tb_rotate(a->quat,tb_v(1,0,0)),by=tb_rotate(a->quat,tb_v(0,1,0)),bz=tb_rotate(a->quat,tb_v(0,0,1));
        TbVec3 vb=tb_inverse_rotate(a->quat,a->vel);
        if(arena){
            double r=hypot(a->pos.x,a->pos.y),safe=fmax(60,300-65);
            if(r>safe){double pull=tb_clamp((r-safe)/45,0,0.85);g.x*=(1-pull);g.y*=(1-pull);}
            g.z=tb_clamp(g.z,28,135);
        }
        err=tb_sub(g,a->pos);
        double desired_yaw=atan2(err.y,err.x),current_yaw=atan2(bx.y,bx.x);
        double ground_speed=hypot(a->vel.x,a->vel.y),course_yaw=ground_speed>2.5?atan2(a->vel.y,a->vel.x):current_yaw;
        double course_err=atan2(sin(desired_yaw-course_yaw),cos(desired_yaw-course_yaw));
        double desired_speed=e->scenario==25?18.5:(arena&&tb_norm(err)>38?26:24);
        double alt_err=g.z-a->pos.z,base=e->scenario==25?0.36:0.44;
        double coll=tb_clamp(base+0.065*(desired_speed-vb.x)+0.018*alt_err,0.12,e->scenario>=25?0.78:0.92);
        double roll=atan2(by.z,bz.z),desired_roll=tb_clamp(-1.35*course_err,-0.55,0.55);
        double diff=tb_clamp(0.90*(desired_roll-roll)-0.34*a->omega.x,-0.45,0.45);
        out[0]=(float)tb_clamp(coll+diff,0.05,1);
        out[1]=(float)tb_clamp(coll-diff,0.05,1);
        out[2]=out[0];out[3]=out[1];
        double pitch=asin(tb_clamp(bx.z,-1,1)),climb_rate=tb_clamp(0.42*alt_err,-5,5);
        double cos_bank=cos(roll),sin_bank=sin(roll),gamma=atan2(climb_rate,fmax(14,ground_speed));
        double desired_pitch=tb_clamp(0.062831853+0.75*gamma+0.035*(climb_rate-a->vel.z)+0.09*(1-fmax(0.5,cos_bank)),-0.15,0.25);
        if(vb.x<15.5)desired_pitch=fmin(desired_pitch,0.02);
        double theta_dot=2.20*(desired_pitch-pitch);
        double psi_dot=(TB_GRAVITY/fmax(12,vb.x))*tan(tb_clamp(-roll,-0.75,0.75))+0.35*course_err;
        double q_up=cos_bank*theta_dot-sin_bank*psi_dot,r_left=sin_bank*theta_dot+cos_bank*psi_dot;
        out[8]=(float)tb_clamp(0.85*(q_up+a->omega.y)+0.03,-TB_PI/6,TB_PI/6);
        out[9]=(float)tb_clamp(-0.65*(r_left-a->omega.z)-0.025*vb.y,-TB_PI/6,TB_PI/6);
        if(arena){
            TbVec3 aim=tb_sub(tb_sim_target(e,0),tb_add(a->pos,tb_scale(a->vel,0.1)));
            double d=fmax(1,tb_norm(aim)),flight=d/75;aim.z+=0.5*TB_GRAVITY*flight*flight;
            TbVec3 ab=tb_inverse_rotate(a->quat,aim);
            double az=tb_clamp(atan2(ab.y,fmax(0.2,ab.x)),-TB_PI/6,TB_PI/6);
            double el=tb_clamp(atan2(ab.z,fmax(0.2,hypot(ab.x,ab.y))),-TB_PI/6,TB_PI/6);
            out[12]=(float)az;out[13]=(float)el;
            out[14]=(a->cooldown<0.02&&a->ammo>0&&hypot(az-a->cannon_az,el-a->cannon_el)<0.045&&d<165&&ab.x>0)?1:0;
        }
        return;
    }
    double mass=c->dry_mass+a->fuel+(a->payload_attached?c->payload_mass:0)+a->ammo*c->ball_mass;
    double hover=(mass*TB_GRAVITY/4-c->max_thrust*c->thrust_factor*0.08)/(c->max_thrust*c->thrust_factor*0.92);
    double thr=tb_clamp(hover+0.025*err.z-0.035*a->vel.z,0,1);
    for(int i=0;i<4;i++)out[i]=(float)thr;
    double vx=tb_clamp(0.05*err.x-0.08*a->vel.x,-0.35,0.35),vy=tb_clamp(0.05*err.y-0.08*a->vel.y,-0.35,0.35);
    out[8]=(float)vx;out[9]=(float)vy;out[10]=(float)-vx;out[11]=(float)-vy;
    out[12]=(float)(tb_is_arena(e->scenario)?TB_PI/4:0);
}
