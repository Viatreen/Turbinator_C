#ifndef TURBINATOR_ONBOARD_FEATURES_CUH
#define TURBINATOR_ONBOARD_FEATURES_CUH

#include <math.h>

__device__ __forceinline__ float tb_of_clamp(float x, float lo, float hi) {
    if (isnan(x)) return x;
    return fminf(hi, fmaxf(lo, x));
}
__device__ __forceinline__ float tb_of_obs(float x) {
    return isnan(x) ? 0.0f : (isinf(x) ? copysignf(1000.0f, x) : x);
}
__device__ __forceinline__ float tb_of_mission(float x) {
    return isnan(x) ? 0.0f : (isinf(x) ? copysignf(1.0f, x) : x);
}
__device__ __forceinline__ float tb_of_wrap(float target, float current) {
    float d = target - current;
    return atan2f(sinf(d), cosf(d));
}
__device__ __forceinline__ float tb_of_hypot3(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}

/* The feature order is the 24-entry onboard tactical tensor contract. */
__device__ __forceinline__ void tb_onboard_tactical(const float *o, const float *m,
                                                     float t[24]) {
    const float pi = 3.14159265358979323846f;
    float spool[4], thrust[4], fz[4], tau_z = 0.0f, spool_mean = 0.0f;
    for (int i = 0; i < 4; ++i) {
        spool[i] = tb_of_clamp(o[16 + i], 0.0f, 1.0f);
        thrust[i] = 0.036f + 0.964f * spool[i];
        float ci = cosf(o[28 + i]);
        fz[i] = thrust[i] * o[24 + i] * ci;
        tau_z += thrust[i] * o[20 + i] * ci;
        spool_mean += spool[i];
    }
    spool_mean *= 0.25f;
    t[0] = fz[1] - fz[3];
    t[1] = fz[2] - fz[0];
    t[2] = 0.5f * tau_z;
    t[3] = 0.25f * (fz[0] + fz[1] + fz[2] + fz[3]) - 0.5f;
    for (int i = 0; i < 4; ++i) t[4 + i] = 2.0f * (spool[i] - spool_mean);
    float qw = o[6], qx = o[7], qy = o[8], qz = o[9];
    float bz_x = 2.0f * (qx * qz + qw * qy);
    float bz_y = 2.0f * (qy * qz - qw * qx);
    float ct = tb_of_clamp(1.0f - 2.0f * (qx * qx + qy * qy), -1.0f, 1.0f);
    float bx_x = 1.0f - 2.0f * (qy * qy + qz * qz);
    float bx_y = 2.0f * (qx * qy + qw * qz);
    float yaw = atan2f(bx_y, bx_x), cy = cosf(yaw), sy = sinf(yaw);
    t[8] = cy * bz_y - sy * bz_x;
    t[9] = -sy * bz_y - cy * bz_x;
    t[10] = ct;
    t[11] = ct < 0.20f ? 1.0f : 0.0f;
    float rx = atanhf(tb_of_clamp(m[7], -0.995f, 0.995f)) * 20.0f;
    float ry = atanhf(tb_of_clamp(m[8], -0.995f, 0.995f)) * 20.0f;
    float rz = atanhf(tb_of_clamp(m[9], -0.995f, 0.995f)) * 20.0f;
    t[12] = tanhf((cy * rx + sy * ry) * 0.08f);
    t[13] = tanhf((-sy * rx + cy * ry) * 0.08f);
    t[14] = tanhf(rz * 0.12f);
    t[15] = tanhf(hypotf(rx, ry) * 0.05f);
    t[16] = tanhf((cy * o[3] + sy * o[4]) * 0.15f);
    t[17] = tanhf((-sy * o[3] + cy * o[4]) * 0.15f);
    t[18] = expf(-fmaxf(o[2] - 0.62f, 0.0f));
    float dist = fmaxf(tb_of_hypot3(rx, ry, rz), 1.0f);
    float flight = dist / 20.0f;
    float wx = rx - o[3] * 0.5f * flight;
    float wy = ry - o[4] * 0.5f * flight;
    float wz = rz - o[5] * 0.5f * flight + 0.5f * 9.80665f * flight * flight;
    float ux = -qx, uy = -qy, uz = -qz;
    float cx = uy * wz - uz * wy + qw * wx;
    float cyq = uz * wx - ux * wz + qw * wy;
    float cz = ux * wy - uy * wx + qw * wz;
    float bx = wx + 2.0f * (uy * cz - uz * cyq);
    float by = wy + 2.0f * (uz * cx - ux * cz);
    float bz = wz + 2.0f * (ux * cyq - uy * cx);
    float az = tb_of_clamp(atan2f(by, bx), pi / 12.0f, pi * 5.0f / 12.0f);
    float el = tb_of_clamp(atan2f(bz, fmaxf(hypotf(bx, by), 0.2f)), -pi / 6.0f, pi / 6.0f);
    float err = hypotf(az - o[44], el - o[45]);
    t[19] = (az - pi * 0.25f) / (pi / 6.0f);
    t[20] = el / (pi / 6.0f);
    t[21] = tanhf(err * 2.0f);
    t[22] = (o[46] <= 0.02f && o[47] > 0.01f && err < 0.26f && dist < 34.0f) ? 1.0f : -1.0f;
    t[23] = sinf(atan2f(ry, rx) - pi * 0.25f - yaw);
    for (int i = 0; i < 24; ++i)
        t[i] = isnan(t[i]) ? 0.0f : (isinf(t[i]) ? copysignf(1.0f, t[i]) : t[i]);
}

__device__ __forceinline__ void tb_onboard_prior(const float *input_obs,
                                                  const float *input_mission,
                                                  float n[15]) {
    const float pi = 3.14159265358979323846f;
    const float max_installed = 294.2f * 0.85f;
    const float idle_frac = 0.08f;
    const float min_thrust = max_installed * idle_frac;
    const float gravity = 9.80665f;
    const float outer_limit = 2.0f;
    const float inner_limit = pi / 4.0f;
    const float cannon_az_ctr = pi / 4.0f;
    const float cannon_az_half = pi / 6.0f;
    const float cannon_el_lim = pi / 6.0f;
    float o[48], m[10];
    for (int i = 0; i < 48; ++i) o[i] = tb_of_obs(input_obs[i]);
    for (int i = 0; i < 10; ++i) m[i] = tb_of_mission(input_mission[i]);
    float px = o[0], py = o[1], pz = o[2], vx = o[3], vy = o[4], vz = o[5];
    float qnorm = fmaxf(tb_of_hypot3(o[6], o[7], o[8]), 0.0f);
    qnorm = fmaxf(sqrtf(qnorm * qnorm + o[9] * o[9]), 1.0e-8f);
    float sign = o[6] < 0.0f ? -1.0f : 1.0f;
    float qw = sign * o[6] / qnorm, qx = sign * o[7] / qnorm;
    float qy = sign * o[8] / qnorm, qz = sign * o[9] / qnorm;
    float ox = o[10], oy = o[11], oz = o[12];
    float omega_norm = tb_of_hypot3(ox, oy, oz);
    float spool[4], thrust[4], outer[4], inner[4], fz[4], ft[4];
    float total_fz = 0.0f, jammed_yaw_ft = 0.0f;
    for (int i = 0; i < 4; ++i) {
        spool[i] = tb_of_clamp(o[16 + i], 0.0f, 1.0f);
        thrust[i] = max_installed * (idle_frac + (1.0f - idle_frac) * spool[i]);
        outer[i] = atan2f(o[20 + i], o[24 + i]);
        inner[i] = o[28 + i];
        fz[i] = thrust[i] * cosf(outer[i]) * cosf(inner[i]);
        ft[i] = thrust[i] * sinf(outer[i]) * cosf(inner[i]);
        total_fz += fz[i];
        jammed_yaw_ft += ft[i];
    }
    bool landing = m[1] > 0.75f;
    bool cargo = m[2] > 0.75f;
    bool inversion = m[3] > 0.75f && m[0] < -0.25f;
    bool turbine_out = m[3] > 0.75f && m[2] < -0.25f;
    bool lateral = m[4] > 0.75f && m[5] > 0.25f;
    bool slalom = m[4] > 0.75f && !lateral;
    bool dodge = m[5] > 0.75f;
    bool arena = m[6] > 0.75f;
    float fuel = tb_of_clamp(o[40], 0.0f, 1.0f) * 9.0f;
    float ammo_mass = tb_of_clamp(o[47], 0.0f, 4.0f) * 6.0f;
    bool payload = o[41] > 0.5f;
    float mass = (arena ? 40.0f : 43.0f) + fuel + ammo_mass + (payload ? 30.0f : 0.0f);
    float weight = mass * gravity;
    float r00 = 1.0f - 2.0f * (qy * qy + qz * qz);
    float r01 = 2.0f * (qx * qy - qw * qz);
    float r02 = 2.0f * (qx * qz + qw * qy);
    float r10 = 2.0f * (qx * qy + qw * qz);
    float r11 = 1.0f - 2.0f * (qx * qx + qz * qz);
    float r12 = 2.0f * (qy * qz - qw * qx);
    float r20 = 2.0f * (qx * qz - qw * qy);
    float r21 = 2.0f * (qy * qz + qw * qx);
    float r22 = 1.0f - 2.0f * (qx * qx + qy * qy);
    /* The prior uses body-z components with a positive qw*qy term. */
    float body_z_x = 2.0f * (qx * qz + qw * qy);
    float body_z_y = 2.0f * (qy * qz - qw * qx);
    float ct = tb_of_clamp(r22, -1.0f, 1.0f);
    float yaw = atan2f(r10, r00);
    float tau_x_actual = fz[1] - fz[3];
    float tau_y_actual = fz[2] - fz[0];
    float az_actual = (total_fz * ct - weight) / mass;
    bool heavy_cargo = cargo && payload;
    bool landing_like = landing || heavy_cargo;
    float relx = atanhf(tb_of_clamp(m[7], -0.9995f, 0.9995f)) * 20.0f;
    float rely = atanhf(tb_of_clamp(m[8], -0.9995f, 0.9995f)) * 20.0f;
    float relz = atanhf(tb_of_clamp(m[9], -0.9995f, 0.9995f)) * 20.0f;
    float tx = (arena || lateral) ? px + relx : 0.0f;
    if (cargo) tx = 15.0f;
    if (slalom) tx = 36.0f;
    float ty = (arena || lateral) ? py + rely : 0.0f;
    float tz = 8.0f;
    if (landing) tz = 0.60f;
    if (inversion) tz = 14.0f;
    if (cargo || arena) tz = pz + relz;
    float ex = tx, ey = ty, ez = tz;
    float dx_opp = tx - px, dy_opp = ty - py;
    float dxy = hypotf(dx_opp, dy_opp);
    bool valid_d = dxy > 1.0e-3f;
    float safe_d = valid_d ? dxy : 1.0f;
    bool intercept = arena && m[0] > 0.75f && m[5] < -0.25f;
    float desired_range = intercept ? 8.5f : 12.5f;
    float ax = valid_d ? tx - (dx_opp / safe_d) * desired_range : tx;
    float ay = valid_d ? ty - (dy_opp / safe_d) * desired_range : ty;
    float scale = fminf(16.0f / fmaxf(hypotf(ax, ay), 1.0e-3f), 1.0f);
    ax *= scale; ay *= scale;
    float pull = tb_of_clamp((hypotf(px, py) - 21.0f) / 10.0f, 0.0f, 1.0f);
    ax *= 1.0f - pull; ay *= 1.0f - pull;
    float az = pz < 8.0f || vz < -1.0f ? 12.2f : tb_of_clamp(tz, 8.5f, 12.2f);
    if (arena) { ex = ax; ey = ay; ez = az; }
    float errx = ex - px, erry = ey - py;
    float horiz_dist = hypotf(errx, erry);
    float avoid_x = 0.0f, avoid_y = 0.0f, slalom_speed = 3.8f, nearest_dx = 30.0f;
    float min_r = 30.0f, min_r_dx = 0.0f, min_r_dy = 0.0f;
    float min_dx_c = 30.0f, cluster_y_sum = 0.0f;
    int cluster_count = 0;
    /* First pass is the exact 1536 selected rays used by the controller. */
    for (int head = 0; head < 2; ++head) for (int el = 0; el < 6; ++el)
        for (int azi = 0; azi < 128; ++azi) {
            int index = 48 + (head * 16 + el) * 128 + azi;
            float range = tb_of_obs(input_obs[index]);
            if (!(range > 0.35f && range < 11.5f)) continue;
            float elev = (head == 0 ? 1.0f : -1.0f) * (pi * el / 30.0f);
            float azim = pi * (0.25f + azi / 64.0f);
            float rx = cosf(elev) * cosf(azim), ry = cosf(elev) * sinf(azim);
            float rz = sinf(elev);
            float rwx = r00 * rx + r01 * ry + r02 * rz;
            float rwy = r10 * rx + r11 * ry + r12 * rz;
            float rwz = r20 * rx + r21 * ry + r22 * rz;
            float wx = rwx * range, wy = rwy * range, wz = rwz * range;
            if (el < 4 && range < min_r) { min_r = range; min_r_dx = wx; min_r_dy = wy; }
            float x_center = wx + rwx * 1.10f;
            float y_center = py + wy + rwy * 1.10f;
            if (fabsf(wz) <= 2.1f && x_center >= -0.35f && x_center <= 10.5f &&
                fabsf(y_center) <= 3.5f && x_center < min_dx_c) min_dx_c = x_center;
        }
    if (dodge && min_r < 7.5f) {
        float min_az = atan2f(min_r_dy, min_r_dx);
        avoid_x = -sinf(min_az) * 2.2f;
        avoid_y = cosf(min_az) * 2.2f;
    }
    /* A second pass recovers cluster membership relative to the nearest ray. */
    for (int head = 0; head < 2; ++head) for (int el = 0; el < 6; ++el)
        for (int azi = 0; azi < 128; ++azi) {
            int index = 48 + (head * 16 + el) * 128 + azi;
            float range = tb_of_obs(input_obs[index]);
            if (!(range > 0.35f && range < 11.5f)) continue;
            float elev = (head == 0 ? 1.0f : -1.0f) * (pi * el / 30.0f);
            float azim = pi * (0.25f + azi / 64.0f);
            float rx = cosf(elev) * cosf(azim), ry = cosf(elev) * sinf(azim), rz = sinf(elev);
            float rwx = r00 * rx + r01 * ry + r02 * rz;
            float rwy = r10 * rx + r11 * ry + r12 * rz;
            float rwz = r20 * rx + r21 * ry + r22 * rz;
            float wx = rwx * range, wy = rwy * range, wz = rwz * range;
            float x_center = wx + rwx * 1.10f;
            float y_center = py + wy + rwy * 1.10f;
            if (fabsf(wz) <= 2.1f && x_center >= -0.35f && x_center <= 10.5f &&
                fabsf(y_center) <= 3.5f && x_center <= min_dx_c + 1.4f) {
                ++cluster_count;
                cluster_y_sum += y_center;
            }
        }
    float cluster_y = cluster_y_sum / fmaxf((float)cluster_count, 1.0f);
    bool has_slalom_obs = slalom && horiz_dist > 2.5f && min_dx_c < 10.2f && cluster_count > 0;
    if (has_slalom_obs) {
        nearest_dx = min_dx_c;
        ey = cluster_y >= 0.0f ? -1.85f : 1.85f;
    }
    float lane_err = fabsf(py - ey);
    float speed_limit = lane_err > 0.55f ? 2.75f : 4.10f;
    if (lane_err > 1.05f) speed_limit = 1.55f;
    if (min_dx_c >= 6.6f && px > 4.0f) speed_limit = 3.10f;
    if (has_slalom_obs) slalom_speed = speed_limit;
    if (slalom && horiz_dist > 3.5f) {
        float hill = tb_of_clamp((px - 6.0f) / 30.0f, 0.0f, 1.0f);
        ez = 2.5f * sinf(pi * hill) + 8.0f;
    }

    float upx = ex - px, upy = ey - py, upz = ez - pz;
    float up_horiz = hypotf(upx, upy);
    bool active_slalom = slalom && nearest_dx < 10.2f && horiz_dist > 3.0f;

    /* Landing/cargo velocity and acceleration target. */
    float pad_z = fmaxf(ez, 0.60f), above_pad = pz - pad_z;
    float horiz_speed = hypotf(vx, vy);
    float transit_cap = heavy_cargo ? 2.65f : 3.4f;
    float low_cap = above_pad < 1.2f ? 0.55f : (heavy_cargo ? 1.25f : 1.6f);
    float max_vxy_land = up_horiz > 1.8f ? transit_cap : low_cap;
    float kp_land = heavy_cargo ? 0.75f : 1.05f;
    float des_vx_land = tb_of_clamp(kp_land * upx, -max_vxy_land, max_vxy_land);
    float des_vy_land = tb_of_clamp(kp_land * upy, -max_vxy_land, max_vxy_land);
    float vz_land_touch = pz > 0.635f ? -0.16f : 0.0f;
    float vz_land_cargo = tb_of_clamp(1.6f * (0.64f - pz), -0.30f, 0.18f);
    float vz_land_low = cargo ? vz_land_cargo : vz_land_touch;
    float max_desc = cargo ? -0.82f : -1.4f;
    float vz_land_mid = tb_of_clamp(-0.70f * above_pad, max_desc, -0.25f);
    float vz_land_align = tb_of_clamp(1.2f * (pad_z + 0.90f - pz), -0.30f, 0.60f);
    float transit_z = heavy_cargo ? fmaxf(1.2f + 0.28f * up_horiz, 2.0f) : 2.0f;
    float max_desc_trans = heavy_cargo ? -1.05f : -1.6f;
    float vz_land_trans = tb_of_clamp(1.3f * (transit_z - pz), max_desc_trans, 1.4f);
    float des_vz_land = above_pad > 1.05f ? vz_land_mid : vz_land_low;
    if ((up_horiz > 0.85f || horiz_speed > 0.65f) && above_pad < 0.95f)
        des_vz_land = vz_land_align;
    if (up_horiz > 1.8f) des_vz_land = vz_land_trans;
    float max_axy_land = heavy_cargo ? 2.2f : 2.6f;
    float des_ax_land = tb_of_clamp(2.4f * (des_vx_land - vx), -max_axy_land, max_axy_land);
    float des_ay_land = tb_of_clamp(2.4f * (des_vy_land - vy), -max_axy_land, max_axy_land);
    float kp_az_land = heavy_cargo ? 5.0f : 4.5f;
    float kd_az_land = heavy_cargo ? 1.0f : 0.80f;
    float min_az_land = heavy_cargo ? -1.8f : -2.8f;
    float des_az_land = tb_of_clamp(kp_az_land * (des_vz_land - vz) - kd_az_land * az_actual,
                                    min_az_land, 5.5f);

    /* Standard/agile/fault/arena flight target. */
    float inv_max_v = up_horiz > 9.0f && fabsf(upz) < 2.5f && ct > 0.90f ? 6.2f : 4.2f;
    float lat_vx = fabsf(upy) > 1.35f ? 1.45f : 3.8f;
    float max_vx = arena ? 3.2f : 3.0f;
    if (inversion) max_vx = inv_max_v;
    float max_vy = max_vx;
    if (lateral) { max_vx = lat_vx; max_vy = 3.4f; }
    if (slalom) max_vx = 3.5f;
    if (active_slalom) { max_vx = slalom_speed; max_vy = 3.2f; }
    float max_vz = vz < -2.0f || inversion ? 6.5f : 2.6f;
    float kp_xy = inversion ? 0.85f : (lateral ? 0.95f : 0.90f);
    float kp_vx = active_slalom ? 1.25f : kp_xy;
    float kp_vy = active_slalom ? 1.45f : (lateral ? 1.05f : kp_xy);
    float des_vx = tb_of_clamp(kp_vx * upx + avoid_x, -max_vx, max_vx);
    float des_vy = tb_of_clamp(kp_vy * upy + avoid_y, -max_vy, max_vy);
    float des_vz = tb_of_clamp(1.25f * upz, -2.6f, max_vz);
    if ((inversion || arena) && vz < -1.0f) des_vz = fmaxf(des_vz, 0.8f);
    float kp_ax = active_slalom ? 3.0f : 2.4f;
    float kp_ay = active_slalom ? 3.0f : 2.6f;
    float des_ax_std = tb_of_clamp(kp_ax * (des_vx - vx), -3.8f, 3.8f);
    float des_ay_std = tb_of_clamp(kp_ay * (des_vy - vy), -3.8f, 3.8f);
    float max_az = vz < -1.8f || inversion ? 14.0f : 6.5f;
    float des_az_std = tb_of_clamp(3.8f * (des_vz - vz) - 0.80f * az_actual, -4.0f, max_az);
    float des_ax = landing_like ? des_ax_land : des_ax_std;
    float des_ay = landing_like ? des_ay_land : des_ay_std;
    float des_az = landing_like ? des_az_land : des_az_std;

    float tilt = acosf(ct);
    bool need_headroom = inversion || arena || ct < 0.88f;
    bool urgent_climb = arena && (pz < 8.5f || vz < -1.2f) && ct > 0.58f;
    float max_coll = urgent_climb ? 3.95f :
        (need_headroom ? tb_of_clamp(3.95f - 0.90f * omega_norm - 1.30f * tilt, 2.55f, 3.95f) : 3.95f);
    float fz_des = tb_of_clamp((weight + mass * des_az) / fmaxf(ct, 0.45f),
                                0.10f * weight, max_coll * max_installed);
    float horiz_scale = urgent_climb ? 0.0f :
        (need_headroom ? tb_of_clamp(1.0f - 1.5f * fmaxf(tilt - 0.15f, 0.0f), 0.0f, 1.0f) : 1.0f);
    float tilt_comp = turbine_out ? 0.65f : 0.0f;
    float damp_v = active_slalom ? 0.8f : 0.4f;
    float fx_world = (mass * des_ax + damp_v * vx - tilt_comp * total_fz * body_z_x) * horiz_scale;
    float fy_world = (mass * des_ay + damp_v * vy - tilt_comp * total_fz * body_z_y) * horiz_scale;
    float cy = cosf(yaw), sy = sinf(yaw);
    float max_fh = active_slalom ? 105.0f : (lateral ? 125.0f : (heavy_cargo ? 52.0f : 80.0f));
    float fx_body = tb_of_clamp(cy * fx_world + sy * fy_world, -max_fh, max_fh);
    float fy_body = tb_of_clamp(-sy * fx_world + cy * fy_world, -max_fh, max_fh);

    float cw_x = body_z_y, cw_y = -body_z_x;
    float ux = -qx, uy = -qy, uz = -qz;
    float tt_x = 2.0f * (-uz * cw_y), tt_y = 2.0f * (uz * cw_x);
    float tt_z = 2.0f * (ux * cw_y - uy * cw_x);
    float te_x = cw_x + qw * tt_x + (uy * tt_z - uz * tt_y);
    float te_y = cw_y + qw * tt_y + (uz * tt_x - ux * tt_z);
    float te_z = qw * tt_z + (ux * tt_y - uy * tt_x);
    float te_norm = tb_of_hypot3(te_x, te_y, te_z);
    float tilt_scale = te_norm > 1.0e-5f ? tilt / fmaxf(te_norm, 1.0e-5f) : 0.0f;
    float e_rx = te_x * tilt_scale, e_ry = te_y * tilt_scale;

    bool inv_hold_yaw = (inversion || arena) && (pz < 6.0f || vz < -0.5f || ct < 0.88f);
    bool opp_bearing_valid = arena && fabsf(m[1]) + fabsf(m[2]) > 0.1f;
    float opp_bearing = atan2f(m[1], m[2]);
    float raw_bearing = atan2f(ty - py, tx - px);
    float target_bearing = opp_bearing_valid ? opp_bearing : raw_bearing;
    float desired_yaw = arena && horiz_dist > 1.0f && !inv_hold_yaw ?
        target_bearing - cannon_az_ctr : (inv_hold_yaw ? yaw : 0.0f);
    float yaw_err = tb_of_wrap(desired_yaw, yaw);

    float parasite_scale = turbine_out ? 0.35f : 1.0f;
    float tau_x_par = -0.1625f * fy_body * parasite_scale;
    float tau_y_par = 0.1625f * fx_body * parasite_scale;
    float tau_x_net = tau_x_actual + tau_x_par;
    float tau_y_net = tau_y_actual + tau_y_par;
    float max_txy = heavy_cargo ? 120.0f : 260.0f;
    float tau_x_cmd = tb_of_clamp(95.0f * e_rx - 64.0f * ox - 0.90f * tau_x_net - tau_x_par,
                                  -max_txy, max_txy);
    float tau_y_cmd = tb_of_clamp(95.0f * e_ry - 64.0f * oy - 0.90f * tau_y_net - tau_y_par,
                                  -max_txy, max_txy);
    float tau_z_cmd = tb_of_clamp(22.0f * yaw_err - 18.0f * oz, -70.0f, 70.0f);
    float fz_cmd[4] = {
        0.25f * fz_des - tau_y_cmd * 0.5f,
        0.25f * fz_des + tau_x_cmd * 0.5f,
        0.25f * fz_des + tau_y_cmd * 0.5f,
        0.25f * fz_des - tau_x_cmd * 0.5f
    };
    float yaw_ft = (tau_z_cmd - 0.40f * jammed_yaw_ft) * 0.25f;
    float outer_cmd[4], inner_cmd[4], throttle[4], u_raw[4];
    float fx_outer = ft[3] - ft[1];
    float fy_outer = ft[0] - ft[2];
    float fx_rem = fx_body - fx_outer, fy_rem = fy_body - fy_outer;
    float max_sin_inner = lateral ? 0.44f : (active_slalom ? 0.42f : (heavy_cargo ? 0.22f : 0.32f));
    float spool_gain = turbine_out ? 10.5f : (heavy_cargo ? 4.5f : 2.4f);
    float max_raw = -INFINITY;
    bool land_idle = landing && pz <= 0.638f && up_horiz < 0.85f && fabsf(vz) < 0.20f;
    for (int i = 0; i < 4; ++i) {
        float yaw_alpha = asinf(tb_of_clamp(yaw_ft / fmaxf(thrust[i], 30.0f), -0.35f, 0.35f));
        outer_cmd[i] = tb_of_clamp(6.5f * tb_of_wrap(yaw_alpha, outer[i]), -outer_limit, outer_limit);
        float urx = i == 0 ? 1.0f : (i == 2 ? -1.0f : 0.0f);
        float ury = i == 1 ? 1.0f : (i == 3 ? -1.0f : 0.0f);
        float fr_req = 0.5f * (fx_rem * urx + fy_rem * ury);
        float sin_in = tb_of_clamp(fr_req / fmaxf(thrust[i], 40.0f), -max_sin_inner, max_sin_inner);
        inner_cmd[i] = tb_of_clamp(asinf(sin_in), -inner_limit * 0.90f, inner_limit * 0.90f);
        float nominal_cos = fmaxf(cosf(yaw_alpha) * cosf(inner_cmd[i]), 0.70f);
        float target_thrust = fmaxf(fz_cmd[i], 0.0f) / nominal_cos;
        if (land_idle) target_thrust = 0.15f * weight * 0.25f;
        float u_des = tb_of_clamp((target_thrust - min_thrust) / (max_installed - min_thrust), 0.0f, 1.0f);
        u_raw[i] = u_des + spool_gain * (u_des - spool[i]);
        max_raw = fmaxf(max_raw, u_raw[i]);
    }
    float excess = tb_of_clamp(max_raw - 1.0f, 0.0f, urgent_climb ? 0.10f : 0.45f);
    for (int i = 0; i < 4; ++i)
        throttle[i] = tb_of_clamp(u_raw[i] - (need_headroom ? excess : 0.0f), 0.0f, 1.0f);
    bool inv_mode = ct < 0.20f;
    if (inv_mode) {
        float inv_rx = te_norm > 1.0e-5f ? e_rx : tilt;
        float tx_inv = 130.0f * inv_rx - 52.0f * ox;
        float ty_inv = 130.0f * e_ry - 52.0f * oy;
        float arm_mom[4] = {-ty_inv, tx_inv, ty_inv, -tx_inv};
        for (int i = 0; i < 4; ++i) {
            throttle[i] = arm_mom[i] > 12.0f ? 0.58f : 0.0f;
            outer_cmd[i] = tb_of_clamp(-outer[i] * 6.0f, -outer_limit, outer_limit);
            inner_cmd[i] = 0.0f;
        }
    }

    float to_x = tx - px, to_y = ty - py, to_z = tz - pz;
    float dist = fmaxf(tb_of_hypot3(to_x, to_y, to_z), 1.0f);
    float flight = dist / 20.0f;
    float aim_x = to_x - vx * (0.65f * flight);
    float aim_y = to_y - vy * (0.65f * flight);
    float aim_z = to_z - vz * (0.65f * flight) + 0.53f * gravity * flight * flight;
    float cx = uy * aim_z - uz * aim_y + qw * aim_x;
    float cyq = uz * aim_x - ux * aim_z + qw * aim_y;
    float cz = ux * aim_y - uy * aim_x + qw * aim_z;
    float aim_bx = aim_x + 2.0f * (uy * cz - uz * cyq);
    float aim_by = aim_y + 2.0f * (uz * cx - ux * cz);
    float aim_bz = aim_z + 2.0f * (ux * cyq - uy * cx);
    float des_cannon_az = tb_of_clamp(atan2f(aim_by, aim_bx),
                                      cannon_az_ctr - cannon_az_half,
                                      cannon_az_ctr + cannon_az_half);
    float des_cannon_el = tb_of_clamp(atan2f(aim_bz, fmaxf(hypotf(aim_bx, aim_by), 0.2f)),
                                      -cannon_el_lim, cannon_el_lim);
    float aim_err = hypotf(tb_of_wrap(des_cannon_az, o[44]), des_cannon_el - o[45]);
    bool fire = arena && !inv_mode && o[46] <= 0.02f && o[47] > 0.01f &&
                aim_err < 0.28f && dist < 34.0f;
    bool cannon = arena && !inv_mode;
    float norm_az = cannon ? tb_of_clamp((des_cannon_az - cannon_az_ctr) / cannon_az_half, -1.0f, 1.0f) : 0.0f;
    if (inv_mode) norm_az = -1.0f;
    float norm_el = cannon ? tb_of_clamp(des_cannon_el / cannon_el_lim, -1.0f, 1.0f) : 0.0f;
    float norm_fire = fire ? 0.8f : -0.8f;

    bool bwb_arena = m[6] > 0.75f && m[4] < -0.25f;
    bool bwb_cruise = m[4] > 0.75f && m[2] > 0.25f;
    bool bwb_intercept = m[6] > 0.75f && m[5] > 0.60f;
    bool bwb_endurance = bwb_cruise && m[0] > 0.25f;
    bool bwb = bwb_arena || bwb_cruise || bwb_intercept;
    if (bwb) {
        float vb_x = r00 * vx + r10 * vy + r20 * vz;
        float vb_y = r01 * vx + r11 * vy + r21 * vz;
        float brx = atanhf(tb_of_clamp(m[7], -0.9999f, 0.9999f)) * 20.0f;
        float bry = atanhf(tb_of_clamp(m[8], -0.9999f, 0.9999f)) * 20.0f;
        float brz = atanhf(tb_of_clamp(m[9], -0.9999f, 0.9999f)) * 20.0f;
        float btx = px + brx, bty = py + bry, btz = pz + brz;
        float rxy = hypotf(px, py);
        float pull_b = tb_of_clamp((rxy - 235.0f) / 45.0f, 0.0f, 0.85f);
        if (arena && rxy > 235.0f) { btx *= 1.0f - pull_b; bty *= 1.0f - pull_b; }
        if (arena) btz = tb_of_clamp(btz, 28.0f, 135.0f);
        float gx = btx - px, gy = bty - py, gz = btz - pz;
        float gdist = fmaxf(tb_of_hypot3(gx, gy, gz), 1.0f);
        float desired_yaw_b = atan2f(gy, gx);
        float gs = hypotf(vx, vy);
        float course_yaw = gs > 2.5f ? atan2f(vy, vx) : yaw;
        float course_err = tb_of_wrap(desired_yaw_b, course_yaw);
        float desired_speed = bwb_endurance ? 18.5f : (arena ? (gdist > 38.0f ? 26.0f : 24.0f) : 22.4f);
        float speed_err = desired_speed - vb_x;
        float u_coll = tb_of_clamp((bwb_endurance ? 0.36f : 0.44f) + 0.065f * speed_err + 0.018f * gz,
                                   0.12f, 0.90f);
        float roll_right = atan2f(r21, r22);
        float desired_roll = tb_of_clamp(-1.35f * course_err, -0.55f, 0.55f);
        float diff_u = tb_of_clamp(0.90f * (desired_roll - roll_right) - 0.34f * ox, -0.45f, 0.45f);
        float u0 = tb_of_clamp(u_coll + diff_u, 0.05f, 1.0f);
        float u1 = tb_of_clamp(u_coll - diff_u, 0.05f, 1.0f);
        throttle[0] = throttle[2] = u0;
        throttle[1] = throttle[3] = u1;
        float pitch_up = asinf(tb_of_clamp(r20, -1.0f, 1.0f));
        float climb_des = tb_of_clamp(0.42f * gz, -5.0f, 5.0f);
        float climb_err = climb_des - vz;
        float cos_bank = cosf(roll_right), sin_bank = sinf(roll_right);
        float gamma_des = atan2f(climb_des, fmaxf(gs, 14.0f));
        float desired_pitch = tb_of_clamp(0.062831853f + 0.75f * gamma_des +
            0.035f * climb_err + 0.09f * (1.0f - fmaxf(cos_bank, 0.5f)), -0.15f, 0.25f);
        if (vb_x < 15.5f) desired_pitch = fminf(desired_pitch, 0.02f);
        float theta_dot = 2.20f * (desired_pitch - pitch_up);
        float psi_dot = (gravity / fmaxf(vb_x, 12.0f)) * tanf(tb_of_clamp(-roll_right, -0.75f, 0.75f)) +
            0.35f * course_err;
        float q_up = cos_bank * theta_dot - sin_bank * psi_dot;
        float r_left = sin_bank * theta_dot + cos_bank * psi_dot;
        float gimbal = pi / 6.0f;
        inner_cmd[0] = tb_of_clamp(0.85f * (q_up + oy) + 0.03f, -gimbal, gimbal);
        inner_cmd[1] = tb_of_clamp(-0.65f * (r_left - oz) - 0.025f * vb_y, -gimbal, gimbal);
        inner_cmd[2] = inner_cmd[3] = 0.0f;
        for (int i = 0; i < 4; ++i) outer_cmd[i] = 0.0f;
        float oppx = brx - vx * 0.10f, oppy = bry - vy * 0.10f, oppz = brz - vz * 0.10f;
        float bdist = fmaxf(tb_of_hypot3(oppx, oppy, oppz), 1.0f);
        bool head_on = bwb_arena && btx * px + bty * py < -800.0f && bdist > 45.0f;
        float closing = head_on ? 75.0f + 2.0f * fmaxf(vb_x, 0.0f) : 75.0f;
        float bflight = bdist / closing;
        float bax = oppx, bay = oppy;
        float baz = oppz + (head_on ? -vz * (0.85f * bflight) : 0.0f) +
                    0.50f * gravity * bflight * bflight;
        float bbx = r00 * bax + r10 * bay + r20 * baz;
        float bby = r01 * bax + r11 * bay + r21 * baz;
        float bbz = r02 * bax + r12 * bay + r22 * baz;
        float b_az = tb_of_clamp(atan2f(bby, fmaxf(bbx, 0.2f)), -pi / 6.0f, pi / 6.0f);
        float b_el = tb_of_clamp(atan2f(bbz, fmaxf(hypotf(bbx, bby), 0.2f)),
                                 -cannon_el_lim, cannon_el_lim);
        float b_err = hypotf(tb_of_wrap(b_az, o[44]), b_el - o[45]);
        bool b_fire = arena && o[46] <= 0.02f && o[47] > 0.01f && b_err < 0.045f &&
                      bdist < 165.0f && bbx > 0.0f;
        norm_az = arena ? tb_of_clamp(b_az / (pi / 6.0f), -1.0f, 1.0f) : 0.0f;
        norm_el = arena ? tb_of_clamp(b_el / cannon_el_lim, -1.0f, 1.0f) : 0.0f;
        norm_fire = b_fire ? 0.8f : -0.8f;
    }
    for (int i = 0; i < 4; ++i) {
        n[i] = tb_of_clamp(throttle[i] * 2.0f - 1.0f, -1.0f, 1.0f);
        n[4 + i] = tb_of_clamp(outer_cmd[i] / outer_limit, -1.0f, 1.0f);
        n[8 + i] = tb_of_clamp(inner_cmd[i] / inner_limit, -1.0f, 1.0f);
    }
    n[12] = norm_az;
    n[13] = norm_el;
    n[14] = norm_fire;
}

__device__ __forceinline__ void tb_onboard_tactical_prior(const float *obs_4148,
    const float *mission_10, float tactical_24[24], float prior_15[15]) {
    tb_onboard_tactical(obs_4148, mission_10, tactical_24);
    tb_onboard_prior(obs_4148, mission_10, prior_15);
}

#endif
