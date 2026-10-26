/* Android NativeActivity front end, written in C11. No Java/Kotlin or C++ app code.
 * The app polls the TUI HTTP status endpoint and runs a local C simulator demo.
 * The selected vehicle is always displayed separately; wind starts hidden.
 */
#define _POSIX_C_SOURCE 200809L
#include <android/input.h>
#include <android/log.h>
#include <android/native_activity.h>
#include <android/native_window.h>
#include <jni.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netdb.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "turbinator/c_api.h"

#define TB_LOG(...) __android_log_print(ANDROID_LOG_INFO, "TurbinatorNative", __VA_ARGS__)
#define TB_HTTP_CAP 65536
#define TB_PI 3.14159265358979323846

typedef struct {
    double env_steps;
    double grad_steps;
    double sps;
    double updates_per_sec;
    double talon_qual;
    double quad_qual;
    double talon_episodes;
    double quad_episodes;
    double cumulative_episodes;
    double qual_rate;
    double alpha;
    double critic_loss;
    double actor_loss;
    int online;
    char state[40];
    char vehicle_mode[40];
} TbRemoteStatus;

typedef struct {
    ANativeActivity *activity;
    pthread_mutex_t lock;
    pthread_cond_t changed;
    pthread_t render_thread;
    pthread_t network_thread;
    int render_started;
    int network_started;
    ANativeWindow *window;
    AInputQueue *input_queue;
    int running;
    int resumed;
    int vehicle;          /* 0 = TALON-6061 BWB, 1 = TURBINATOR quad */
    int solo;             /* arena is initial mode */
    int wind_visible;     /* always OFF at startup */
    int sim_vehicle;
    int sim_solo;
    void *sim;
    uint64_t reset_seed;
    double truth[TB_TRUTH_DIM];
    TbRemoteStatus remote;
    char server_host[128];
    int server_port;
} TbAndroidApp;

/* RGBA_8888 bytes on little-endian Android, represented as 0xAABBGGRR. */
#define TB_RGBA(r,g,b) (0xff000000u | ((uint32_t)(b) << 16) | ((uint32_t)(g) << 8) | (uint32_t)(r))
#define TB_BG TB_RGBA(8,14,25)
#define TB_PANEL TB_RGBA(15,23,42)
#define TB_GRID TB_RGBA(44,65,85)
#define TB_WHITE TB_RGBA(226,232,240)
#define TB_MUTED TB_RGBA(148,163,184)
#define TB_GREEN TB_RGBA(110,231,183)
#define TB_CYAN TB_RGBA(56,189,248)
#define TB_RED TB_RGBA(248,113,113)
#define TB_YELLOW TB_RGBA(250,204,21)

typedef struct {
    uint32_t *pixels;
    int width;
    int height;
    int stride;
} TbCanvas;

typedef struct { char ch; uint8_t col[5]; } TbGlyph;

static const TbGlyph tb_glyphs[] = {
    {'A',{0x7e,0x11,0x11,0x11,0x7e}}, {'B',{0x7f,0x49,0x49,0x49,0x36}},
    {'C',{0x3e,0x41,0x41,0x41,0x22}}, {'D',{0x7f,0x41,0x41,0x22,0x1c}},
    {'E',{0x7f,0x49,0x49,0x49,0x41}}, {'F',{0x7f,0x09,0x09,0x09,0x01}},
    {'G',{0x3e,0x41,0x49,0x49,0x7a}}, {'H',{0x7f,0x08,0x08,0x08,0x7f}},
    {'I',{0x00,0x41,0x7f,0x41,0x00}}, {'J',{0x20,0x40,0x41,0x3f,0x01}},
    {'K',{0x7f,0x08,0x14,0x22,0x41}}, {'L',{0x7f,0x40,0x40,0x40,0x40}},
    {'M',{0x7f,0x02,0x0c,0x02,0x7f}}, {'N',{0x7f,0x04,0x08,0x10,0x7f}},
    {'O',{0x3e,0x41,0x41,0x41,0x3e}}, {'P',{0x7f,0x09,0x09,0x09,0x06}},
    {'Q',{0x3e,0x41,0x51,0x21,0x5e}}, {'R',{0x7f,0x09,0x19,0x29,0x46}},
    {'S',{0x46,0x49,0x49,0x49,0x31}}, {'T',{0x01,0x01,0x7f,0x01,0x01}},
    {'U',{0x3f,0x40,0x40,0x40,0x3f}}, {'V',{0x1f,0x20,0x40,0x20,0x1f}},
    {'W',{0x7f,0x20,0x18,0x20,0x7f}}, {'X',{0x63,0x14,0x08,0x14,0x63}},
    {'Y',{0x03,0x04,0x78,0x04,0x03}}, {'Z',{0x61,0x51,0x49,0x45,0x43}},
    {'0',{0x3e,0x45,0x49,0x51,0x3e}}, {'1',{0x00,0x21,0x7f,0x40,0x00}},
    {'2',{0x42,0x61,0x51,0x49,0x46}}, {'3',{0x21,0x41,0x45,0x4b,0x31}},
    {'4',{0x18,0x14,0x12,0x7f,0x10}}, {'5',{0x27,0x45,0x45,0x45,0x39}},
    {'6',{0x3c,0x4a,0x49,0x49,0x30}}, {'7',{0x01,0x71,0x09,0x05,0x03}},
    {'8',{0x36,0x49,0x49,0x49,0x36}}, {'9',{0x06,0x49,0x49,0x29,0x1e}},
    {' ',{0,0,0,0,0}}, {'.',{0,0x60,0x60,0,0}},
    {':',{0,0x36,0x36,0,0}}, {'-',{0x08,0x08,0x08,0x08,0x08}},
    {'/',{0x20,0x10,0x08,0x04,0x02}}, {'%',{0x63,0x13,0x08,0x64,0x63}},
    {'+',{0x08,0x08,0x3e,0x08,0x08}}, {'=',{0x14,0x14,0x14,0x14,0x14}},
    {'(',{0,0x1c,0x22,0x41,0}}, {')',{0,0x41,0x22,0x1c,0}},
    {'?',{0x02,0x01,0x51,0x09,0x06}}, {'_',{0x40,0x40,0x40,0x40,0x40}},
    {'#',{0x14,0x7f,0x14,0x7f,0x14}}
};

static const uint8_t *tb_font(char ch) {
    size_t i;
    if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 'a' + 'A');
    for (i = 0; i < sizeof(tb_glyphs) / sizeof(tb_glyphs[0]); ++i)
        if (tb_glyphs[i].ch == ch) return tb_glyphs[i].col;
    return tb_glyphs[sizeof(tb_glyphs) / sizeof(tb_glyphs[0]) - 1].col;
}

static void tb_rect(TbCanvas *c, int x, int y, int w, int h, uint32_t color) {
    int x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
    int x1 = x + w > c->width ? c->width : x + w;
    int y1 = y + h > c->height ? c->height : y + h;
    int yy, xx;
    if (x1 <= x0 || y1 <= y0) return;
    for (yy = y0; yy < y1; ++yy) {
        uint32_t *row = c->pixels + yy * c->stride;
        for (xx = x0; xx < x1; ++xx) row[xx] = color;
    }
}

static void tb_line(TbCanvas *c, int x0, int y0, int x1, int y1, uint32_t color, int thick) {
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    for (;;) {
        tb_rect(c, x0 - thick / 2, y0 - thick / 2, thick, thick, color);
        if (x0 == x1 && y0 == y1) break;
        {
            int e2 = 2 * error;
            if (e2 >= dy) { error += dy; x0 += sx; }
            if (e2 <= dx) { error += dx; y0 += sy; }
        }
    }
}

static void tb_circle(TbCanvas *c, int cx, int cy, int radius, uint32_t color, int thick) {
    int i;
    int last_x = cx + radius, last_y = cy;
    for (i = 1; i <= 96; ++i) {
        double t = (2.0 * TB_PI * i) / 96.0;
        int x = cx + (int)lrint(radius * cos(t));
        int y = cy + (int)lrint(radius * sin(t));
        tb_line(c, last_x, last_y, x, y, color, thick);
        last_x = x; last_y = y;
    }
}

static void tb_text(TbCanvas *c, int x, int y, int scale,
                    uint32_t color, const char *message, int max_chars) {
    int n = 0;
    if (!message || scale <= 0) return;
    while (*message && n++ < max_chars) {
        const uint8_t *glyph = tb_font(*message++);
        int col, row;
        for (col = 0; col < 5; ++col)
            for (row = 0; row < 7; ++row)
                if ((glyph[col] >> row) & 1u)
                    tb_rect(c, x + col * scale, y + row * scale,
                            scale, scale, color);
        x += 6 * scale;
    }
}

static void tb_icon(TbCanvas *c, int x, int y, double yaw,
                    int bwb, uint32_t color, int radius) {
    double ca = cos(yaw), sa = sin(yaw);
    int fx = x + (int)lrint(ca * radius);
    int fy = y - (int)lrint(sa * radius);
    int bx = x - (int)lrint(ca * radius * 0.70);
    int by = y + (int)lrint(sa * radius * 0.70);
    int lx = x - (int)lrint(sa * radius * (bwb ? 1.7 : 0.9));
    int ly = y - (int)lrint(ca * radius * (bwb ? 1.7 : 0.9));
    int rx = x + (int)lrint(sa * radius * (bwb ? 1.7 : 0.9));
    int ry = y + (int)lrint(ca * radius * (bwb ? 1.7 : 0.9));
    tb_line(c, fx, fy, lx, ly, color, bwb ? 8 : 6);
    tb_line(c, fx, fy, rx, ry, color, bwb ? 8 : 6);
    tb_line(c, lx, ly, bx, by, color, 5);
    tb_line(c, rx, ry, bx, by, color, 5);
    if (!bwb) {
        tb_circle(c, lx, ly, radius / 3, color, 3);
        tb_circle(c, rx, ry, radius / 3, color, 3);
        tb_circle(c, bx - (lx - x), by - (ly - y), radius / 3, color, 3);
        tb_circle(c, bx - (rx - x), by - (ry - y), radius / 3, color, 3);
    }
}

static double tb_yaw(const double *t, int opponent) {
    int k = opponent ? 53 : 7;
    double w = t[k], x = t[k + 1], y = t[k + 2], z = t[k + 3];
    return atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

static void tb_render(TbCanvas *c, int vehicle, int solo, int wind_visible,
                      const double *t, const TbRemoteStatus *s, const char *host, int port) {
    int w = c->width, h = c->height;
    int scale = w >= 1000 ? 3 : 2;
    int margin = w / 30;
    int cy = h * 43 / 100;
    int cx = w / 2;
    int radius = (w < h ? w : h) * 34 / 100;
    double dome = vehicle == 0 ? 300.0 : 40.0;
    int self_x, self_y, opp_x, opp_y;
    char line[192];
    double speed = sqrt(t[4] * t[4] + t[5] * t[5] + t[6] * t[6]);
    double wind_x = t[69] + t[72], wind_y = t[70] + t[73];
    int txt_y;
    tb_rect(c, 0, 0, w, h, TB_BG);
    tb_rect(c, 0, 0, w, h / 10, TB_PANEL);
    tb_text(c, margin, h / 38, scale, TB_GREEN, "TURBINATOR C MOBILE", 30);
    tb_rect(c, margin, h / 9, (w - 3 * margin) / 2, h / 17,
            vehicle == 0 ? TB_RGBA(20,83,45) : TB_PANEL);
    tb_rect(c, margin * 2 + (w - 3 * margin) / 2, h / 9,
            (w - 3 * margin) / 2, h / 17,
            vehicle == 1 ? TB_RGBA(20,83,45) : TB_PANEL);
    tb_text(c, margin + scale * 4, h / 9 + scale * 3, scale, TB_WHITE,
            "TALON BWB", 30);
    tb_text(c, margin * 2 + (w - 3 * margin) / 2 + scale * 4,
            h / 9 + scale * 3, scale, TB_WHITE, "QUAD JET", 30);
    tb_text(c, margin, h / 5, scale, TB_CYAN,
            vehicle == 0 ? "TALON-6061 STAMPED BWB" : "TURBINATOR QUAD-JET VTOL", 38);
    tb_text(c, margin, h / 5 + 12 * scale, scale, TB_MUTED,
            solo ? "LOCAL REFERENCE SIM - SOLO" : "LOCAL REFERENCE SIM - ARENA", 40);
    tb_circle(c, cx, cy, radius, TB_GRID, 3);
    tb_line(c, cx - radius, cy, cx + radius, cy, TB_GRID, 1);
    tb_line(c, cx, cy - radius, cx, cy + radius, TB_GRID, 1);
    self_x = cx + (int)lrint((t[1] / dome) * radius);
    self_y = cy - (int)lrint((t[2] / dome) * radius);
    if (self_x < margin) self_x = margin;
    if (self_x > w - margin) self_x = w - margin;
    if (self_y < cy - radius) self_y = cy - radius;
    if (self_y > cy + radius) self_y = cy + radius;
    tb_icon(c, self_x, self_y, tb_yaw(t, 0), vehicle == 0, TB_CYAN, w / 38);
    if (!solo) {
        opp_x = cx + (int)lrint((t[47] / dome) * radius);
        opp_y = cy - (int)lrint((t[48] / dome) * radius);
        if (opp_x < margin) opp_x = margin;
        if (opp_x > w - margin) opp_x = w - margin;
        if (opp_y < cy - radius) opp_y = cy - radius;
        if (opp_y > cy + radius) opp_y = cy + radius;
        tb_icon(c, opp_x, opp_y, tb_yaw(t, 1), vehicle == 0, TB_RED, w / 42);
    }
    if (solo && wind_visible) {
        int wx = cx + (int)lrint(wind_x * 8.0);
        int wy = cy - (int)lrint(wind_y * 8.0);
        tb_line(c, cx, cy, wx, wy, TB_YELLOW, 5);
    }
    if (!solo) {
        tb_text(c, margin, h * 66 / 100, scale, TB_GREEN, "ARENA: NO WIND (0 M/S)", 38);
    } else if (wind_visible) {
        snprintf(line, sizeof(line), "WIND %.1F M/S", hypot(wind_x, wind_y));
        tb_text(c, margin, h * 66 / 100, scale, TB_YELLOW, line, 38);
    } else {
        tb_text(c, margin, h * 66 / 100, scale, TB_MUTED, "WIND DISPLAY OFF", 38);
    }
    txt_y = h * 70 / 100;
    snprintf(line, sizeof(line), "SIM T %.1F S  Z %.1F M  V %.1F M/S", t[0], t[3], speed);
    tb_text(c, margin, txt_y, scale, TB_WHITE, line, 46);
    snprintf(line, sizeof(line), "HITS %.0F:%.0F  BAT/FUEL %.1F", t[43], t[65], t[34]);
    tb_text(c, margin, txt_y + 11 * scale, scale, TB_WHITE, line, 46);
    snprintf(line, sizeof(line), "SERVER %s:%d %s", host, port, s->online ? "LIVE" : "OFFLINE");
    tb_text(c, margin, txt_y + 23 * scale, scale, s->online ? TB_GREEN : TB_YELLOW, line, 55);
    if (s->online) {
        snprintf(line, sizeof(line), "STEPS %.0F  SPS %.1F", s->env_steps, s->sps);
        tb_text(c, margin, txt_y + 34 * scale, scale, TB_WHITE, line, 46);
        if (isfinite(vehicle == 0 ? s->talon_qual : s->quad_qual) &&
            isfinite(vehicle == 0 ? s->talon_episodes : s->quad_episodes)) {
            snprintf(line, sizeof(line), "%s QUAL %.1F%%  EP %.0F",
                     vehicle == 0 ? "TALON" : "QUAD",
                     100.0 * (vehicle == 0 ? s->talon_qual : s->quad_qual),
                     vehicle == 0 ? s->talon_episodes : s->quad_episodes);
        } else {
            snprintf(line, sizeof(line), "TRAIN %s  EP %.0F",
                     s->vehicle_mode[0] ? s->vehicle_mode : "UNKNOWN",
                     isfinite(s->cumulative_episodes) ? s->cumulative_episodes : 0.0);
        }
        tb_text(c, margin, txt_y + 45 * scale, scale, TB_CYAN, line, 46);
    } else {
        tb_text(c, margin, txt_y + 34 * scale, scale, TB_MUTED,
                "START TUI HTTP SERVER ON :8765", 48);
    }
    tb_rect(c, margin, h * 92 / 100, (w - 3 * margin) / 2, h / 18, TB_PANEL);
    tb_rect(c, margin * 2 + (w - 3 * margin) / 2, h * 92 / 100,
            (w - 3 * margin) / 2, h / 18, TB_PANEL);
    tb_text(c, margin + scale * 3, h * 92 / 100 + scale * 3, scale,
            TB_WHITE, solo ? "MODE: SOLO" : "MODE: ARENA", 30);
    tb_text(c, margin * 2 + (w - 3 * margin) / 2 + scale * 3,
            h * 92 / 100 + scale * 3, scale, TB_WHITE,
            wind_visible ? "WIND: ON" : "WIND: OFF", 30);
}

static double tb_json_number(const char *json, const char *key) {
    char needle[80];
    const char *p;
    char *end;
    double value;
    if (!json || !key) return NAN;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(json, needle);
    if (!p) return NAN;
    p = strchr(p + strlen(needle), ':');
    if (!p) return NAN;
    ++p;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    value = strtod(p, &end);
    return end == p ? NAN : value;
}

static void tb_json_string(const char *json, const char *key,
                           char *out, size_t cap) {
    char needle[80];
    const char *p;
    size_t n = 0;
    if (!out || cap == 0) return;
    out[0] = '\0';
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(json, needle);
    if (!p || !(p = strchr(p + strlen(needle), ':'))) return;
    p = strchr(p, '"');
    if (!p) return;
    ++p;
    while (p[n] && p[n] != '"' && n + 1 < cap) { out[n] = p[n]; ++n; }
    out[n] = '\0';
}

static int tb_http_get(const char *host, int port, const char *route,
                       char *out, size_t cap) {
    struct addrinfo hints, *resolved = NULL, *it;
    char service[16], request[512];
    int fd = -1, received = 0;
    size_t used = 0;
    ssize_t n;
    struct timeval timeout;
    if (!out || cap < 256) return 0;
    out[0] = '\0';
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    snprintf(service, sizeof(service), "%d", port);
    if (getaddrinfo(host, service, &hints, &resolved) != 0) return 0;
    for (it = resolved; it; it = it->ai_next) {
        int flags;
        fd_set wfds;
        fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
        if (fd < 0) continue;
        flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0) fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        if (connect(fd, it->ai_addr, it->ai_addrlen) == 0 || errno == EINPROGRESS) {
            FD_ZERO(&wfds);
            FD_SET(fd, &wfds);
            timeout.tv_sec = 0;
            timeout.tv_usec = 700000;
            if (select(fd + 1, NULL, &wfds, NULL, &timeout) > 0) {
                int error = 0;
                socklen_t len = sizeof(error);
                if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) == 0 && error == 0)
                    break;
            }
        }
        close(fd); fd = -1;
    }
    freeaddrinfo(resolved);
    if (fd < 0) return 0;
    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK) < 0) {
        close(fd); return 0;
    }
    timeout.tv_sec = 0;
    timeout.tv_usec = 700000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.0\r\nHost: %s\r\nConnection: close\r\n\r\n", route, host);
    if (send(fd, request, strlen(request), 0) < 0) { close(fd); return 0; }
    while (used + 1 < cap && (n = recv(fd, out + used, cap - used - 1, 0)) > 0) {
        used += (size_t)n;
        out[used] = '\0';
        received = 1;
    }
    close(fd);
    if (!received || strncmp(out, "HTTP/1.", 7) != 0 ||
        out[8] != ' ' || strncmp(out + 9, "200", 3) != 0) return 0;
    {
        char *body = strstr(out, "\r\n\r\n");
        if (!body) return 0;
        body += 4;
        memmove(out, body, strlen(body) + 1);
    }
    return 1;
}

static void tb_remote_parse(TbRemoteStatus *s, const char *json) {
    memset(s, 0, sizeof(*s));
    s->online = 1;
    s->env_steps = tb_json_number(json, "env_steps");
    s->grad_steps = tb_json_number(json, "grad_steps");
    s->sps = tb_json_number(json, "sps");
    s->updates_per_sec = tb_json_number(json, "updates_per_sec");
    s->talon_qual = tb_json_number(json, "talon_6061_qual_rate");
    s->quad_qual = tb_json_number(json, "turbinator_qual_rate");
    s->talon_episodes = tb_json_number(json, "talon_6061_episodes");
    s->quad_episodes = tb_json_number(json, "turbinator_episodes");
    s->cumulative_episodes = tb_json_number(json, "cumulative_episodes");
    s->qual_rate = tb_json_number(json, "qual_rate");
    s->alpha = tb_json_number(json, "alpha");
    s->critic_loss = tb_json_number(json, "critic_loss");
    s->actor_loss = tb_json_number(json, "actor_loss");
    tb_json_string(json, "state", s->state, sizeof(s->state));
    tb_json_string(json, "vehicle_mode", s->vehicle_mode, sizeof(s->vehicle_mode));
}

static void tb_nap_ms(int ms) {
    struct timespec t;
    t.tv_sec = ms / 1000;
    t.tv_nsec = (long)(ms % 1000) * 1000000L;
    nanosleep(&t, NULL);
}

static void *tb_network_main(void *arg) {
    TbAndroidApp *app = (TbAndroidApp *)arg;
    char *json = (char *)malloc(TB_HTTP_CAP);
    if (!json) return NULL;
    for (;;) {
        TbRemoteStatus status;
        int alive, i;
        pthread_mutex_lock(&app->lock);
        alive = app->running;
        pthread_mutex_unlock(&app->lock);
        if (!alive) break;
        memset(&status, 0, sizeof(status));
        if (tb_http_get(app->server_host, app->server_port,
                        "/api/v1/status", json, TB_HTTP_CAP))
            tb_remote_parse(&status, json);
        pthread_mutex_lock(&app->lock);
        app->remote = status;
        pthread_mutex_unlock(&app->lock);
        for (i = 0; i < 20; ++i) {
            pthread_mutex_lock(&app->lock);
            alive = app->running;
            pthread_mutex_unlock(&app->lock);
            if (!alive) break;
            tb_nap_ms(100);
        }
    }
    free(json);
    return NULL;
}

static void tb_handle_input(TbAndroidApp *app, int width, int height) {
    AInputQueue *queue;
    AInputEvent *event = NULL;
    int result;
    pthread_mutex_lock(&app->lock);
    queue = app->input_queue;
    if (!queue) { pthread_mutex_unlock(&app->lock); return; }
    while ((result = AInputQueue_getEvent(queue, &event)) >= 0) {
        int handled = 0;
        if (AInputQueue_preDispatchEvent(queue, event)) continue;
        if (AInputEvent_getType(event) == AINPUT_EVENT_TYPE_MOTION &&
            (AMotionEvent_getAction(event) & AMOTION_EVENT_ACTION_MASK) == AMOTION_EVENT_ACTION_UP) {
            float x = AMotionEvent_getX(event, 0);
            float y = AMotionEvent_getY(event, 0);
            if (y >= height * 0.10f && y <= height * 0.19f) {
                app->vehicle = x < width * 0.5f ? 0 : 1;
                handled = 1;
            } else if (y >= height * 0.90f) {
                if (x < width * 0.5f) app->solo = !app->solo;
                else app->wind_visible = !app->wind_visible;
                handled = 1;
            }
        }
        AInputQueue_finishEvent(queue, event, handled);
    }
    pthread_mutex_unlock(&app->lock);
    (void)result;
}

static void *tb_render_main(void *arg) {
    TbAndroidApp *app = (TbAndroidApp *)arg;
    float *obs = (float *)calloc(TB_OBSERVATION_DIM, sizeof(float));
    float action[TB_ACTION_DIM];
    uint8_t term = 0, trunc = 0;
    float reward = 0;
    int frame = 0;
    if (!obs) return NULL;
    app->sim = tb_create(1, 42);
    if (app->sim) tb_reset_one(app->sim, 0, 22, 42);
    app->sim_vehicle = 0;
    app->sim_solo = 0;
    app->reset_seed = 43;
    for (;;) {
        ANativeWindow *window = NULL;
        TbRemoteStatus status;
        int running, resumed, vehicle, solo, wind_visible;
        int width = 0, height = 0;
        pthread_mutex_lock(&app->lock);
        running = app->running;
        resumed = app->resumed;
        vehicle = app->vehicle;
        solo = app->solo;
        wind_visible = app->wind_visible;
        status = app->remote;
        if (app->window) {
            window = app->window;
            ANativeWindow_acquire(window);
            width = ANativeWindow_getWidth(window);
            height = ANativeWindow_getHeight(window);
        }
        pthread_mutex_unlock(&app->lock);
        if (!running) { if (window) ANativeWindow_release(window); break; }
        if (window && width > 0 && height > 0) tb_handle_input(app, width, height);
        if (app->sim && (vehicle != app->sim_vehicle || solo != app->sim_solo)) {
            int scenario = vehicle == 0 ? (solo ? 23 : 22) : (solo ? 0 : 3);
            tb_reset_one(app->sim, 0, scenario, ++app->reset_seed);
            app->sim_vehicle = vehicle;
            app->sim_solo = solo;
        }
        if (app->sim && resumed) {
            memset(action, 0, sizeof(action));
            tb_reference_action(app->sim, 0, NULL, action);
            tb_step(app->sim, action, obs, &reward, &term, &trunc);
            if (term || trunc) {
                int scenario = vehicle == 0 ? (solo ? 23 : 22) : (solo ? 0 : 3);
                tb_reset_one(app->sim, 0, scenario, ++app->reset_seed);
            }
            tb_get_truth(app->sim, 0, app->truth);
        }
        if (window && resumed && (frame++ % 2 == 0)) {
            ANativeWindow_Buffer buffer;
            if (ANativeWindow_lock(window, &buffer, NULL) == 0) {
                TbCanvas canvas;
                canvas.pixels = (uint32_t *)buffer.bits;
                canvas.width = buffer.width;
                canvas.height = buffer.height;
                canvas.stride = buffer.stride;
                tb_render(&canvas, vehicle, solo, wind_visible, app->truth,
                          &status, app->server_host, app->server_port);
                ANativeWindow_unlockAndPost(window);
            }
        }
        if (window) ANativeWindow_release(window);
        tb_nap_ms(20);
    }
    if (app->sim) { tb_destroy(app->sim); app->sim = NULL; }
    free(obs);
    return NULL;
}

static void tb_read_intent_host(TbAndroidApp *app) {
    JNIEnv *env = app->activity->env;
    jobject activity = app->activity->clazz;
    jclass cls = (*env)->GetObjectClass(env, activity);
    jmethodID get_intent;
    jobject intent;
    jclass intent_cls;
    jmethodID get_string;
    jstring key, host;
    const char *utf;
    if (!cls) return;
    get_intent = (*env)->GetMethodID(env, cls, "getIntent", "()Landroid/content/Intent;");
    if (!get_intent) { (*env)->DeleteLocalRef(env, cls); return; }
    intent = (*env)->CallObjectMethod(env, activity, get_intent);
    (*env)->DeleteLocalRef(env, cls);
    if (!intent) return;
    intent_cls = (*env)->GetObjectClass(env, intent);
    get_string = (*env)->GetMethodID(env, intent_cls, "getStringExtra",
                                      "(Ljava/lang/String;)Ljava/lang/String;");
    if (get_string) {
        key = (*env)->NewStringUTF(env, "server_host");
        host = (jstring)(*env)->CallObjectMethod(env, intent, get_string, key);
        if (host) {
            utf = (*env)->GetStringUTFChars(env, host, NULL);
            if (utf && strlen(utf) < sizeof(app->server_host))
                snprintf(app->server_host, sizeof(app->server_host), "%s", utf);
            if (utf) (*env)->ReleaseStringUTFChars(env, host, utf);
            (*env)->DeleteLocalRef(env, host);
        }
        (*env)->DeleteLocalRef(env, key);
    }
    (*env)->DeleteLocalRef(env, intent_cls);
    (*env)->DeleteLocalRef(env, intent);
}

static void tb_read_host_file(TbAndroidApp *app) {
    char path[1024], line[180];
    FILE *f;
    char *colon;
    if (!app->activity->internalDataPath) return;
    snprintf(path, sizeof(path), "%s/server_host.txt", app->activity->internalDataPath);
    f = fopen(path, "rb");
    if (!f) return;
    if (fgets(line, sizeof(line), f)) {
        size_t n = strcspn(line, "\r\n");
        line[n] = '\0';
        colon = strrchr(line, ':');
        if (colon) {
            int port = atoi(colon + 1);
            *colon = '\0';
            if (port > 0 && port < 65536) app->server_port = port;
        }
        if (line[0] && strlen(line) < sizeof(app->server_host))
            snprintf(app->server_host, sizeof(app->server_host), "%s", line);
    }
    fclose(f);
}

static void tb_on_start(ANativeActivity *activity) { (void)activity; }
static void tb_on_resume(ANativeActivity *activity) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    pthread_mutex_lock(&app->lock); app->resumed = 1; pthread_mutex_unlock(&app->lock);
}
static void tb_on_pause(ANativeActivity *activity) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    pthread_mutex_lock(&app->lock); app->resumed = 0; pthread_mutex_unlock(&app->lock);
}
static void tb_on_stop(ANativeActivity *activity) { (void)activity; }

static void tb_on_window_created(ANativeActivity *activity, ANativeWindow *window) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    ANativeWindow_setBuffersGeometry(window, 0, 0, WINDOW_FORMAT_RGBA_8888);
    pthread_mutex_lock(&app->lock);
    if (app->window) ANativeWindow_release(app->window);
    ANativeWindow_acquire(window);
    app->window = window;
    pthread_cond_broadcast(&app->changed);
    pthread_mutex_unlock(&app->lock);
}

static void tb_on_window_destroyed(ANativeActivity *activity, ANativeWindow *window) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    (void)window;
    pthread_mutex_lock(&app->lock);
    if (app->window) { ANativeWindow_release(app->window); app->window = NULL; }
    pthread_mutex_unlock(&app->lock);
}

static void tb_on_input_created(ANativeActivity *activity, AInputQueue *queue) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    pthread_mutex_lock(&app->lock); app->input_queue = queue; pthread_mutex_unlock(&app->lock);
}

static void tb_on_input_destroyed(ANativeActivity *activity, AInputQueue *queue) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    pthread_mutex_lock(&app->lock);
    if (app->input_queue == queue) app->input_queue = NULL;
    pthread_mutex_unlock(&app->lock);
}

static void tb_on_destroy(ANativeActivity *activity) {
    TbAndroidApp *app = (TbAndroidApp *)activity->instance;
    if (!app) return;
    pthread_mutex_lock(&app->lock);
    app->running = 0;
    pthread_cond_broadcast(&app->changed);
    pthread_mutex_unlock(&app->lock);
    if (app->render_started) pthread_join(app->render_thread, NULL);
    if (app->network_started) pthread_join(app->network_thread, NULL);
    if (app->window) ANativeWindow_release(app->window);
    pthread_cond_destroy(&app->changed);
    pthread_mutex_destroy(&app->lock);
    free(app);
    activity->instance = NULL;
}

JNIEXPORT void ANativeActivity_onCreate(ANativeActivity *activity,
                                        void *saved_state, size_t saved_state_size) {
    TbAndroidApp *app = (TbAndroidApp *)calloc(1, sizeof(*app));
    (void)saved_state;
    (void)saved_state_size;
    if (!app) return;
    app->activity = activity;
    app->running = 1;
    app->vehicle = 0;
    app->solo = 0;
    app->wind_visible = 0;
    app->server_port = 8765;
    snprintf(app->server_host, sizeof(app->server_host), "10.0.2.2");
    tb_read_host_file(app);
    tb_read_intent_host(app);
    pthread_mutex_init(&app->lock, NULL);
    pthread_cond_init(&app->changed, NULL);
    activity->instance = app;
    activity->callbacks->onStart = tb_on_start;
    activity->callbacks->onResume = tb_on_resume;
    activity->callbacks->onPause = tb_on_pause;
    activity->callbacks->onStop = tb_on_stop;
    activity->callbacks->onDestroy = tb_on_destroy;
    activity->callbacks->onNativeWindowCreated = tb_on_window_created;
    activity->callbacks->onNativeWindowDestroyed = tb_on_window_destroyed;
    activity->callbacks->onInputQueueCreated = tb_on_input_created;
    activity->callbacks->onInputQueueDestroyed = tb_on_input_destroyed;
    if (pthread_create(&app->render_thread, NULL, tb_render_main, app) != 0) {
        TB_LOG("render thread failed");
        ANativeActivity_finish(activity);
        return;
    }
    app->render_started = 1;
    if (pthread_create(&app->network_thread, NULL, tb_network_main, app) != 0) {
        TB_LOG("network thread failed");
        ANativeActivity_finish(activity);
        return;
    }
    app->network_started = 1;
}
