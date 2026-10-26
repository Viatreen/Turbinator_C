/* Turbinator desktop telemetry viewer. C11 + Win32/GDI, no Qt or C++ runtime.
 * Reads season JSON artifacts without modifying them. The HDF5 trajectory
 * payload is intentionally not decoded here; replay sidecars are catalogued.
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <windowsx.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TB_PATH_MAX 1024
#define TB_STATUS_MAX 131072
#define TB_REPLAY_MAX 512
#define TB_TREND_MAX 240
#define TB_TIMER_ID 1
#define TB_TIMER_MS 2000

enum { TB_BWB = 0, TB_QUAD = 1 };

typedef struct {
    char replay_id[160];
    char scenario[96];
    char controller[96];
    char opponent_mode[64];
    char timestamp[64];
    int vehicle;
    int scenario_id;
    int steps;
    int player_hits;
    int opponent_hits;
    double duration_s;
    double total_reward;
    uint64_t mtime;
} TbReplay;

typedef struct {
    char state[80];
    char updated_utc[80];
    char season_name[160];
    double env_steps;
    double grad_steps;
    double sps;
    double updates_per_sec;
    double elapsed_minutes;
    double alpha;
    double actor_loss;
    double critic_loss;
    double replay_fill_ratio;
    double talon_qual_rate;
    double turbinator_qual_rate;
    double talon_episodes;
    double turbinator_episodes;
    double arena_win_rate;
    double ballistic_hit_rate;
    int available;
} TbStatus;

typedef struct {
    HWND hwnd;
    char season_dir[TB_PATH_MAX];
    char replay_dir[TB_PATH_MAX];
    TbStatus status;
    TbReplay replays[TB_REPLAY_MAX];
    int replay_count;
    int vehicle;
    int wind_visible;
    int selected_replay;
    int replay_scroll[2];
    int tick;
    float trend[2][TB_TREND_MAX];
    int trend_count;
    HFONT font_title;
    HFONT font_body;
    HFONT font_small;
    HFONT font_mono;
} TbViewer;

static TbViewer g_viewer;

static COLORREF tb_rgb(int r, int g, int b) { return RGB(r, g, b); }

static void tb_path(char *out, size_t cap, const char *dir, const char *name) {
    if (!out || cap == 0) return;
    snprintf(out, cap, "%s\\%s", dir ? dir : "", name ? name : "");
}

static size_t tb_read_file(const char *path, char *out, size_t cap) {
    FILE *f;
    size_t n;
    if (!out || cap == 0) return 0;
    out[0] = '\0';
    f = fopen(path, "rb");
    if (!f) return 0;
    n = fread(out, 1, cap - 1, f);
    out[n] = '\0';
    fclose(f);
    return n;
}

/* Find an object member at its top nesting level, ignoring nested objects and
 * quoted strings. This keeps fields in `scenarios` from shadowing live fields. */
static const char *tb_json_member(const char *json, const char *key) {
    const char *p = json;
    size_t klen = strlen(key);
    int depth = 0;
    if (!json || !key) return NULL;
    while (*p) {
        if (*p == '{' || *p == '[') { ++depth; ++p; continue; }
        if (*p == '}' || *p == ']') { --depth; ++p; continue; }
        if (*p == '"') {
            const char *start = ++p;
            while (*p && *p != '"') {
                if (*p == '\\' && p[1]) ++p;
                ++p;
            }
            if (*p == '"') {
                const char *after = p + 1;
                while (*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
                if (depth == 1 && (size_t)(p - start) == klen &&
                    memcmp(start, key, klen) == 0 && *after == ':') {
                    ++after;
                    while (*after == ' ' || *after == '\t' || *after == '\r' || *after == '\n') ++after;
                    return after;
                }
                ++p;
            }
            continue;
        }
        ++p;
    }
    return NULL;
}

static double tb_json_number(const char *json, const char *key) {
    const char *p = tb_json_member(json, key);
    char *end;
    double v;
    if (!p) return NAN;
    v = strtod(p, &end);
    return end == p ? NAN : v;
}

static int tb_json_string(const char *json, const char *key, char *out, size_t cap) {
    const char *p = tb_json_member(json, key);
    size_t n = 0;
    if (!out || cap == 0) return 0;
    out[0] = '\0';
    if (!p || *p != '"') return 0;
    ++p;
    while (*p && *p != '"' && n + 1 < cap) {
        if (*p == '\\' && p[1]) ++p;
        out[n++] = *p++;
    }
    out[n] = '\0';
    return 1;
}

static void tb_load_status(TbViewer *v) {
    static char json[TB_STATUS_MAX];
    char path[TB_PATH_MAX + 40];
    TbStatus s;
    memset(&s, 0, sizeof(s));
    tb_path(path, sizeof(path), v->season_dir, "live_status.json");
    if (!tb_read_file(path, json, sizeof(json))) {
        v->status = s;
        return;
    }
    s.available = 1;
    tb_json_string(json, "state", s.state, sizeof(s.state));
    tb_json_string(json, "updated_utc", s.updated_utc, sizeof(s.updated_utc));
    tb_json_string(json, "season_name", s.season_name, sizeof(s.season_name));
    s.env_steps = tb_json_number(json, "env_steps");
    s.grad_steps = tb_json_number(json, "grad_steps");
    s.sps = tb_json_number(json, "sps");
    s.updates_per_sec = tb_json_number(json, "updates_per_sec");
    s.elapsed_minutes = tb_json_number(json, "elapsed_minutes");
    s.alpha = tb_json_number(json, "alpha");
    s.actor_loss = tb_json_number(json, "actor_loss");
    s.critic_loss = tb_json_number(json, "critic_loss");
    s.replay_fill_ratio = tb_json_number(json, "replay_fill_ratio");
    s.talon_qual_rate = tb_json_number(json, "talon_6061_qual_rate");
    s.turbinator_qual_rate = tb_json_number(json, "turbinator_qual_rate");
    s.talon_episodes = tb_json_number(json, "talon_6061_episodes");
    s.turbinator_episodes = tb_json_number(json, "turbinator_episodes");
    s.arena_win_rate = tb_json_number(json, "arena_win_rate");
    s.ballistic_hit_rate = tb_json_number(json, "ballistic_hit_rate");
    v->status = s;
}

static int tb_replay_compare(const void *a, const void *b) {
    const TbReplay *x = (const TbReplay *)a;
    const TbReplay *y = (const TbReplay *)b;
    if (x->mtime < y->mtime) return 1;
    if (x->mtime > y->mtime) return -1;
    return strcmp(y->replay_id, x->replay_id);
}

static void tb_load_replays(TbViewer *v) {
    char wildcard[TB_PATH_MAX + 40];
    WIN32_FIND_DATAA fd;
    HANDLE find;
    v->replay_count = 0;
    tb_path(wildcard, sizeof(wildcard), v->replay_dir, "*.json");
    find = FindFirstFileA(wildcard, &fd);
    if (find == INVALID_HANDLE_VALUE) return;
    do {
        char path[TB_PATH_MAX + MAX_PATH + 2];
        char json[8192];
        TbReplay r;
        char vehicle[64];
        if ((fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ||
            strcmp(fd.cFileName, "episodes_index.json") == 0 ||
            strstr(fd.cFileName, "_full.json") != NULL) continue;
        tb_path(path, sizeof(path), v->replay_dir, fd.cFileName);
        if (!tb_read_file(path, json, sizeof(json))) continue;
        memset(&r, 0, sizeof(r));
        if (!tb_json_string(json, "replay_id", r.replay_id, sizeof(r.replay_id))) continue;
        tb_json_string(json, "scenario", r.scenario, sizeof(r.scenario));
        tb_json_string(json, "controller", r.controller, sizeof(r.controller));
        tb_json_string(json, "opponent_mode", r.opponent_mode, sizeof(r.opponent_mode));
        tb_json_string(json, "timestamp_utc", r.timestamp, sizeof(r.timestamp));
        tb_json_string(json, "vehicle_mode", vehicle, sizeof(vehicle));
        r.scenario_id = (int)tb_json_number(json, "scenario_id");
        r.steps = (int)tb_json_number(json, "steps");
        r.player_hits = (int)tb_json_number(json, "player_hits");
        r.opponent_hits = (int)tb_json_number(json, "opponent_hits");
        r.duration_s = tb_json_number(json, "duration_s");
        r.total_reward = tb_json_number(json, "total_reward");
        r.vehicle = strcmp(vehicle, "bwb") == 0 || r.scenario_id >= 22 ? TB_BWB : TB_QUAD;
        r.mtime = ((uint64_t)fd.ftLastWriteTime.dwHighDateTime << 32) |
                   (uint64_t)fd.ftLastWriteTime.dwLowDateTime;
        if (v->replay_count < TB_REPLAY_MAX) v->replays[v->replay_count++] = r;
    } while (FindNextFileA(find, &fd));
    FindClose(find);
    qsort(v->replays, (size_t)v->replay_count, sizeof(v->replays[0]), tb_replay_compare);
    if (v->selected_replay >= v->replay_count ||
        (v->selected_replay >= 0 && v->replays[v->selected_replay].vehicle != v->vehicle))
        v->selected_replay = -1;
}

static void tb_load_trend(TbViewer *v) {
    char path[TB_PATH_MAX + 40];
    char *buf;
    long end;
    long start;
    size_t n;
    FILE *f;
    char *p;
    v->trend_count = 0;
    tb_path(path, sizeof(path), v->season_dir, "metrics_history.jsonl");
    f = fopen(path, "rb");
    if (!f) return;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return; }
    end = ftell(f);
    if (end <= 0) { fclose(f); return; }
    start = end > 131072 ? end - 131072 : 0;
    if (fseek(f, start, SEEK_SET) != 0) { fclose(f); return; }
    buf = (char *)malloc((size_t)(end - start) + 1);
    if (!buf) { fclose(f); return; }
    n = fread(buf, 1, (size_t)(end - start), f);
    fclose(f);
    buf[n] = '\0';
    p = buf;
    if (start > 0) {
        p = strchr(p, '\n');
        if (p) ++p;
        else { free(buf); return; }
    }
    while (*p) {
        char *eol = strchr(p, '\n');
        double bwb, quad;
        if (eol) *eol = '\0';
        bwb = tb_json_number(p, "talon_6061_qual_rate");
        quad = tb_json_number(p, "turbinator_qual_rate");
        if (isfinite(bwb) || isfinite(quad)) {
            if (v->trend_count == TB_TREND_MAX) {
                memmove(v->trend[0], v->trend[0] + 1, (TB_TREND_MAX - 1) * sizeof(float));
                memmove(v->trend[1], v->trend[1] + 1, (TB_TREND_MAX - 1) * sizeof(float));
                --v->trend_count;
            }
            v->trend[0][v->trend_count] = isfinite(bwb) ? (float)bwb : NAN;
            v->trend[1][v->trend_count] = isfinite(quad) ? (float)quad : NAN;
            ++v->trend_count;
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(buf);
}

static void tb_refresh(TbViewer *v) {
    tb_load_status(v);
    tb_load_trend(v);
    if ((v->tick++ % 5) == 0) tb_load_replays(v);
    InvalidateRect(v->hwnd, NULL, FALSE);
}

static void tb_fill(HDC dc, int x, int y, int w, int h, COLORREF c) {
    RECT r;
    HBRUSH brush;
    if (w <= 0 || h <= 0) return;
    r.left = x; r.top = y; r.right = x + w; r.bottom = y + h;
    brush = CreateSolidBrush(c);
    FillRect(dc, &r, brush);
    DeleteObject(brush);
}

static void tb_border(HDC dc, int x, int y, int w, int h, COLORREF c) {
    HPEN pen = CreatePen(PS_SOLID, 1, c);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    HGDIOBJ old_brush = SelectObject(dc, GetStockObject(NULL_BRUSH));
    Rectangle(dc, x, y, x + w, y + h);
    SelectObject(dc, old_brush);
    SelectObject(dc, old_pen);
    DeleteObject(pen);
}

static void tb_text(HDC dc, HFONT font, int x, int y, int w, int h,
                    const char *s, COLORREF color, UINT flags) {
    RECT r;
    HGDIOBJ old = SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, color);
    r.left = x; r.top = y; r.right = x + w; r.bottom = y + h;
    DrawTextA(dc, s ? s : "", -1, &r, flags | DT_NOPREFIX);
    SelectObject(dc, old);
}

static void tb_line(HDC dc, int x0, int y0, int x1, int y1, COLORREF color, int width) {
    HPEN pen = CreatePen(PS_SOLID, width, color);
    HGDIOBJ old = SelectObject(dc, pen);
    MoveToEx(dc, x0, y0, NULL);
    LineTo(dc, x1, y1);
    SelectObject(dc, old);
    DeleteObject(pen);
}

static void tb_polygon(HDC dc, const POINT *pts, int count, COLORREF fill, COLORREF edge) {
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, 2, edge);
    HGDIOBJ old_brush = SelectObject(dc, brush);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    Polygon(dc, pts, count);
    SelectObject(dc, old_pen);
    SelectObject(dc, old_brush);
    DeleteObject(pen);
    DeleteObject(brush);
}

static void tb_circle(HDC dc, int cx, int cy, int radius, COLORREF fill, COLORREF edge) {
    HBRUSH brush = CreateSolidBrush(fill);
    HPEN pen = CreatePen(PS_SOLID, 2, edge);
    HGDIOBJ old_brush = SelectObject(dc, brush);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    Ellipse(dc, cx - radius, cy - radius, cx + radius, cy + radius);
    SelectObject(dc, old_pen);
    SelectObject(dc, old_brush);
    DeleteObject(pen);
    DeleteObject(brush);
}

static void tb_metric(HDC dc, TbViewer *v, int x, int y, int w,
                      const char *label, double value, const char *fmt, COLORREF c) {
    char out[96];
    tb_text(dc, v->font_small, x, y, w, 20, label, tb_rgb(148, 163, 184), DT_LEFT | DT_SINGLELINE);
    if (isfinite(value)) snprintf(out, sizeof(out), fmt, value);
    else snprintf(out, sizeof(out), "--");
    tb_text(dc, v->font_mono, x, y + 17, w, 27, out, c, DT_LEFT | DT_SINGLELINE);
}

static int tb_selected_is_arena(const TbViewer *v) {
    int id;
    if (v->selected_replay < 0 || v->selected_replay >= v->replay_count) return 0;
    id = v->replays[v->selected_replay].scenario_id;
    return id == 3 || id == 14 || id == 18 || id == 22 || id == 24 || id == 27;
}

static void tb_draw_vehicle(HDC dc, TbViewer *v, RECT panel) {
    int cx = (panel.left + panel.right) / 2;
    int cy = panel.top + (panel.bottom - panel.top) / 2 - 8;
    int scale = (panel.right - panel.left) / 6;
    char info[256];
    if (scale > 96) scale = 96;
    if (scale < 42) scale = 42;
    tb_fill(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
            tb_rgb(11, 20, 34));
    tb_border(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
              tb_rgb(43, 65, 85));
    tb_text(dc, v->font_body, panel.left + 18, panel.top + 15,
            panel.right - panel.left - 36, 26,
            v->vehicle == TB_BWB ? "TALON-6061 | STAMPED BWB" : "TURBINATOR | QUAD-JET VTOL",
            tb_rgb(110, 231, 183), DT_LEFT | DT_SINGLELINE);
    if (v->vehicle == TB_BWB) {
        POINT wing[13];
        int i = 0;
        wing[i++] = (POINT){cx, cy - scale};
        wing[i++] = (POINT){cx + scale / 4, cy - scale / 2};
        wing[i++] = (POINT){cx + scale * 2, cy + scale / 3};
        wing[i++] = (POINT){cx + scale * 2, cy + scale / 2};
        wing[i++] = (POINT){cx + scale * 6 / 5, cy + scale / 3};
        wing[i++] = (POINT){cx + scale / 4, cy + scale};
        wing[i++] = (POINT){cx, cy + scale * 6 / 5};
        wing[i++] = (POINT){cx - scale / 4, cy + scale};
        wing[i++] = (POINT){cx - scale * 6 / 5, cy + scale / 3};
        wing[i++] = (POINT){cx - scale * 2, cy + scale / 2};
        wing[i++] = (POINT){cx - scale * 2, cy + scale / 3};
        wing[i++] = (POINT){cx - scale / 4, cy - scale / 2};
        wing[i++] = (POINT){cx, cy - scale};
        tb_polygon(dc, wing, i, tb_rgb(37, 63, 76), tb_rgb(56, 189, 248));
        tb_line(dc, cx, cy - scale, cx, cy + scale, tb_rgb(148, 163, 184), 2);
        tb_circle(dc, cx - scale / 5, cy + scale * 4 / 5, scale / 5,
                  tb_rgb(8, 47, 73), tb_rgb(125, 211, 252));
        tb_circle(dc, cx + scale / 5, cy + scale * 4 / 5, scale / 5,
                  tb_rgb(8, 47, 73), tb_rgb(125, 211, 252));
        tb_text(dc, v->font_small, panel.left + 18, panel.bottom - 56,
                panel.right - panel.left - 36, 38,
                "BWB planform / counter-rotating coaxial props / rear gimbal",
                tb_rgb(148, 163, 184), DT_LEFT | DT_WORDBREAK);
    } else {
        int radius = scale * 6 / 5;
        tb_line(dc, cx - radius, cy - radius, cx + radius, cy + radius,
                tb_rgb(56, 189, 248), 12);
        tb_line(dc, cx + radius, cy - radius, cx - radius, cy + radius,
                tb_rgb(56, 189, 248), 12);
        tb_circle(dc, cx, cy, scale / 2, tb_rgb(40, 56, 72), tb_rgb(148, 163, 184));
        tb_circle(dc, cx - radius, cy - radius, scale / 3, tb_rgb(9, 28, 48), tb_rgb(125, 211, 252));
        tb_circle(dc, cx + radius, cy - radius, scale / 3, tb_rgb(9, 28, 48), tb_rgb(125, 211, 252));
        tb_circle(dc, cx - radius, cy + radius, scale / 3, tb_rgb(9, 28, 48), tb_rgb(125, 211, 252));
        tb_circle(dc, cx + radius, cy + radius, scale / 3, tb_rgb(9, 28, 48), tb_rgb(125, 211, 252));
        tb_text(dc, v->font_small, panel.left + 18, panel.bottom - 56,
                panel.right - panel.left - 36, 38,
                "Four turbines / dual-axis vectoring / independent spool",
                tb_rgb(148, 163, 184), DT_LEFT | DT_WORDBREAK);
    }
    if (tb_selected_is_arena(v)) {
        snprintf(info, sizeof(info), "ARENA: NO WIND (0 m/s)");
    } else if (v->wind_visible) {
        snprintf(info, sizeof(info), "WIND DISPLAY ON - live vector unavailable");
    } else {
        snprintf(info, sizeof(info), "WIND DISPLAY OFF");
    }
    tb_text(dc, v->font_small, panel.right - 310, panel.top + 16, 290, 22,
            info, tb_selected_is_arena(v) ? tb_rgb(110, 231, 183) : tb_rgb(148, 163, 184),
            DT_RIGHT | DT_SINGLELINE);
}

static void tb_draw_status(HDC dc, TbViewer *v, RECT panel) {
    const TbStatus *s = &v->status;
    int x = panel.left + 18;
    int y = panel.top + 15;
    int inner_w = panel.right - panel.left - 36;
    int col_w = (inner_w - 18) / 2;
    char line[300];
    tb_fill(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
            tb_rgb(15, 23, 42));
    tb_border(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
              tb_rgb(43, 65, 85));
    tb_text(dc, v->font_body, x, y, inner_w, 26, "LIVE TRAINING TELEMETRY",
            tb_rgb(110, 231, 183), DT_LEFT | DT_SINGLELINE);
    y += 30;
    if (!s->available) {
        snprintf(line, sizeof(line), "Waiting for %.220s\\live_status.json", v->season_dir);
        tb_text(dc, v->font_small, x, y, inner_w, 75, line,
                tb_rgb(148, 163, 184), DT_LEFT | DT_WORDBREAK);
        return;
    }
    snprintf(line, sizeof(line), "STATE  %s", *s->state ? s->state : "--");
    tb_text(dc, v->font_mono, x, y, inner_w, 25, line,
            tb_rgb(250, 204, 21), DT_LEFT | DT_SINGLELINE);
    y += 27;
    snprintf(line, sizeof(line), "SEASON  %.170s", *s->season_name ? s->season_name : "--");
    tb_text(dc, v->font_small, x, y, inner_w, 25, line,
            tb_rgb(203, 213, 225), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
    y += 30;
    tb_metric(dc, v, x, y, col_w, "Environment steps", s->env_steps, "%.0f", tb_rgb(226, 232, 240));
    tb_metric(dc, v, x + col_w + 18, y, col_w, "Gradient steps", s->grad_steps, "%.0f", tb_rgb(226, 232, 240));
    y += 54;
    tb_metric(dc, v, x, y, col_w, "Simulation steps / s", s->sps, "%.1f", tb_rgb(56, 189, 248));
    tb_metric(dc, v, x + col_w + 18, y, col_w, "Learner updates / s", s->updates_per_sec, "%.2f", tb_rgb(56, 189, 248));
    y += 54;
    tb_metric(dc, v, x, y, col_w, "Vehicle qualification", v->vehicle == TB_BWB ?
              s->talon_qual_rate * 100.0 : s->turbinator_qual_rate * 100.0, "%.1f%%", tb_rgb(110, 231, 183));
    tb_metric(dc, v, x + col_w + 18, y, col_w, "Vehicle episodes", v->vehicle == TB_BWB ?
              s->talon_episodes : s->turbinator_episodes, "%.0f", tb_rgb(110, 231, 183));
    y += 54;
    tb_metric(dc, v, x, y, col_w, "Actor loss", s->actor_loss, "%.4f", tb_rgb(226, 232, 240));
    tb_metric(dc, v, x + col_w + 18, y, col_w, "Critic loss", s->critic_loss, "%.4f", tb_rgb(226, 232, 240));
    y += 54;
    tb_metric(dc, v, x, y, col_w, "Entropy alpha", s->alpha, "%.5f", tb_rgb(226, 232, 240));
    tb_metric(dc, v, x + col_w + 18, y, col_w, "Replay filled", s->replay_fill_ratio * 100.0,
              "%.1f%%", tb_rgb(226, 232, 240));
    if (y + 57 < panel.bottom) {
        snprintf(line, sizeof(line), "Updated UTC  %s", *s->updated_utc ? s->updated_utc : "--");
        tb_text(dc, v->font_small, x, panel.bottom - 25, inner_w, 20, line,
                tb_rgb(100, 116, 139), DT_LEFT | DT_SINGLELINE);
    }
}

static int tb_filtered_replay_index(const TbViewer *v, int row) {
    int i;
    for (i = 0; i < v->replay_count; ++i) {
        if (v->replays[i].vehicle != v->vehicle) continue;
        if (row-- == 0) return i;
    }
    return -1;
}

static int tb_filtered_replay_count(const TbViewer *v) {
    int i, n = 0;
    for (i = 0; i < v->replay_count; ++i) if (v->replays[i].vehicle == v->vehicle) ++n;
    return n;
}

static void tb_draw_replays(HDC dc, TbViewer *v, RECT panel) {
    int row_h = 30;
    int top = panel.top + 48;
    int visible = (panel.bottom - top - 10) / row_h;
    int count = tb_filtered_replay_count(v);
    int i;
    char line[320];
    tb_fill(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
            tb_rgb(15, 23, 42));
    tb_border(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
              tb_rgb(43, 65, 85));
    snprintf(line, sizeof(line), "REPLAYS - %s (%d)", v->vehicle == TB_BWB ? "TALON-6061" : "TURBINATOR", count);
    tb_text(dc, v->font_body, panel.left + 18, panel.top + 13,
            panel.right - panel.left - 36, 27, line,
            tb_rgb(110, 231, 183), DT_LEFT | DT_SINGLELINE);
    for (i = 0; i < visible; ++i) {
        int index = tb_filtered_replay_index(v, i + v->replay_scroll[v->vehicle]);
        int y = top + i * row_h;
        const TbReplay *r;
        if (index < 0) break;
        r = &v->replays[index];
        if (index == v->selected_replay) tb_fill(dc, panel.left + 8, y,
                panel.right - panel.left - 16, row_h - 2, tb_rgb(20, 83, 45));
        snprintf(line, sizeof(line), "%.120s  |  %.70s  |  %.9s",
                 r->replay_id, r->scenario, r->opponent_mode);
        tb_text(dc, v->font_small, panel.left + 18, y + 6,
                panel.right - panel.left - 36, 23, line,
                index == v->selected_replay ? tb_rgb(220, 252, 231) : tb_rgb(203, 213, 225),
                DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
    }
    if (count == 0) {
        tb_text(dc, v->font_small, panel.left + 18, top + 12,
                panel.right - panel.left - 36, 60,
                "No replay metadata for this vehicle in the selected season.",
                tb_rgb(148, 163, 184), DT_LEFT | DT_WORDBREAK);
    }
}

static void tb_draw_chart(HDC dc, TbViewer *v, RECT panel) {
    int x0 = panel.left + 48;
    int y0 = panel.top + 54;
    int x1 = panel.right - 20;
    int y1 = panel.bottom - 42;
    int i;
    char line[230];
    tb_fill(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
            tb_rgb(11, 20, 34));
    tb_border(dc, panel.left, panel.top, panel.right - panel.left, panel.bottom - panel.top,
              tb_rgb(43, 65, 85));
    tb_text(dc, v->font_body, panel.left + 18, panel.top + 13,
            panel.right - panel.left - 36, 27,
            "QUALIFICATION TREND", tb_rgb(110, 231, 183), DT_LEFT | DT_SINGLELINE);
    if (x1 <= x0 || y1 <= y0) return;
    for (i = 0; i <= 4; ++i) {
        int y = y1 - (y1 - y0) * i / 4;
        tb_line(dc, x0, y, x1, y, tb_rgb(43, 65, 85), 1);
        snprintf(line, sizeof(line), "%d%%", i * 25);
        tb_text(dc, v->font_small, panel.left + 8, y - 9, 38, 18, line,
                tb_rgb(100, 116, 139), DT_RIGHT | DT_SINGLELINE);
    }
    if (v->trend_count <= 1) {
        tb_text(dc, v->font_small, x0 + 14, y0 + 14, x1 - x0 - 28, 45,
                "Waiting for metrics_history.jsonl samples.",
                tb_rgb(148, 163, 184), DT_LEFT | DT_WORDBREAK);
        return;
    }
    for (i = 1; i < v->trend_count; ++i) {
        float a = v->trend[v->vehicle][i - 1];
        float b = v->trend[v->vehicle][i];
        int px0 = x0 + (x1 - x0) * (i - 1) / (v->trend_count - 1);
        int px1 = x0 + (x1 - x0) * i / (v->trend_count - 1);
        int py0, py1;
        if (!isfinite(a) || !isfinite(b)) continue;
        if (a < 0) a = 0;
        if (a > 1) a = 1;
        if (b < 0) b = 0;
        if (b > 1) b = 1;
        py0 = y1 - (int)((y1 - y0) * a);
        py1 = y1 - (int)((y1 - y0) * b);
        tb_line(dc, px0, py0, px1, py1, tb_rgb(56, 189, 248), 2);
    }
    tb_text(dc, v->font_small, x0, panel.bottom - 31, x1 - x0, 20,
            "Recent telemetry samples (newest at right)",
            tb_rgb(100, 116, 139), DT_CENTER | DT_SINGLELINE);
}

static void tb_draw_selected_replay(HDC dc, TbViewer *v, RECT panel) {
    const TbReplay *r;
    char line[350];
    if (v->selected_replay < 0 || v->selected_replay >= v->replay_count) return;
    r = &v->replays[v->selected_replay];
    tb_fill(dc, panel.left + 10, panel.bottom - 104,
            panel.right - panel.left - 20, 68, tb_rgb(15, 23, 42));
    tb_border(dc, panel.left + 10, panel.bottom - 104,
              panel.right - panel.left - 20, 68, tb_rgb(43, 65, 85));
    snprintf(line, sizeof(line), "%.90s | %.65s", r->scenario, r->controller);
    tb_text(dc, v->font_small, panel.left + 22, panel.bottom - 99,
            panel.right - panel.left - 42, 22, line,
            tb_rgb(226, 232, 240), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
    snprintf(line, sizeof(line), "%.1f s | %d steps | reward %.2f | hits %d:%d | %s",
             r->duration_s, r->steps, r->total_reward,
             r->player_hits, r->opponent_hits, r->opponent_mode);
    tb_text(dc, v->font_small, panel.left + 22, panel.bottom - 74,
            panel.right - panel.left - 42, 26, line,
            tb_rgb(125, 211, 252), DT_LEFT | DT_SINGLELINE | DT_END_ELLIPSIS);
}

static void tb_paint(TbViewer *v, HDC dc, RECT client) {
    int w = client.right - client.left;
    int h = client.bottom - client.top;
    int gap = 14;
    int left_w = (w - gap * 3) * 51 / 100;
    int right_x = gap * 2 + left_w;
    int right_w = w - right_x - gap;
    int panel_top = 145;
    int top_h = (h - panel_top - gap * 3) * 58 / 100;
    int lower_y = panel_top + top_h + gap;
    int lower_h = h - lower_y - gap;
    RECT vehicle, status, replays, chart;
    tb_fill(dc, 0, 0, w, h, tb_rgb(8, 14, 25));
    tb_fill(dc, 0, 0, w, 76, tb_rgb(15, 23, 42));
    tb_text(dc, v->font_title, 18, 15, w - 36, 38,
            "AEROFORGE APEX | C DESKTOP VIEWER",
            tb_rgb(110, 231, 183), DT_LEFT | DT_SINGLELINE);
    tb_text(dc, v->font_small, 20, 52, w - 40, 20,
            "Live season telemetry and separate vehicle replay catalogs",
            tb_rgb(148, 163, 184), DT_LEFT | DT_SINGLELINE);
    tb_fill(dc, gap, 91, 272, 40,
            v->vehicle == TB_BWB ? tb_rgb(20, 83, 45) : tb_rgb(30, 41, 59));
    tb_border(dc, gap, 91, 272, 40, tb_rgb(52, 211, 153));
    tb_text(dc, v->font_body, gap + 10, 98, 252, 27,
            "1  TALON-6061 BWB", tb_rgb(226, 232, 240), DT_CENTER | DT_SINGLELINE);
    tb_fill(dc, gap + 282, 91, 280, 40,
            v->vehicle == TB_QUAD ? tb_rgb(20, 83, 45) : tb_rgb(30, 41, 59));
    tb_border(dc, gap + 282, 91, 280, 40, tb_rgb(52, 211, 153));
    tb_text(dc, v->font_body, gap + 292, 98, 260, 27,
            "2  TURBINATOR QUAD", tb_rgb(226, 232, 240), DT_CENTER | DT_SINGLELINE);
    tb_fill(dc, w - 215, 91, 201, 40, tb_rgb(30, 41, 59));
    tb_border(dc, w - 215, 91, 201, 40, tb_rgb(71, 85, 105));
    tb_text(dc, v->font_small, w - 207, 99, 185, 25,
            v->wind_visible ? "W  WIND DISPLAY ON" : "W  WIND DISPLAY OFF",
            tb_rgb(203, 213, 225), DT_CENTER | DT_SINGLELINE);
    vehicle = (RECT){gap, panel_top, gap + left_w, panel_top + top_h};
    status = (RECT){right_x, panel_top, right_x + right_w, panel_top + top_h};
    replays = (RECT){gap, lower_y, gap + left_w, lower_y + lower_h};
    chart = (RECT){right_x, lower_y, right_x + right_w, lower_y + lower_h};
    tb_draw_vehicle(dc, v, vehicle);
    tb_draw_selected_replay(dc, v, vehicle);
    tb_draw_status(dc, v, status);
    tb_draw_replays(dc, v, replays);
    tb_draw_chart(dc, v, chart);
}

static int tb_parse_arg(const char **src, char *out, size_t cap) {
    const char *p = *src;
    size_t n = 0;
    int quoted = 0;
    while (*p == ' ' || *p == '\t') ++p;
    if (!*p) { *src = p; return 0; }
    if (*p == '"') { quoted = 1; ++p; }
    while (*p && ((quoted && *p != '"') || (!quoted && *p != ' ' && *p != '\t'))) {
        if (n + 1 < cap) out[n++] = *p;
        ++p;
    }
    if (quoted && *p == '"') ++p;
    out[n] = '\0';
    *src = p;
    return 1;
}

static void tb_parse_command_line(TbViewer *v, const char *cmd) {
    char arg[TB_PATH_MAX];
    int replay_given = 0;
    snprintf(v->season_dir, sizeof(v->season_dir), "seasons\\default");
    while (tb_parse_arg(&cmd, arg, sizeof(arg))) {
        if (strcmp(arg, "--season-dir") == 0) {
            tb_parse_arg(&cmd, v->season_dir, sizeof(v->season_dir));
        } else if (strcmp(arg, "--replay-dir") == 0) {
            tb_parse_arg(&cmd, v->replay_dir, sizeof(v->replay_dir));
            replay_given = 1;
        }
    }
    if (!replay_given) tb_path(v->replay_dir, sizeof(v->replay_dir), v->season_dir, "replays");
}

static LRESULT CALLBACK tb_window_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    TbViewer *v = &g_viewer;
    switch (msg) {
        case WM_CREATE:
            v->hwnd = hwnd;
            v->font_title = CreateFontA(-25, 0, 0, 0, FW_BOLD, 0, 0, 0, ANSI_CHARSET,
                                        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                        DEFAULT_PITCH, "Segoe UI");
            v->font_body = CreateFontA(-18, 0, 0, 0, FW_SEMIBOLD, 0, 0, 0, ANSI_CHARSET,
                                       OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                       DEFAULT_PITCH, "Segoe UI");
            v->font_small = CreateFontA(-15, 0, 0, 0, FW_NORMAL, 0, 0, 0, ANSI_CHARSET,
                                        OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                        DEFAULT_PITCH, "Segoe UI");
            v->font_mono = CreateFontA(-20, 0, 0, 0, FW_BOLD, 0, 0, 0, ANSI_CHARSET,
                                       OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                       DEFAULT_PITCH, "Consolas");
            SetTimer(hwnd, TB_TIMER_ID, TB_TIMER_MS, NULL);
            tb_refresh(v);
            return 0;
        case WM_TIMER:
            if (wp == TB_TIMER_ID) tb_refresh(v);
            return 0;
        case WM_GETMINMAXINFO: {
            MINMAXINFO *mmi = (MINMAXINFO *)lp;
            mmi->ptMinTrackSize.x = 960;
            mmi->ptMinTrackSize.y = 700;
            return 0;
        }
        case WM_ERASEBKGND:
            return 1;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(hwnd, &ps);
            RECT rect;
            HDC mem;
            HBITMAP bitmap;
            HGDIOBJ old_bitmap;
            GetClientRect(hwnd, &rect);
            mem = CreateCompatibleDC(dc);
            bitmap = CreateCompatibleBitmap(dc, rect.right, rect.bottom);
            old_bitmap = SelectObject(mem, bitmap);
            tb_paint(v, mem, rect);
            BitBlt(dc, 0, 0, rect.right, rect.bottom, mem, 0, 0, SRCCOPY);
            SelectObject(mem, old_bitmap);
            DeleteObject(bitmap);
            DeleteDC(mem);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_LBUTTONDOWN: {
            RECT rect;
            int x = GET_X_LPARAM(lp);
            int y = GET_Y_LPARAM(lp);
            int w, h, panel_top = 145, gap = 14, top_h, lower_y, left_w;
            GetClientRect(hwnd, &rect);
            w = rect.right; h = rect.bottom;
            left_w = (w - gap * 3) * 51 / 100;
            top_h = (h - panel_top - gap * 3) * 58 / 100;
            lower_y = panel_top + top_h + gap;
            if (y >= 91 && y < 131) {
                if (x >= gap && x < gap + 272) { v->vehicle = TB_BWB; v->selected_replay = -1; }
                else if (x >= gap + 282 && x < gap + 562) { v->vehicle = TB_QUAD; v->selected_replay = -1; }
                else if (x >= w - 215 && x < w - 14) v->wind_visible = !v->wind_visible;
            } else if (x >= gap && x < gap + left_w && y >= lower_y + 48) {
                int row = (y - (lower_y + 48)) / 30 + v->replay_scroll[v->vehicle];
                int index = tb_filtered_replay_index(v, row);
                if (index >= 0) v->selected_replay = index;
            }
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }
        case WM_MOUSEWHEEL: {
            int delta = GET_WHEEL_DELTA_WPARAM(wp);
            int max_scroll = tb_filtered_replay_count(v) - 1;
            if (max_scroll < 0) max_scroll = 0;
            v->replay_scroll[v->vehicle] -= delta / WHEEL_DELTA;
            if (v->replay_scroll[v->vehicle] < 0) v->replay_scroll[v->vehicle] = 0;
            if (v->replay_scroll[v->vehicle] > max_scroll) v->replay_scroll[v->vehicle] = max_scroll;
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }
        case WM_KEYDOWN:
            if (wp == '1') { v->vehicle = TB_BWB; v->selected_replay = -1; }
            else if (wp == '2') { v->vehicle = TB_QUAD; v->selected_replay = -1; }
            else if (wp == 'W') v->wind_visible = !v->wind_visible;
            else if (wp == VK_F5) tb_refresh(v);
            else if (wp == VK_DOWN || wp == VK_UP) {
                int row = v->replay_scroll[v->vehicle];
                int i;
                if (v->selected_replay >= 0) {
                    row = 0;
                    for (i = 0; i < v->replay_count; ++i) {
                        if (v->replays[i].vehicle != v->vehicle) continue;
                        if (i == v->selected_replay) break;
                        ++row;
                    }
                }
                row += wp == VK_DOWN ? 1 : -1;
                if (row < 0) row = 0;
                i = tb_filtered_replay_index(v, row);
                if (i >= 0) v->selected_replay = i;
            }
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        case WM_DESTROY:
            KillTimer(hwnd, TB_TIMER_ID);
            DeleteObject(v->font_title);
            DeleteObject(v->font_body);
            DeleteObject(v->font_small);
            DeleteObject(v->font_mono);
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcA(hwnd, msg, wp, lp);
    }
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE prev, LPSTR command_line, int show) {
    WNDCLASSA wc;
    MSG msg;
    HWND hwnd;
    (void)prev;
    memset(&g_viewer, 0, sizeof(g_viewer));
    g_viewer.selected_replay = -1;
    g_viewer.vehicle = TB_BWB;
    g_viewer.wind_visible = 0;
    tb_parse_command_line(&g_viewer, command_line ? command_line : "");
    memset(&wc, 0, sizeof(wc));
    wc.lpfnWndProc = tb_window_proc;
    wc.hInstance = instance;
    wc.lpszClassName = "TurbinatorCViewer";
    wc.hCursor = LoadCursorA(NULL, IDC_ARROW);
    wc.hIcon = LoadIconA(NULL, IDI_APPLICATION);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    if (!RegisterClassA(&wc)) return 1;
    hwnd = CreateWindowExA(0, wc.lpszClassName,
                           "Turbinator C Desktop Viewer | TALON-6061 + TURBINATOR",
                           WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
                           1240, 820, NULL, NULL, instance, NULL);
    if (!hwnd) return 2;
    ShowWindow(hwnd, show);
    UpdateWindow(hwnd);
    while (GetMessageA(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    return (int)msg.wParam;
}
