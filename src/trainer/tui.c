#include "turbinator/c_api.h"
#include "turbinator/trainer/tui.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <conio.h>
#include <direct.h>
#include <windows.h>
#define tb_mkdir(p) _mkdir(p)
#else
#include <sys/select.h>
#include <sys/stat.h>
#include <unistd.h>
#define tb_mkdir(p) mkdir((p), 0775)
#endif

typedef struct {
    char season_dir[1024];
    char status[16384];
    char last_log[256];
    char frame[20000];
    int flags;
    int enabled;
    double max_duration_minutes;
    double last_render;
#ifdef _WIN32
    HANDLE viewer_process;
#endif
} TbTui;

static double tb_now_seconds(void) {
#ifdef _WIN32
    return (double)GetTickCount64() / 1000.0;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static int tb_make_dirs(const char *path) {
    char tmp[1024];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(tmp)) return 0;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; ++i) {
        if (tmp[i] != '/' && tmp[i] != '\\') continue;
        if (i == 2 && tmp[1] == ':') continue;
        char save = tmp[i]; tmp[i] = 0;
        if (tb_mkdir(tmp) != 0 && errno != EEXIST) return 0;
        tmp[i] = save;
    }
    return tb_mkdir(tmp) == 0 || errno == EEXIST;
}

static int tb_path(char *dst, size_t cap, const TbTui *t, const char *name) {
    int n = snprintf(dst, cap, "%s/%s", t->season_dir, name);
    return n > 0 && (size_t)n < cap;
}

static void tb_write_atomic(const char *path, const char *body) {
    char tmp[1200];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) return;
    FILE *f = fopen(tmp, "wb");
    if (!f) return;
    fwrite(body, 1, strlen(body), f);
    if (fclose(f) != 0) return;
#ifdef _WIN32
    remove(path);
#endif
    rename(tmp, path);
}

static void tb_control(TbTui *t, const char *body) {
    char path[1200];
    if (tb_path(path, sizeof(path), t, "training_control.json")) tb_write_atomic(path, body);
}

static void tb_log_file(TbTui *t, const char *msg) {
    char path[1200];
    if (!tb_path(path, sizeof(path), t, "tui_events.log")) return;
    FILE *f = fopen(path, "ab");
    if (!f) return;
    fprintf(f, "%s\n", msg);
    fclose(f);
}

static size_t tb_read_file(const char *path, char *dst, size_t cap) {
    if (cap == 0) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) { dst[0] = 0; return 0; }
    size_t n = fread(dst, 1, cap - 1, f);
    dst[n] = 0;
    fclose(f);
    return n;
}

static double tb_json_number(const char *json, const char *key, double fallback) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return fallback;
    p = strchr(p + strlen(pattern), ':');
    if (!p) return fallback;
    char *end = NULL;
    double v = strtod(p + 1, &end);
    return end == p + 1 ? fallback : v;
}

static void tb_json_string(const char *json, const char *key, char *out, size_t cap, const char *fallback) {
    if (cap == 0) return;
    snprintf(out, cap, "%s", fallback);
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p || !(p = strchr(p + strlen(pattern), ':'))) return;
    p = strchr(p, '"');
    if (!p) return;
    ++p;
    size_t i = 0;
    while (p[i] && p[i] != '"' && i + 1 < cap) { out[i] = p[i]; ++i; }
    out[i] = 0;
}

TB_API void *tb_tui_create(const char *season_dir, int tui_enabled, double max_duration_minutes, int log_capacity) {
    (void)log_capacity;
    TbTui *t = (TbTui *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    snprintf(t->season_dir, sizeof(t->season_dir), "%s", season_dir && *season_dir ? season_dir : "seasons/default");
    if (!tb_make_dirs(t->season_dir)) { free(t); return NULL; }
    t->enabled = tui_enabled != 0;
    t->max_duration_minutes = max_duration_minutes > 0 ? max_duration_minutes : -1.0;
    t->flags = TB_TUI_CFD | (max_duration_minutes > 0 ? 0 : TB_TUI_INDEFINITE);
    tb_tui_sync_from_disk(t);
    return t;
}

TB_API void tb_tui_destroy(void *handle) {
    TbTui *t = (TbTui *)handle;
    if (!t) return;
#ifdef _WIN32
    if (t->viewer_process) CloseHandle(t->viewer_process);
#endif
    free(t);
}

TB_API void tb_tui_log(void *handle, const char *message, int also_print_if_not_tui) {
    TbTui *t = (TbTui *)handle;
    if (!t || !message) return;
    snprintf(t->last_log, sizeof(t->last_log), "%s", message);
    tb_log_file(t, message);
    if (!t->enabled && also_print_if_not_tui) fprintf(stdout, "%s\n", message);
}

TB_API void tb_tui_poll_controls(void *handle, int write_to_control_file) {
    TbTui *t = (TbTui *)handle;
    if (!t || !t->enabled) return;
    int key = -1;
#ifdef _WIN32
    if (_kbhit()) key = _getch();
#else
    fd_set r;
    struct timeval tv = {0, 0};
    FD_ZERO(&r); FD_SET(STDIN_FILENO, &r);
    if (select(STDIN_FILENO + 1, &r, NULL, NULL, &tv) > 0) key = getchar();
#endif
    if (key < 0) return;
    const char *cmd = NULL;
    if (key == 'p' || key == ' ') {
        t->flags ^= TB_TUI_PAUSED;
        cmd = (t->flags & TB_TUI_PAUSED) ? "{\"requested_state\":\"PAUSED\"}" : "{\"requested_state\":\"TRAINING_ACTIVE\"}";
    } else if (key == 'q') {
        t->flags |= TB_TUI_STOP;
        cmd = "{\"requested_state\":\"STOPPED_SAVED_UNLOADED\"}";
    } else if (key == 'c') {
        t->flags |= TB_TUI_CHECKPOINT;
        cmd = "{\"save_checkpoint_now\":true}";
    } else if (key == 'e') {
        t->flags |= TB_TUI_EVALUATE;
        cmd = "{\"run_eval_now\":true}";
    } else if (key == 'r') {
        t->flags |= TB_TUI_RECORD;
        cmd = "{\"record_replay_now\":true}";
    } else if (key == 'i') {
        t->flags ^= TB_TUI_INDEFINITE;
        t->max_duration_minutes = (t->flags & TB_TUI_INDEFINITE) ? -1.0 : 600.0;
        cmd = (t->flags & TB_TUI_INDEFINITE) ? "{\"indefinite_training\":true}" : "{\"indefinite_training\":false,\"set_duration_minutes\":600}";
    } else if (key == 'v') {
        t->flags ^= TB_TUI_BWB;
    } else if (key == 'o') {
        tb_tui_toggle_desktop_viewer(t);
    } else if (key == 'a') {
        t->flags ^= TB_TUI_CFD;
    }
    if (cmd && write_to_control_file) tb_control(t, cmd);
}

TB_API void tb_tui_sync_from_disk(void *handle) {
    TbTui *t = (TbTui *)handle;
    char path[1200];
    if (t && tb_path(path, sizeof(path), t, "live_status.json"))
        tb_read_file(path, t->status, sizeof(t->status));
}

TB_API void tb_tui_record_status_json(void *handle, const char *status_json) {
    TbTui *t = (TbTui *)handle;
    if (t && status_json) snprintf(t->status, sizeof(t->status), "%s", status_json);
}

TB_API const char *tb_tui_format_frame(void *handle, double session_elapsed_min, const char *mobile_api_summary) {
    TbTui *t = (TbTui *)handle;
    if (!t) return "";
    char state[64], vehicle[64];
    tb_json_string(t->status, "state", state, sizeof(state), "UNKNOWN");
    tb_json_string(t->status, "vehicle_mode", vehicle, sizeof(vehicle), "dual_unified");
    snprintf(t->frame, sizeof(t->frame),
        "TURBINATOR C | %s | %s\n"
        "Season: %s\n"
        "Elapsed: %.1f min | Env steps: %.0f | Grad steps: %.0f | Episodes: %.0f\n"
        "SPS: %.1f | Qual: %.1f%% | Return: %.3f | Actor loss: %.4f | Critic loss: %.4f\n"
        "%s\n"
        "Controls: p pause, i indefinite, c checkpoint, e evaluate, r replay, v vehicle, o viewer, q stop\n"
        "%s\n",
        state, vehicle, t->season_dir, session_elapsed_min,
        tb_json_number(t->status, "env_steps", 0), tb_json_number(t->status, "grad_steps", 0),
        tb_json_number(t->status, "cumulative_episodes", 0), tb_json_number(t->status, "sps", 0),
        100.0 * tb_json_number(t->status, "qual_rate", 0), tb_json_number(t->status, "episode_return", 0),
        tb_json_number(t->status, "actor_loss", 0), tb_json_number(t->status, "critic_loss", 0),
        mobile_api_summary ? mobile_api_summary : "", t->last_log);
    return t->frame;
}

TB_API void tb_tui_render_if_due(void *handle, double session_elapsed_min, const char *mobile_api_summary, int force) {
    TbTui *t = (TbTui *)handle;
    if (!t || !t->enabled) return;
    double now = tb_now_seconds();
    if (!force && now - t->last_render < 0.5) return;
    t->last_render = now;
    fputs("\033[2J\033[H", stdout);
    fputs(tb_tui_format_frame(t, session_elapsed_min, mobile_api_summary), stdout);
    fflush(stdout);
}

TB_API int tb_tui_get_state_flags(void *handle, int clear_requests) {
    TbTui *t = (TbTui *)handle;
    if (!t) return 0;
    int flags = t->flags;
    if (clear_requests) t->flags &= ~(TB_TUI_CHECKPOINT | TB_TUI_EVALUATE | TB_TUI_RECORD);
    return flags;
}
TB_API double tb_tui_get_max_duration_minutes(void *handle) {
    TbTui *t = (TbTui *)handle;
    return t ? t->max_duration_minutes : -1.0;
}
TB_API void tb_tui_set_max_duration_minutes(void *handle, double minutes) {
    TbTui *t = (TbTui *)handle;
    if (!t) return;
    t->max_duration_minutes = minutes > 0 ? minutes : -1;
    if (minutes > 0) t->flags &= ~TB_TUI_INDEFINITE; else t->flags |= TB_TUI_INDEFINITE;
}
TB_API int tb_tui_is_indefinite_training(void *handle) {
    TbTui *t = (TbTui *)handle;
    return t && (t->flags & TB_TUI_INDEFINITE) ? 1 : 0;
}
TB_API void tb_tui_set_indefinite_training(void *handle, int enabled, int write_to_control_file) {
    TbTui *t = (TbTui *)handle;
    if (!t) return;
    tb_tui_set_max_duration_minutes(t, enabled ? -1 : 600);
    if (write_to_control_file) tb_control(t, enabled ? "{\"indefinite_training\":true}" : "{\"indefinite_training\":false,\"set_duration_minutes\":600}");
}
TB_API void tb_tui_switch_vehicle_mode(void *handle) {
    TbTui *t = (TbTui *)handle;
    if (t) t->flags ^= TB_TUI_BWB;
}
TB_API void tb_tui_toggle_desktop_viewer(void *handle) {
    TbTui *t = (TbTui *)handle;
    if (!t) return;
#ifdef _WIN32
    if (t->viewer_process) {
        DWORD state = WaitForSingleObject(t->viewer_process, 0);
        if (state == WAIT_TIMEOUT) {
            TerminateProcess(t->viewer_process, 0);
            tb_tui_log(t, "Desktop viewer closed.", 1);
            CloseHandle(t->viewer_process);
            t->viewer_process = NULL;
            return;
        }
        CloseHandle(t->viewer_process);
        t->viewer_process = NULL;
    }
    char exe[MAX_PATH], command[2048];
    if (!GetModuleFileNameA(NULL, exe, MAX_PATH)) return;
    char *slash = strrchr(exe, '\\');
    if (!slash) return;
    slash[1] = 0;
    if (strlen(exe) + strlen("turbinator_viewer.exe") + 1 >= MAX_PATH) return;
    strcat(exe, "turbinator_viewer.exe");
    if (snprintf(command, sizeof(command), "\"%s\" --season-dir \"%s\"", exe, t->season_dir) >= (int)sizeof(command)) return;
    STARTUPINFOA si = {0};
    PROCESS_INFORMATION pi = {0};
    si.cb = sizeof(si);
    if (CreateProcessA(exe, command, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        t->viewer_process = pi.hProcess;
        CloseHandle(pi.hThread);
        tb_tui_log(t, "Desktop viewer opened.", 1);
    } else tb_tui_log(t, "Desktop viewer executable unavailable.", 1);
#else
    tb_tui_log(t, "Desktop viewer is available on Windows.", 1);
#endif
}
TB_API const char *tb_tui_get_season_dir(void *handle) {
    TbTui *t = (TbTui *)handle;
    return t ? t->season_dir : "";
}
TB_API const char *tb_tui_format_mst_now(void) {
    static char out[64];
    time_t now = time(NULL);
    struct tm value;
#ifdef _WIN32
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    /* Arizona stays at UTC-7 and does not observe daylight saving time. */
    now -= 7 * 3600;
#ifdef _WIN32
    gmtime_s(&value, &now);
#else
    gmtime_r(&now, &value);
#endif
    strftime(out, sizeof(out), "%Y-%m-%d %H:%M:%S MST", &value);
    return out;
}
