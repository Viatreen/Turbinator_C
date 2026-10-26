#include "turbinator/c_api.h"
#include "turbinator/trainer/http.h"
#include "turbinator/trainer/tui.h"
#include "turbinator/training/training.h"
#include "turbinator/network/network.h"
#ifdef TURBINATOR_HAS_CUDA
#include "turbinator/training/cuda_features.h"
#endif
#ifdef TURBINATOR_HAS_LARGE_GPU
#include "turbinator/gpu/trainer.h"
#endif

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#define tb_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define tb_mkdir(p) mkdir((p), 0775)
#endif

typedef struct {
    const char *command;
    const char *season_dir;
    const char *checkpoint;
    const char *output;
    const char *vehicle_mode;
    const char *engine;
    const char *host;
    int scenario;
    int batch;
    int steps;
    int trials;
    int eval_steps;
    int eval_interval;
    int updates_per_step;
    int checkpoint_interval;
    int status_interval;
    int serve_port;
    int tui;
    int serve;
    int indefinite;
    int tail;
    int gpu_count;
    int gpu_device;
    int gpu_batch;
    int gpu_replay_capacity;
    int no_save;
    int batch_explicit;
    uint64_t seed;
} Options;

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static void sleep_ms(unsigned ms) {
#ifdef _WIN32
    Sleep(ms);
#else
    struct timespec t = {(time_t)(ms / 1000), (long)(ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
#endif
}

static int make_dirs(const char *path) {
    char tmp[1200];
    size_t n = strlen(path);
    if (!n || n >= sizeof(tmp)) return 0;
    memcpy(tmp, path, n + 1);
    for (size_t i = 1; i < n; ++i) {
        if (tmp[i] != '/' && tmp[i] != '\\') continue;
        if (i == 2 && tmp[1] == ':') continue;
        char save = tmp[i]; tmp[i] = 0;
        if (tb_mkdir(tmp) && errno != EEXIST) return 0;
        tmp[i] = save;
    }
    return tb_mkdir(tmp) == 0 || errno == EEXIST;
}

static int join_path(char *out, size_t cap, const char *a, const char *b) {
    int n = snprintf(out, cap, "%s/%s", a, b);
    return n > 0 && (size_t)n < cap;
}

static int write_atomic(const char *path, const char *data) {
    char tmp[1400];
    if (snprintf(tmp, sizeof(tmp), "%s.tmp", path) >= (int)sizeof(tmp)) return 0;
    FILE *f = fopen(tmp, "wb");
    if (!f) return 0;
    size_t n = strlen(data);
    int ok = fwrite(data, 1, n, f) == n && fclose(f) == 0;
    if (!ok) { remove(tmp); return 0; }
#ifdef _WIN32
    remove(path);
#endif
    return rename(tmp, path) == 0;
}

static char *read_text(const char *path, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *data = (char *)malloc(cap + 1);
    if (!data) { fclose(f); return NULL; }
    size_t n = fread(data, 1, cap, f);
    data[n] = 0;
    fclose(f);
    return data;
}

static int json_number(const char *json, const char *key, double *out) {
    char pattern[96];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p || !(p = strchr(p + strlen(pattern), ':'))) return 0;
    char *end;
    double v = strtod(p + 1, &end);
    if (end == p + 1) return 0;
    *out = v;
    return 1;
}

static int contains_true(const char *json, const char *key) {
    char pattern[96];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p || !(p = strchr(p + strlen(pattern), ':'))) return 0;
    return strstr(p, "true") == p + strspn(p, ": \t\r\n");
}

static void usage(const char *app) {
    printf("Turbinator C native simulator and SAC trainer\n"
           "Usage: %s <train|evaluate|record|benchmark|network-info|tui|init-season|ipc-status|ipc-logs|ipc-cmd> [options]\n"
           "Options: --season-dir DIR --checkpoint PATH --output FILE --vehicle-mode dual_unified|bwb|quad_jet\n"
           "         --scenario ID --batch N --steps N --trials N --seed N --tui --serve --serve-host IP\n"
           "         --serve-port N --indefinite --checkpoint-interval N --status-interval N\n"
           "         --updates-per-step N --eval-steps N --eval-interval N --tail N\n"
           "         --engine cpu|cuda-618m|cuda-p3000 --gpu-device ID --gpu-count N\n"
           "         --gpu-batch N --gpu-replay-capacity N --no-save\n"
           "ipc-cmd requires one of pause, resume, stop, force_eval, force_ckpt, force_replay.\n", app);
}

static int parse_args(int argc, char **argv, Options *o) {
    memset(o, 0, sizeof(*o));
    o->command = argc > 1 ? argv[1] : "tui";
    o->season_dir = "seasons/default";
    o->vehicle_mode = "dual_unified";
    o->engine = "cpu";
    o->host = "127.0.0.1";
    o->scenario = -1;
    o->batch = 8;
    o->steps = 1000;
    o->trials = 4;
    o->eval_steps = 256;
    o->eval_interval = 0;
    o->updates_per_step = 1;
    o->checkpoint_interval = 500;
    o->status_interval = 10;
    o->serve_port = 8765;
    o->tail = 60;
    o->seed = 42;
    for (int i = 2; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) return -1;
        if (!strcmp(o->command, "ipc-cmd") && arg[0] != '-' && !o->output) {
            o->output = arg;
            continue;
        }
        if (!strcmp(arg, "--tui")) { o->tui = 1; continue; }
        if (!strcmp(arg, "--no-tui")) { o->tui = 0; continue; }
        if (!strcmp(arg, "--serve")) { o->serve = 1; continue; }
        if (!strcmp(arg, "--no-serve")) { o->serve = 0; continue; }
        if (!strcmp(arg, "--indefinite")) { o->indefinite = 1; continue; }
        if (!strcmp(arg, "--no-save")) { o->no_save = 1; continue; }
        if (i + 1 >= argc) {
            if (!strcmp(o->command, "ipc-cmd")) { o->output = arg; continue; }
            fprintf(stderr, "Option %s requires a value\n", arg); return 0;
        }
        const char *value = argv[++i];
        if (!strcmp(arg, "--season-dir")) o->season_dir = value;
        else if (!strcmp(arg, "--checkpoint") || !strcmp(arg, "--resume-from")) o->checkpoint = value;
        else if (!strcmp(arg, "--output")) o->output = value;
        else if (!strcmp(arg, "--vehicle-mode")) o->vehicle_mode = value;
        else if (!strcmp(arg, "--engine")) o->engine = value;
        else if (!strcmp(arg, "--serve-host") || !strcmp(arg, "--host")) o->host = value;
        else if (!strcmp(arg, "--scenario")) o->scenario = atoi(value);
        else if (!strcmp(arg, "--batch")) { o->batch = atoi(value); o->batch_explicit = 1; }
        else if (!strcmp(arg, "--steps") || !strcmp(arg, "--max-steps")) o->steps = atoi(value);
        else if (!strcmp(arg, "--trials")) o->trials = atoi(value);
        else if (!strcmp(arg, "--eval-steps")) o->eval_steps = atoi(value);
        else if (!strcmp(arg, "--eval-interval")) o->eval_interval = atoi(value);
        else if (!strcmp(arg, "--updates-per-step")) o->updates_per_step = atoi(value);
        else if (!strcmp(arg, "--checkpoint-interval")) o->checkpoint_interval = atoi(value);
        else if (!strcmp(arg, "--status-interval")) o->status_interval = atoi(value);
        else if (!strcmp(arg, "--serve-port") || !strcmp(arg, "--port")) o->serve_port = atoi(value);
        else if (!strcmp(arg, "--seed")) o->seed = (uint64_t)strtoull(value, NULL, 10);
        else if (!strcmp(arg, "--tail")) o->tail = atoi(value);
        else if (!strcmp(arg, "--gpu-count")) o->gpu_count = atoi(value);
        else if (!strcmp(arg, "--gpu-device")) o->gpu_device = atoi(value);
        else if (!strcmp(arg, "--gpu-batch")) o->gpu_batch = atoi(value);
        else if (!strcmp(arg, "--gpu-replay-capacity")) o->gpu_replay_capacity = atoi(value);
        else { fprintf(stderr, "Unknown option: %s\n", arg); return 0; }
    }
    if (!strcmp(o->engine, "cuda-p3000") && !o->batch_explicit) o->batch = 32;
    if (o->batch < 1 || o->batch > 1024 || o->steps < 1 || o->trials < 1 ||
        o->eval_steps < 1 || o->eval_interval < 0 || o->updates_per_step < 0 ||
        o->checkpoint_interval < 1 || o->status_interval < 1 || o->serve_port < 1 || o->serve_port > 65535 ||
        o->gpu_count < 0 || o->gpu_count > 4 || o->gpu_device < 0 ||
        o->gpu_batch < 0 || o->gpu_replay_capacity < 0 ||
        (strcmp(o->engine, "cpu") && strcmp(o->engine, "cuda-618m") &&
         strcmp(o->engine, "cuda-p3000"))) return 0;
    return 1;
}

static int scenario_for(const Options *o, uint64_t offset) {
    static const int dual[] = {0, 1, 4, 17, 20, 21, 19, 18, 14, 3, 23, 25, 26, 24, 27, 22};
    static const int bwb[] = {23, 25, 26, 24, 27, 22};
    static const int quad[] = {0, 1, 4, 17, 20, 21, 19, 18, 14, 3};
    if (o->scenario >= 0) return o->scenario;
    if (!strcmp(o->vehicle_mode, "bwb")) return bwb[offset % (sizeof(bwb) / sizeof(bwb[0]))];
    if (!strcmp(o->vehicle_mode, "quad_jet")) return quad[offset % (sizeof(quad) / sizeof(quad[0]))];
    return dual[offset % (sizeof(dual) / sizeof(dual[0]))];
}

static int is_bwb(int scenario) { return scenario >= 22 && scenario <= 27; }
static int is_arena(int scenario) { return scenario == 3 || scenario == 14 || scenario == 18 || scenario == 22 || scenario == 24 || scenario == 27; }

typedef struct {
    float *obs, *mission, *priv, *next_obs, *next_mission, *next_priv;
    float *feature, *next_feature, *opp_feature;
    float *opp_obs, *opp_mission, *reference, *actions, *normalized, *opp_actions;
    float *rewards;
    uint8_t *cameras, *next_cameras, *opp_cameras, *terminated, *truncated, *opp_mask;
    int *scenarios;
    double *episode_returns;
} Buffers;

static void free_buffers(Buffers *b) {
    free(b->obs); free(b->mission); free(b->priv); free(b->next_obs); free(b->next_mission);
    free(b->feature); free(b->next_feature); free(b->opp_feature);
    free(b->next_priv); free(b->opp_obs); free(b->opp_mission); free(b->reference);
    free(b->actions); free(b->normalized); free(b->opp_actions); free(b->rewards);
    free(b->cameras); free(b->next_cameras); free(b->opp_cameras);
    free(b->terminated); free(b->truncated); free(b->opp_mask); free(b->scenarios);
    free(b->episode_returns);
    memset(b, 0, sizeof(*b));
}

static int alloc_buffers(Buffers *b, int batch) {
    memset(b, 0, sizeof(*b));
#define TB_ALLOC(field, n, type) b->field = (type *)calloc((size_t)(batch) * (n), sizeof(type)); if (!b->field) goto fail
    TB_ALLOC(obs, TB_OBSERVATION_DIM, float);
    TB_ALLOC(mission, TB_MISSION_DIM, float);
    TB_ALLOC(priv, TB_PRIVILEGED_DIM, float);
    TB_ALLOC(next_obs, TB_OBSERVATION_DIM, float);
    TB_ALLOC(next_mission, TB_MISSION_DIM, float);
    TB_ALLOC(next_priv, TB_PRIVILEGED_DIM, float);
    TB_ALLOC(feature, TB_TRAIN_FEATURE_DIM, float);
    TB_ALLOC(next_feature, TB_TRAIN_FEATURE_DIM, float);
    TB_ALLOC(opp_feature, TB_TRAIN_FEATURE_DIM, float);
    TB_ALLOC(opp_obs, TB_OBSERVATION_DIM, float);
    TB_ALLOC(opp_mission, TB_MISSION_DIM, float);
    TB_ALLOC(reference, TB_ACTION_DIM, float);
    TB_ALLOC(actions, TB_ACTION_DIM, float);
    TB_ALLOC(normalized, TB_ACTION_DIM, float);
    TB_ALLOC(opp_actions, TB_ACTION_DIM, float);
    TB_ALLOC(rewards, 1, float);
    TB_ALLOC(cameras, TB_CAMERA_PIXELS, uint8_t);
    TB_ALLOC(next_cameras, TB_CAMERA_PIXELS, uint8_t);
    TB_ALLOC(opp_cameras, TB_CAMERA_PIXELS, uint8_t);
    TB_ALLOC(terminated, 1, uint8_t);
    TB_ALLOC(truncated, 1, uint8_t);
    TB_ALLOC(opp_mask, 1, uint8_t);
    TB_ALLOC(scenarios, 1, int);
    TB_ALLOC(episode_returns, 1, double);
#undef TB_ALLOC
    return 1;
fail:
    free_buffers(b);
    return 0;
}

static void poll_commands(const Options *o, int *pause, int *stop, int *save, int *eval, int *record, int *indefinite, double *last_seq) {
    char path[1400];
    if (join_path(path, sizeof(path), o->season_dir, "training_control.json")) {
        char *control = read_text(path, 4096);
        if (control) {
            if (strstr(control, "PAUSED") || strstr(control, "\"pause\"")) *pause = 1;
            if (strstr(control, "TRAINING_ACTIVE") || strstr(control, "\"resume\"")) *pause = 0;
            if (strstr(control, "STOPPED_SAVED_UNLOADED") || strstr(control, "\"stop\"")) *stop = 1;
            if (contains_true(control, "save_checkpoint_now")) *save = 1;
            if (contains_true(control, "run_eval_now")) *eval = 1;
            if (contains_true(control, "record_replay_now")) *record = 1;
            if (contains_true(control, "indefinite_training")) *indefinite = 1;
            free(control);
            remove(path);
        }
    }
    if (join_path(path, sizeof(path), o->season_dir, "ipc_command.json")) {
        char *cmd = read_text(path, 4096);
        if (cmd) {
            double seq = 0;
            if (json_number(cmd, "seq", &seq) && seq > *last_seq) {
                *last_seq = seq;
                if (strstr(cmd, "\"pause\"")) *pause = 1;
                if (strstr(cmd, "\"resume\"")) *pause = 0;
                if (strstr(cmd, "\"stop\"")) *stop = 1;
                if (strstr(cmd, "\"force_ckpt\"")) *save = 1;
                if (strstr(cmd, "\"force_eval\"")) *eval = 1;
                if (strstr(cmd, "\"force_replay\"")) *record = 1;
                if (strstr(cmd, "\"toggle_indefinite\"")) *indefinite = !*indefinite;
                char ack[256];
                snprintf(ack, sizeof(ack), "{\"ack_seq\":%.0f}", seq);
                char ack_path[1400];
                if (join_path(ack_path, sizeof(ack_path), o->season_dir, "ipc_ack.json")) write_atomic(ack_path, ack);
            }
            free(cmd);
        }
    }
}

static void write_status(const Options *o, void *tui, uint64_t env_steps, uint64_t episodes,
                         uint64_t grad_steps, double elapsed, double return_sum, const tb_train_metrics *m, int paused) {
    char status[2048], path[1400];
    snprintf(status, sizeof(status),
        "{\"state\":\"%s\",\"vehicle_mode\":\"%s\",\"env_steps\":%llu,\"grad_steps\":%llu,"
        "\"cumulative_episodes\":%llu,\"sps\":%.3f,\"episode_return\":%.6f,"
        "\"actor_loss\":%.6f,\"critic_loss\":%.6f,\"alpha\":%.6f,\"qual_rate\":0.0,"
        "\"architecture\":\"%s\"}",
        paused ? "PAUSED" : "TRAINING_ACTIVE", o->vehicle_mode,
        (unsigned long long)env_steps, (unsigned long long)grad_steps,
        (unsigned long long)episodes, elapsed > 0 ? (double)env_steps / elapsed : 0.0,
        episodes ? return_sum / (double)episodes : 0.0,
        m ? m->actor_loss : 0.0f, m ? m->critic_loss : 0.0f, m ? m->alpha : 0.0f,
        !strcmp(o->engine, "cuda-618m") ? "CUDA 617973042 parameter SAC" :
        !strcmp(o->engine, "cuda-p3000") ? "CUDA sensors with compact C11 SAC" :
        "C11 compact multimodal SAC");
    if (join_path(path, sizeof(path), o->season_dir, "live_status.json")) write_atomic(path, status);
    if (join_path(path, sizeof(path), o->season_dir, "ipc_state.json")) write_atomic(path, status);
    tb_tui_record_status_json(tui, status);
}

static int save_checkpoint(const Options *o, tb_sac_agent *agent) {
    char folder[1400], path[1600];
    if (!join_path(folder, sizeof(folder), o->season_dir, "checkpoints") || !make_dirs(folder) ||
        !join_path(path, sizeof(path), folder, "ckpt_latest.tbc")) return 0;
    return tb_sac_save(agent, path);
}

/* Run held-out rollouts in a separate simulator. The live training batch and
 * replay buffer remain at their current step. Fixed trial seeds make reports
 * comparable across policy snapshots. */
static int capture_training_evaluation(const Options *o, tb_sac_agent *agent,
                                       uint64_t env_steps, char *out_path, size_t out_cap) {
    Buffers b = {0};
    void *sim = tb_create(1, o->seed + UINT64_C(1000000));
    if (!sim || !alloc_buffers(&b, 1)) { tb_destroy(sim); return 0; }
    double total_return = 0.0;
    int completed = 0;
    int ok = 1;
    for (int trial = 0; trial < o->trials && ok; ++trial) {
        const int scenario = scenario_for(o, (uint64_t)trial);
        if (tb_reset_one(sim, 0, scenario, o->seed + UINT64_C(1000000) + (uint64_t)trial)) {
            ok = 0; break;
        }
        double episode_return = 0.0;
        for (int step = 0; step < o->eval_steps; ++step) {
            if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL) ||
                !tb_sac_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized, NULL)) {
                ok = 0; break;
            }
            tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
            if (tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated)) {
                ok = 0; break;
            }
            episode_return += b.rewards[0];
            if (b.terminated[0] || b.truncated[0]) { ++completed; break; }
        }
        total_return += episode_return;
    }
    if (ok) {
        char report[1024];
        const double mean_return = total_return / (double)o->trials;
        const int n = snprintf(report, sizeof(report),
            "{\"env_steps\":%llu,\"grad_steps\":%llu,\"trials\":%d,\"max_steps_per_trial\":%d,"
            "\"mean_return\":%.8g,\"completed\":%d,\"completion_rate\":%.8g}\n",
            (unsigned long long)env_steps, (unsigned long long)tb_sac_update_count(agent),
            o->trials, o->eval_steps, mean_return, completed, (double)completed / (double)o->trials);
        ok = n > 0 && (size_t)n < sizeof(report) &&
             join_path(out_path, out_cap, o->season_dir, "evaluation_latest.json") &&
             write_atomic(out_path, report);
    }
    free_buffers(&b);
    tb_destroy(sim);
    return ok;
}

static int capture_training_replay(const Options *o, tb_sac_agent *agent,
                                   uint64_t env_steps, char *out_path, size_t out_cap) {
    char filename[96];
    snprintf(filename, sizeof(filename), "replay_step_%llu.jsonl", (unsigned long long)env_steps);
    if (!join_path(out_path, out_cap, o->season_dir, filename)) return 0;
    FILE *file = fopen(out_path, "wb");
    if (!file) return 0;
    Buffers b = {0};
    void *sim = tb_create(1, o->seed + env_steps + UINT64_C(2000000));
    int ok = sim && alloc_buffers(&b, 1);
    const int scenario = scenario_for(o, env_steps);
    if (ok) ok = tb_reset_one(sim, 0, scenario, o->seed + env_steps + UINT64_C(2000000)) == 0;
    for (int step = 0; ok && step < o->eval_steps; ++step) {
        double truth[TB_TRUTH_DIM];
        if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, truth) ||
            !tb_sac_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized, NULL)) {
            ok = 0; break;
        }
        tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
        if (tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated)) {
            ok = 0; break;
        }
        if (fprintf(file,
            "{\"step\":%d,\"env_steps\":%llu,\"scenario\":%d,\"reward\":%.8g,\"done\":%s,\"action\":[",
            step, (unsigned long long)env_steps, scenario, b.rewards[0],
            b.terminated[0] || b.truncated[0] ? "true" : "false") < 0) { ok = 0; break; }
        for (int k = 0; k < TB_ACTION_DIM; ++k)
            if (fprintf(file, "%s%.8g", k ? "," : "", b.actions[k]) < 0) ok = 0;
        if (fputs("],\"truth\":[", file) == EOF) ok = 0;
        for (int k = 0; k < TB_TRUTH_DIM; ++k)
            if (fprintf(file, "%s%.8g", k ? "," : "", truth[k]) < 0) ok = 0;
        if (fputs("]}\n", file) == EOF) ok = 0;
        if (!ok || b.terminated[0] || b.truncated[0]) break;
    }
    if (fclose(file) != 0) ok = 0;
    if (!ok) remove(out_path);
    free_buffers(&b);
    tb_destroy(sim);
    return ok;
}

static void process_training_captures(const Options *o, tb_sac_agent *agent, void *tui,
                                      uint64_t env_steps, int *force_eval, int *force_record,
                                      int periodic_eval) {
    if (*force_eval || periodic_eval) {
        char path[1600], message[1800];
        if (capture_training_evaluation(o, agent, env_steps, path, sizeof(path))) {
            snprintf(message, sizeof(message), "Evaluation saved: %s", path);
            tb_tui_log(tui, message, 1);
        } else {
            tb_tui_log(tui, "Evaluation failed; see simulator or output directory.", 1);
        }
        *force_eval = 0;
    }
    if (*force_record) {
        char path[1600], message[1800];
        if (capture_training_replay(o, agent, env_steps, path, sizeof(path))) {
            snprintf(message, sizeof(message), "Replay saved: %s", path);
            tb_tui_log(tui, message, 1);
        } else {
            tb_tui_log(tui, "Replay capture failed; see simulator or output directory.", 1);
        }
        *force_record = 0;
    }
}

static int run_training(const Options *o) {
#ifdef TURBINATOR_HAS_CUDA
    tb_cuda_feature_batch *cuda_features = NULL;
#endif
    void *tui = tb_tui_create(o->season_dir, o->tui, o->indefinite ? -1 : 600, 100);
    if (!tui) { fprintf(stderr, "Cannot create season directory\n"); return 1; }
    tb_http_server http = {0};
    if (o->serve && !tb_http_start(&http, o->season_dir, o->host, (unsigned)o->serve_port))
        fprintf(stderr, "HTTP listener unavailable at %s:%d\n", o->host, o->serve_port);
    tb_train_config cfg; tb_train_config_default(&cfg); cfg.seed = o->seed;
    char default_checkpoint[1600];
    snprintf(default_checkpoint, sizeof(default_checkpoint), "%s/checkpoints/ckpt_latest.tbc", o->season_dir);
    int default_checkpoint_exists = 0;
    if (!o->checkpoint) {
        FILE *probe = fopen(default_checkpoint, "rb");
        if (probe) { default_checkpoint_exists = 1; fclose(probe); }
    }
    tb_sac_agent *agent = o->checkpoint ? tb_sac_load(o->checkpoint) : tb_sac_load(default_checkpoint);
    if ((o->checkpoint || default_checkpoint_exists) && !agent) {
        fprintf(stderr, "Cannot load checkpoint: %s\n", o->checkpoint ? o->checkpoint : default_checkpoint);
        tb_tui_destroy(tui); tb_http_stop(&http); return 1;
    }
    if (!agent) agent = tb_sac_create(&cfg);
    void *sim = tb_create(o->batch, o->seed);
    Buffers b;
    if (!agent || !sim || !alloc_buffers(&b, o->batch)) {
        fprintf(stderr, "Failed to initialize trainer: %s\n", tb_last_error());
        tb_sac_destroy(agent); tb_destroy(sim); tb_tui_destroy(tui); tb_http_stop(&http); return 1;
    }
#ifdef TURBINATOR_HAS_CUDA
    if (!strcmp(o->engine, "cuda-p3000")) {
        cuda_features = tb_cuda_feature_batch_create(o->gpu_device, (size_t)o->batch);
        if (!cuda_features) {
            fprintf(stderr, "Cannot initialize CUDA feature extractor on device %d\n", o->gpu_device);
            goto failure;
        }
    }
#endif
    for (int i = 0; i < o->batch; ++i) {
        b.scenarios[i] = scenario_for(o, (uint64_t)i);
        if (tb_reset_one(sim, i, b.scenarios[i], o->seed + (uint64_t)i)) { fprintf(stderr, "Simulator reset failed\n"); goto failure; }
    }
    if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL)) goto failure;
#ifdef TURBINATOR_HAS_CUDA
    if (cuda_features && !tb_cuda_extract_features_host(cuda_features, b.obs, b.mission,
            b.cameras, (size_t)o->batch, b.feature)) {
        fprintf(stderr, "CUDA feature extraction failed: %s\n", tb_cuda_feature_batch_error(cuda_features));
        goto failure;
    }
#endif
    uint64_t env_steps = 0, episodes = 0, last_reset = 0;
    double return_sum = 0, started = now_seconds(), seq = 0;
    int paused = 0, stop = 0, force_save = 0, force_eval = 0, force_record = 0, indefinite = o->indefinite;
    int last_tui_paused = 0, last_tui_indefinite = indefinite;
    tb_train_metrics metrics = {0};
    for (int step = 0; !stop && (indefinite || step < o->steps); ++step) {
        tb_tui_poll_controls(tui, 0);
        int flags = tb_tui_get_state_flags(tui, 1);
        int tui_paused = (flags & TB_TUI_PAUSED) != 0;
        int tui_indefinite = (flags & TB_TUI_INDEFINITE) != 0;
        if (tui_paused != last_tui_paused) paused = tui_paused;
        if (tui_indefinite != last_tui_indefinite) indefinite = tui_indefinite;
        last_tui_paused = tui_paused;
        last_tui_indefinite = tui_indefinite;
        if (flags & TB_TUI_STOP) stop = 1;
        if (flags & TB_TUI_CHECKPOINT) force_save = 1;
        if (flags & TB_TUI_EVALUATE) force_eval = 1;
        if (flags & TB_TUI_RECORD) force_record = 1;
        poll_commands(o, &paused, &stop, &force_save, &force_eval, &force_record, &indefinite, &seq);
        if (stop) break;
        if (paused) {
            process_training_captures(o, agent, tui, env_steps, &force_eval, &force_record, 0);
            tb_http_poll(&http);
            write_status(o, tui, env_steps, episodes, tb_sac_update_count(agent), now_seconds() - started, return_sum, &metrics, 1);
            tb_tui_render_if_due(tui, (now_seconds() - started) / 60.0, http.running ? "HTTP active" : "HTTP disabled", 0);
            sleep_ms(100);
            --step;
            continue;
        }
        if (tb_reference_action_batch(sim, b.reference) ||
            tb_observe_opponents_training_batch(sim, b.opp_obs, b.opp_mission, b.opp_cameras)) goto failure;
#ifdef TURBINATOR_HAS_CUDA
        int any_arena = 0;
        for (int i = 0; i < o->batch; ++i) any_arena |= is_arena(b.scenarios[i]);
        if (cuda_features && any_arena && !tb_cuda_extract_features_host(cuda_features, b.opp_obs,
                b.opp_mission, b.opp_cameras, (size_t)o->batch, b.opp_feature)) {
            fprintf(stderr, "CUDA opponent features failed: %s\n", tb_cuda_feature_batch_error(cuda_features));
            goto failure;
        }
#endif
        int any_done = 0;
        for (int i = 0; i < o->batch; ++i) {
            float policy[TB_ACTION_DIM], ref_norm[TB_ACTION_DIM];
            size_t a = (size_t)i * TB_ACTION_DIM;
            int acted = 0;
#ifdef TURBINATOR_HAS_CUDA
            if (cuda_features)
                acted = tb_sac_act_features(agent,
                    b.feature + (size_t)i * TB_TRAIN_FEATURE_DIM, 0, policy, NULL);
            else
#endif
                acted = tb_sac_act(agent, b.obs + (size_t)i * TB_OBSERVATION_DIM,
                    b.mission + (size_t)i * TB_MISSION_DIM,
                    b.cameras + (size_t)i * TB_CAMERA_PIXELS, 0, policy, NULL);
            if (!acted) goto failure;
            tb_train_action_from_physical(b.reference + a, is_bwb(b.scenarios[i]), ref_norm);
            float guide = env_steps < 1024 ? 0.75f : 0.05f;
            for (int k = 0; k < TB_ACTION_DIM; ++k) b.normalized[a + (size_t)k] =
                guide * ref_norm[k] + (1.0f - guide) * policy[k];
            tb_train_action_to_physical(b.normalized + a, is_bwb(b.scenarios[i]), b.actions + a);
            b.opp_mask[i] = (uint8_t)is_arena(b.scenarios[i]);
            if (b.opp_mask[i]) {
                float opp_norm[TB_ACTION_DIM];
                int opponent_acted = 0;
#ifdef TURBINATOR_HAS_CUDA
                if (cuda_features)
                    opponent_acted = tb_sac_act_features(agent,
                        b.opp_feature + (size_t)i * TB_TRAIN_FEATURE_DIM,
                        0, opp_norm, NULL);
                else
#endif
                    opponent_acted = tb_sac_act(agent,
                        b.opp_obs + (size_t)i * TB_OBSERVATION_DIM,
                        b.opp_mission + (size_t)i * TB_MISSION_DIM,
                        b.opp_cameras + (size_t)i * TB_CAMERA_PIXELS, 0, opp_norm, NULL);
                if (!opponent_acted) goto failure;
                tb_train_action_to_physical(opp_norm, is_bwb(b.scenarios[i]), b.opp_actions + a);
            }
        }
        /* Refresh every arena opponent action on every simulator step. */
        if (tb_set_opponent_actions_masked(sim, b.opp_actions, b.opp_mask)) goto failure;
        if (tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated)) {
            fprintf(stderr, "Simulator step failed: %s\n", tb_last_error()); goto failure;
        }
        if (tb_observe_training_batch(sim, b.next_obs, b.next_mission, b.next_cameras, b.next_priv, NULL)) goto failure;
#ifdef TURBINATOR_HAS_CUDA
        if (cuda_features && !tb_cuda_extract_features_host(cuda_features, b.next_obs,
                b.next_mission, b.next_cameras, (size_t)o->batch, b.next_feature)) {
            fprintf(stderr, "CUDA next features failed: %s\n", tb_cuda_feature_batch_error(cuda_features));
            goto failure;
        }
#endif
        for (int i = 0; i < o->batch; ++i) {
            size_t a = (size_t)i * TB_ACTION_DIM;
            float ref_norm[TB_ACTION_DIM];
            tb_train_action_from_physical(b.reference + a, is_bwb(b.scenarios[i]), ref_norm);
            int done = b.terminated[i] || b.truncated[i];
            any_done |= done;
            int added = 0;
#ifdef TURBINATOR_HAS_CUDA
            if (cuda_features)
                added = tb_sac_add_transition_features(agent,
                    b.feature + (size_t)i * TB_TRAIN_FEATURE_DIM,
                    b.priv + (size_t)i * TB_PRIVILEGED_DIM,
                    b.normalized + a, ref_norm, b.rewards[i],
                    b.next_feature + (size_t)i * TB_TRAIN_FEATURE_DIM,
                    b.next_priv + (size_t)i * TB_PRIVILEGED_DIM, done);
            else
#endif
                added = tb_sac_add_transition(agent,
                    b.obs + (size_t)i * TB_OBSERVATION_DIM,
                    b.mission + (size_t)i * TB_MISSION_DIM,
                    b.cameras + (size_t)i * TB_CAMERA_PIXELS,
                    b.priv + (size_t)i * TB_PRIVILEGED_DIM,
                    b.normalized + a, ref_norm, b.rewards[i],
                    b.next_obs + (size_t)i * TB_OBSERVATION_DIM,
                    b.next_mission + (size_t)i * TB_MISSION_DIM,
                    b.next_cameras + (size_t)i * TB_CAMERA_PIXELS,
                    b.next_priv + (size_t)i * TB_PRIVILEGED_DIM, done);
            if (!added) goto failure;
            b.episode_returns[i] += b.rewards[i];
            if (done) {
                ++episodes;
                return_sum += b.episode_returns[i];
                b.episode_returns[i] = 0.0;
                b.scenarios[i] = scenario_for(o, (uint64_t)i + ++last_reset);
                if (tb_reset_one(sim, i, b.scenarios[i], o->seed + env_steps + (uint64_t)i + 1)) goto failure;
            }
        }
#ifdef TURBINATOR_HAS_CUDA
        if (cuda_features && !any_done) {
            float *tmp = b.obs; b.obs = b.next_obs; b.next_obs = tmp;
            tmp = b.mission; b.mission = b.next_mission; b.next_mission = tmp;
            tmp = b.priv; b.priv = b.next_priv; b.next_priv = tmp;
            uint8_t *camera_tmp = b.cameras; b.cameras = b.next_cameras; b.next_cameras = camera_tmp;
            tmp = b.feature; b.feature = b.next_feature; b.next_feature = tmp;
        } else
#endif
        if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL)) goto failure;
#ifdef TURBINATOR_HAS_CUDA
        if (cuda_features && any_done &&
            !tb_cuda_extract_features_host(cuda_features, b.obs, b.mission,
                                           b.cameras, (size_t)o->batch, b.feature)) {
            fprintf(stderr, "CUDA feature extraction failed: %s\n",
                    tb_cuda_feature_batch_error(cuda_features));
            goto failure;
        }
#endif
        env_steps += (uint64_t)o->batch;
        if (tb_sac_replay_size(agent) >= 64) {
            for (int u = 0; u < o->updates_per_step; ++u)
                if (!tb_sac_update(agent, 32, &metrics)) goto failure;
        }
        if (force_save || (!o->no_save && (step + 1) % o->checkpoint_interval == 0)) {
            if (!save_checkpoint(o, agent)) fprintf(stderr, "Checkpoint save failed\n");
            force_save = 0;
        }
        process_training_captures(o, agent, tui, env_steps, &force_eval, &force_record,
                                  o->eval_interval > 0 && (step + 1) % o->eval_interval == 0);
        if ((step + 1) % o->status_interval == 0 || step == 0) {
            write_status(o, tui, env_steps, episodes, tb_sac_update_count(agent), now_seconds() - started, return_sum, &metrics, 0);
            tb_tui_render_if_due(tui, (now_seconds() - started) / 60.0, http.running ? "HTTP active" : "HTTP disabled", 1);
            if (!o->tui) printf("steps=%llu replay=%zu updates=%llu reward=%.4f\n",
                (unsigned long long)env_steps, tb_sac_replay_size(agent),
                (unsigned long long)tb_sac_update_count(agent), b.rewards[0]);
        }
        tb_http_poll(&http);
    }
    if (!o->no_save) save_checkpoint(o, agent);
    write_status(o, tui, env_steps, episodes, tb_sac_update_count(agent), now_seconds() - started, return_sum, &metrics, 0);
    printf("{\"env_steps\":%llu,\"grad_steps\":%llu,\"episodes\":%llu}\n",
           (unsigned long long)env_steps, (unsigned long long)tb_sac_update_count(agent), (unsigned long long)episodes);
    free_buffers(&b); tb_destroy(sim); tb_sac_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui);
#ifdef TURBINATOR_HAS_CUDA
    tb_cuda_feature_batch_destroy(cuda_features);
#endif
    return 0;
failure:
    free_buffers(&b); tb_destroy(sim); tb_sac_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui);
#ifdef TURBINATOR_HAS_CUDA
    tb_cuda_feature_batch_destroy(cuda_features);
#endif
    return 1;
}

#ifdef TURBINATOR_HAS_LARGE_GPU
static int evaluate_gpu_agent(const Options *o, tb_gpu_trainer *agent,
                              uint64_t env_steps, const char *output) {
    void *sim = tb_create(1, o->seed + UINT64_C(1000000));
    Buffers b = {0};
    if (!sim || !alloc_buffers(&b, 1)) { tb_destroy(sim); return 0; }
    double total = 0.0;
    int completed = 0, ok = 1;
    for (int trial = 0; trial < o->trials && ok; ++trial) {
        int scenario = scenario_for(o, (uint64_t)trial);
        if (tb_reset_one(sim, 0, scenario, o->seed + UINT64_C(1000000) + (uint64_t)trial)) {
            ok = 0; break;
        }
        double ret = 0.0;
        for (int step = 0; step < o->eval_steps; ++step) {
            if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL) ||
                !tb_gpu_trainer_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized)) {
                ok = 0; break;
            }
            tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
            if (tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated)) {
                ok = 0; break;
            }
            ret += b.rewards[0];
            if (b.terminated[0] || b.truncated[0]) { ++completed; break; }
        }
        total += ret;
    }
    if (ok) {
        char report[1024];
        int n = snprintf(report, sizeof(report),
            "{\"env_steps\":%llu,\"grad_steps\":%llu,\"trials\":%d,\"max_steps_per_trial\":%d,"
            "\"mean_return\":%.8g,\"completed\":%d,\"completion_rate\":%.8g}\n",
            (unsigned long long)env_steps,
            (unsigned long long)tb_gpu_trainer_update_count(agent), o->trials, o->eval_steps,
            total / (double)o->trials, completed, (double)completed / (double)o->trials);
        ok = n > 0 && (size_t)n < sizeof(report) && write_atomic(output, report);
        if (ok) fputs(report, stdout);
    }
    free_buffers(&b); tb_destroy(sim);
    return ok;
}

static int record_gpu_agent(const Options *o, tb_gpu_trainer *agent,
                            uint64_t env_steps, const char *output) {
    FILE *file = fopen(output, "wb");
    if (!file) return 0;
    void *sim = tb_create(1, o->seed + env_steps + UINT64_C(2000000));
    Buffers b = {0};
    int ok = sim && alloc_buffers(&b, 1);
    int scenario = scenario_for(o, env_steps);
    if (ok) ok = tb_reset_one(sim, 0, scenario, o->seed + env_steps + UINT64_C(2000000)) == 0;
    for (int step = 0; ok && step < o->eval_steps; ++step) {
        double truth[TB_TRUTH_DIM];
        if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, truth) ||
            !tb_gpu_trainer_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized)) {
            ok = 0; break;
        }
        tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
        if (tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated)) {
            ok = 0; break;
        }
        if (fprintf(file,
            "{\"step\":%d,\"env_steps\":%llu,\"scenario\":%d,\"reward\":%.8g,\"done\":%s,\"action\":[",
            step, (unsigned long long)env_steps, scenario, b.rewards[0],
            b.terminated[0] || b.truncated[0] ? "true" : "false") < 0) { ok = 0; break; }
        for (int k = 0; k < TB_ACTION_DIM; ++k)
            if (fprintf(file, "%s%.8g", k ? "," : "", b.actions[k]) < 0) ok = 0;
        if (fputs("],\"truth\":[", file) == EOF) ok = 0;
        for (int k = 0; k < TB_TRUTH_DIM; ++k)
            if (fprintf(file, "%s%.8g", k ? "," : "", truth[k]) < 0) ok = 0;
        if (fputs("]}\n", file) == EOF) ok = 0;
        if (b.terminated[0] || b.truncated[0]) break;
    }
    if (fclose(file)) ok = 0;
    if (!ok) remove(output);
    free_buffers(&b); tb_destroy(sim);
    return ok;
}

static int save_gpu_checkpoint(const Options *o, tb_gpu_trainer *agent) {
    char folder[1400];
    if (o->checkpoint) {
        if (!make_dirs(o->checkpoint)) return 0;
        return tb_gpu_trainer_save(agent, o->checkpoint);
    }
    if (!join_path(folder, sizeof(folder), o->season_dir, "checkpoints/gpu_latest") ||
        !make_dirs(folder)) return 0;
    return tb_gpu_trainer_save(agent, folder);
}

static int run_gpu_training(const Options *o) {
    tb_gpu_train_config cfg;
    tb_gpu_train_config_default(&cfg);
    cfg.seed = o->seed;
    if (o->gpu_count) {
        cfg.device_count = o->gpu_count;
        for (int i = 0; i < o->gpu_count; ++i) cfg.device_ids[i] = o->gpu_device + i;
    }
    if (o->gpu_batch) cfg.per_device_batch = o->gpu_batch;
    if (o->gpu_replay_capacity) cfg.per_device_replay_capacity = (size_t)o->gpu_replay_capacity;
    void *tui = tb_tui_create(o->season_dir, o->tui, o->indefinite ? -1 : 600, 100);
    if (!tui) { fprintf(stderr, "Cannot create season directory\n"); return 1; }
    tb_http_server http = {0};
    if (o->serve && !tb_http_start(&http, o->season_dir, o->host, (unsigned)o->serve_port))
        fprintf(stderr, "HTTP listener unavailable at %s:%d\n", o->host, o->serve_port);
    tb_gpu_trainer *agent = tb_gpu_trainer_create(&cfg);
    if (!agent) {
        fprintf(stderr, "CUDA trainer initialization failed\n");
        tb_http_stop(&http); tb_tui_destroy(tui); return 1;
    }
    if (o->checkpoint && !tb_gpu_trainer_load(agent, o->checkpoint)) {
        fprintf(stderr, "Cannot load CUDA checkpoint %s: %s\n", o->checkpoint,
                tb_gpu_trainer_last_error(agent));
        tb_gpu_trainer_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui); return 1;
    }
    if (o->batch < tb_gpu_trainer_device_count(agent)) {
        fprintf(stderr, "CUDA environment batch must be at least the GPU count\n");
        tb_gpu_trainer_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui); return 1;
    }
    void *sim = tb_create(o->batch, o->seed);
    Buffers b = {0};
    float *reference_normalized = (float *)calloc((size_t)o->batch * TB_ACTION_DIM, sizeof(float));
    float *dones = (float *)calloc((size_t)o->batch, sizeof(float));
    if (!sim || !alloc_buffers(&b, o->batch) || !reference_normalized || !dones) {
        fprintf(stderr, "Cannot allocate CUDA training rollout buffers\n");
        goto gpu_failure;
    }
    for (int i = 0; i < o->batch; ++i) {
        b.scenarios[i] = scenario_for(o, (uint64_t)i);
        if (tb_reset_one(sim, i, b.scenarios[i], o->seed + (uint64_t)i)) goto gpu_failure;
    }
    if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL)) goto gpu_failure;
    uint64_t env_steps = 0, episodes = 0, last_reset = 0;
    double return_sum = 0, started = now_seconds(), seq = 0;
    int paused = 0, stop = 0, force_save = 0, force_eval = 0, force_record = 0, indefinite = o->indefinite;
    int last_tui_paused = 0, last_tui_indefinite = indefinite;
    tb_gpu_train_metrics gpu_metrics = {0};
    tb_train_metrics metrics = {0};
    printf("CUDA learner: %llu parameters across %d GPU(s)\n",
           (unsigned long long)tb_gpu_trainer_parameter_count(agent), tb_gpu_trainer_device_count(agent));
    for (int step = 0; !stop && (indefinite || step < o->steps); ++step) {
        tb_tui_poll_controls(tui, 0);
        int flags = tb_tui_get_state_flags(tui, 1);
        int tui_paused = (flags & TB_TUI_PAUSED) != 0;
        int tui_indefinite = (flags & TB_TUI_INDEFINITE) != 0;
        if (tui_paused != last_tui_paused) paused = tui_paused;
        if (tui_indefinite != last_tui_indefinite) indefinite = tui_indefinite;
        last_tui_paused = tui_paused;
        last_tui_indefinite = tui_indefinite;
        if (flags & TB_TUI_STOP) stop = 1;
        if (flags & TB_TUI_CHECKPOINT) force_save = 1;
        if (flags & TB_TUI_EVALUATE) force_eval = 1;
        if (flags & TB_TUI_RECORD) force_record = 1;
        poll_commands(o, &paused, &stop, &force_save, &force_eval, &force_record, &indefinite, &seq);
        if (stop) break;
        if (paused) {
            tb_http_poll(&http);
            write_status(o, tui, env_steps, episodes, tb_gpu_trainer_update_count(agent),
                         now_seconds() - started, return_sum, &metrics, 1);
            tb_tui_render_if_due(tui, (now_seconds() - started) / 60.0,
                                 http.running ? "HTTP active" : "HTTP disabled", 0);
            sleep_ms(100);
            --step;
            continue;
        }
        if (tb_reference_action_batch(sim, b.reference) ||
            tb_observe_opponents_training_batch(sim, b.opp_obs, b.opp_mission, b.opp_cameras) ||
            !tb_gpu_trainer_act(agent, b.obs, b.mission, b.cameras, o->batch, b.normalized)) {
            fprintf(stderr, "CUDA policy inference failed: %s\n", tb_gpu_trainer_last_error(agent));
            goto gpu_failure;
        }
        int any_arena = 0;
        for (int i = 0; i < o->batch; ++i) any_arena |= is_arena(b.scenarios[i]);
        if (any_arena && !tb_gpu_trainer_act(agent, b.opp_obs, b.opp_mission, b.opp_cameras,
                                              o->batch, b.opp_actions)) {
            fprintf(stderr, "CUDA opponent inference failed: %s\n", tb_gpu_trainer_last_error(agent));
            goto gpu_failure;
        }
        for (int i = 0; i < o->batch; ++i) {
            size_t a = (size_t)i * TB_ACTION_DIM;
            tb_train_action_from_physical(b.reference + a, is_bwb(b.scenarios[i]),
                                          reference_normalized + a);
            float guide = env_steps < 1024 ? 0.75f : 0.05f;
            for (int k = 0; k < TB_ACTION_DIM; ++k)
                b.normalized[a + (size_t)k] = guide * reference_normalized[a + (size_t)k] +
                    (1.0f - guide) * b.normalized[a + (size_t)k];
            tb_train_action_to_physical(b.normalized + a, is_bwb(b.scenarios[i]), b.actions + a);
            b.opp_mask[i] = (uint8_t)is_arena(b.scenarios[i]);
            if (b.opp_mask[i]) {
                float normalized[TB_ACTION_DIM];
                memcpy(normalized, b.opp_actions + a, sizeof(normalized));
                tb_train_action_to_physical(normalized, is_bwb(b.scenarios[i]), b.opp_actions + a);
            }
        }
        if (tb_set_opponent_actions_masked(sim, b.opp_actions, b.opp_mask) ||
            tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated) ||
            tb_observe_training_batch(sim, b.next_obs, b.next_mission, b.next_cameras,
                                      b.next_priv, NULL)) goto gpu_failure;
        for (int i = 0; i < o->batch; ++i) {
            int done = b.terminated[i] || b.truncated[i];
            dones[i] = done ? 1.0f : 0.0f;
            b.episode_returns[i] += b.rewards[i];
        }
        tb_gpu_host_batch host = {
            b.obs, b.mission, b.cameras, b.priv, b.normalized, reference_normalized,
            b.rewards, b.next_obs, b.next_mission, b.next_cameras, b.next_priv, dones
        };
        if (!tb_gpu_trainer_add(agent, &host, (size_t)o->batch)) {
            fprintf(stderr, "CUDA replay insertion failed: %s\n", tb_gpu_trainer_last_error(agent));
            goto gpu_failure;
        }
        for (int i = 0; i < o->batch; ++i) {
            if (dones[i] != 0.0f) {
                ++episodes;
                return_sum += b.episode_returns[i];
                b.episode_returns[i] = 0.0;
                b.scenarios[i] = scenario_for(o, (uint64_t)i + ++last_reset);
                if (tb_reset_one(sim, i, b.scenarios[i], o->seed + env_steps + (uint64_t)i + 1))
                    goto gpu_failure;
            }
        }
        if (tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL)) goto gpu_failure;
        env_steps += (uint64_t)o->batch;
        if (tb_gpu_trainer_replay_size(agent) >=
            (size_t)cfg.per_device_batch * (size_t)tb_gpu_trainer_device_count(agent)) {
            for (int u = 0; u < o->updates_per_step; ++u) {
                if (!tb_gpu_trainer_update(agent, &gpu_metrics)) {
                    const char *error = tb_gpu_trainer_last_error(agent);
                    if (error && *error) {
                        fprintf(stderr, "CUDA learner update failed: %s\n", error);
                        goto gpu_failure;
                    }
                    break;
                }
                metrics.critic_loss = gpu_metrics.critic_loss;
                metrics.actor_loss = gpu_metrics.actor_loss;
                metrics.reference_loss = gpu_metrics.auxiliary_loss;
                metrics.q_mean = gpu_metrics.q_mean;
                metrics.alpha = gpu_metrics.alpha;
                metrics.actor_grad_norm = gpu_metrics.gradient_norm;
                metrics.update_count = gpu_metrics.updates;
            }
        }
        if (force_save || (!o->no_save && (step + 1) % o->checkpoint_interval == 0)) {
            if (!save_gpu_checkpoint(o, agent))
                fprintf(stderr, "CUDA checkpoint save failed: %s\n", tb_gpu_trainer_last_error(agent));
            force_save = 0;
        }
        if (force_eval || (o->eval_interval > 0 && (step + 1) % o->eval_interval == 0)) {
            char path[1600];
            if (!join_path(path, sizeof(path), o->season_dir, "evaluation_latest.json") ||
                !evaluate_gpu_agent(o, agent, env_steps, path))
                tb_tui_log(tui, "CUDA evaluation failed", 1);
            else tb_tui_log(tui, "CUDA evaluation saved", 1);
            force_eval = 0;
        }
        if (force_record) {
            char filename[96], path[1600];
            snprintf(filename, sizeof(filename), "replay_step_%llu.jsonl",
                     (unsigned long long)env_steps);
            if (!join_path(path, sizeof(path), o->season_dir, filename) ||
                !record_gpu_agent(o, agent, env_steps, path))
                tb_tui_log(tui, "CUDA replay capture failed", 1);
            else tb_tui_log(tui, "CUDA replay captured", 1);
            force_record = 0;
        }
        if ((step + 1) % o->status_interval == 0 || step == 0) {
            write_status(o, tui, env_steps, episodes, tb_gpu_trainer_update_count(agent),
                         now_seconds() - started, return_sum, &metrics, 0);
            tb_tui_render_if_due(tui, (now_seconds() - started) / 60.0,
                                 http.running ? "HTTP active" : "HTTP disabled", 1);
            if (!o->tui) printf("steps=%llu replay=%zu updates=%llu reward=%.4f\n",
                (unsigned long long)env_steps, tb_gpu_trainer_replay_size(agent),
                (unsigned long long)tb_gpu_trainer_update_count(agent), b.rewards[0]);
        }
        tb_http_poll(&http);
    }
    int saved = o->no_save || save_gpu_checkpoint(o, agent);
    if (!saved)
        fprintf(stderr, "CUDA final checkpoint save failed: %s\n", tb_gpu_trainer_last_error(agent));
    write_status(o, tui, env_steps, episodes, tb_gpu_trainer_update_count(agent),
                 now_seconds() - started, return_sum, &metrics, 0);
    printf("{\"env_steps\":%llu,\"grad_steps\":%llu,\"episodes\":%llu,\"parameters\":%llu}\n",
           (unsigned long long)env_steps,
           (unsigned long long)tb_gpu_trainer_update_count(agent),
           (unsigned long long)episodes,
           (unsigned long long)tb_gpu_trainer_parameter_count(agent));
    free(reference_normalized); free(dones); free_buffers(&b); tb_destroy(sim);
    tb_gpu_trainer_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui);
    return saved ? 0 : 1;
gpu_failure:
    free(reference_normalized); free(dones); free_buffers(&b); tb_destroy(sim);
    tb_gpu_trainer_destroy(agent); tb_http_stop(&http); tb_tui_destroy(tui);
    return 1;
}

static tb_gpu_trainer *load_gpu_agent(const Options *o) {
    tb_gpu_train_config cfg;
    tb_gpu_train_config_default(&cfg);
    cfg.device_count = o->gpu_count ? o->gpu_count : 1;
    for (int i = 0; i < cfg.device_count; ++i) cfg.device_ids[i] = o->gpu_device + i;
    if (o->gpu_batch) cfg.per_device_batch = o->gpu_batch;
    if (o->gpu_replay_capacity) cfg.per_device_replay_capacity = (size_t)o->gpu_replay_capacity;
    cfg.seed = o->seed;
    tb_gpu_trainer *agent = tb_gpu_trainer_create(&cfg);
    if (!agent) { fprintf(stderr, "CUDA trainer initialization failed\n"); return NULL; }
    char default_checkpoint[1600];
    if (!join_path(default_checkpoint, sizeof(default_checkpoint), o->season_dir,
                   "checkpoints/gpu_latest")) {
        tb_gpu_trainer_destroy(agent); return NULL;
    }
    const char *checkpoint = o->checkpoint ? o->checkpoint : default_checkpoint;
    if (!tb_gpu_trainer_load(agent, checkpoint)) {
        fprintf(stderr, "Cannot load CUDA checkpoint %s: %s\n", checkpoint,
                tb_gpu_trainer_last_error(agent));
        tb_gpu_trainer_destroy(agent); return NULL;
    }
    return agent;
}

static int run_gpu_evaluation(const Options *o) {
    tb_gpu_trainer *agent = load_gpu_agent(o);
    if (!agent) return 1;
    char path[1600];
    int ok = o->output || join_path(path, sizeof(path), o->season_dir,
                                    "evaluation_latest.json");
    Options evaluation = *o;
    evaluation.eval_steps = o->steps;
    if (ok) ok = evaluate_gpu_agent(&evaluation, agent, 0, o->output ? o->output : path);
    if (!ok) fprintf(stderr, "CUDA evaluation failed: %s\n", tb_gpu_trainer_last_error(agent));
    tb_gpu_trainer_destroy(agent);
    return ok ? 0 : 1;
}

static int run_gpu_record(const Options *o) {
    tb_gpu_trainer *agent = load_gpu_agent(o);
    if (!agent) return 1;
    const char *output = o->output ? o->output : "record.jsonl";
    Options recording = *o;
    recording.eval_steps = o->steps;
    int ok = record_gpu_agent(&recording, agent, 0, output);
    if (!ok) fprintf(stderr, "CUDA recording failed: %s\n", tb_gpu_trainer_last_error(agent));
    else printf("Recorded %s\n", output);
    tb_gpu_trainer_destroy(agent);
    return ok ? 0 : 1;
}
#endif

static int run_evaluation(const Options *o) {
    tb_sac_agent *agent = o->checkpoint ? tb_sac_load(o->checkpoint) : NULL;
    if (o->checkpoint && !agent) { fprintf(stderr, "Cannot load checkpoint: %s\n", o->checkpoint); return 1; }
    void *sim = tb_create(1, o->seed);
    Buffers b;
    if (!sim || !alloc_buffers(&b, 1)) { tb_sac_destroy(agent); tb_destroy(sim); return 1; }
    double total = 0;
    int completed = 0;
    for (int trial = 0; trial < o->trials; ++trial) {
        int scenario = scenario_for(o, (uint64_t)trial);
        tb_reset_one(sim, 0, scenario, o->seed + (uint64_t)trial);
        double ret = 0;
        for (int step = 0; step < o->steps; ++step) {
            tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, NULL);
            if (agent) {
                tb_sac_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized, NULL);
                tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
            } else tb_reference_action_batch(sim, b.actions);
            tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated);
            ret += b.rewards[0];
            if (b.terminated[0] || b.truncated[0]) { ++completed; break; }
        }
        total += ret;
        printf("trial=%d scenario=%d return=%.6f\n", trial, scenario, ret);
    }
    printf("{\"trials\":%d,\"mean_return\":%.6f,\"completed\":%d}\n", o->trials, total / (double)o->trials, completed);
    free_buffers(&b); tb_destroy(sim); tb_sac_destroy(agent); return 0;
}

static int run_record(const Options *o) {
    const char *output = o->output ? o->output : "record.jsonl";
    FILE *f = fopen(output, "wb");
    if (!f) { perror(output); return 1; }
    tb_sac_agent *agent = o->checkpoint ? tb_sac_load(o->checkpoint) : NULL;
    if (o->checkpoint && !agent) {
        fprintf(stderr, "Cannot load checkpoint: %s\n", o->checkpoint);
        fclose(f); remove(output); return 1;
    }
    void *sim = tb_create(1, o->seed);
    Buffers b;
    if (!sim || !alloc_buffers(&b, 1)) { fclose(f); tb_sac_destroy(agent); tb_destroy(sim); return 1; }
    int scenario = scenario_for(o, 0);
    tb_reset_one(sim, 0, scenario, o->seed);
    for (int step = 0; step < o->steps; ++step) {
        double truth[TB_TRUTH_DIM];
        tb_observe_training_batch(sim, b.obs, b.mission, b.cameras, b.priv, truth);
        if (agent) {
            tb_sac_act(agent, b.obs, b.mission, b.cameras, 1, b.normalized, NULL);
            tb_train_action_to_physical(b.normalized, is_bwb(scenario), b.actions);
        } else tb_reference_action_batch(sim, b.actions);
        tb_step(sim, b.actions, b.next_obs, b.rewards, b.terminated, b.truncated);
        fprintf(f, "{\"step\":%d,\"scenario\":%d,\"reward\":%.8g,\"done\":%s,\"action\":[",
                step, scenario, b.rewards[0], b.terminated[0] || b.truncated[0] ? "true" : "false");
        for (int k = 0; k < TB_ACTION_DIM; ++k) fprintf(f, "%s%.8g", k ? "," : "", b.actions[k]);
        fputs("],\"truth\":[", f);
        for (int k = 0; k < TB_TRUTH_DIM; ++k) fprintf(f, "%s%.8g", k ? "," : "", truth[k]);
        fputs("]}\n", f);
        if (b.terminated[0] || b.truncated[0]) break;
    }
    fclose(f); free_buffers(&b); tb_destroy(sim); tb_sac_destroy(agent);
    printf("Recorded %s\n", output);
    return 0;
}

static int run_benchmark(const Options *o) {
    void *sim = tb_create(o->batch, o->seed);
    Buffers b;
    if (!sim || !alloc_buffers(&b, o->batch)) { tb_destroy(sim); return 1; }
    for (int i = 0; i < o->batch; ++i) tb_reset_one(sim, i, scenario_for(o, (uint64_t)i), o->seed + (uint64_t)i);
    double start = now_seconds();
    for (int step = 0; step < o->steps; ++step) {
        tb_reference_action_batch(sim, b.actions);
        tb_step(sim, b.actions, b.obs, b.rewards, b.terminated, b.truncated);
        for (int i = 0; i < o->batch; ++i) if (b.terminated[i] || b.truncated[i])
            tb_reset_one(sim, i, scenario_for(o, (uint64_t)i + (uint64_t)step), o->seed + (uint64_t)step + (uint64_t)i);
    }
    double elapsed = now_seconds() - start;
    printf("{\"batch\":%d,\"steps\":%d,\"elapsed_s\":%.6f,\"sim_steps_per_s\":%.3f}\n",
        o->batch, o->steps, elapsed, elapsed > 0 ? (double)o->batch * o->steps / elapsed : 0.0);
    free_buffers(&b); tb_destroy(sim); return 0;
}

static int run_tui(const Options *o) {
    void *tui = tb_tui_create(o->season_dir, 1, o->indefinite ? -1 : 600, 100);
    if (!tui) return 1;
    tb_http_server http = {0};
    if (o->serve) tb_http_start(&http, o->season_dir, o->host, (unsigned)o->serve_port);
    double start = now_seconds();
    for (;;) {
        tb_tui_poll_controls(tui, 1);
        tb_tui_sync_from_disk(tui);
        tb_tui_render_if_due(tui, (now_seconds() - start) / 60.0, http.running ? "HTTP active" : "HTTP disabled", 1);
        tb_http_poll(&http);
        if (tb_tui_get_state_flags(tui, 0) & TB_TUI_STOP) break;
        sleep_ms(500);
    }
    tb_http_stop(&http); tb_tui_destroy(tui); return 0;
}

static int run_ipc(const Options *o) {
    char path[1400];
    if (!strcmp(o->command, "ipc-status")) {
        if (!join_path(path, sizeof(path), o->season_dir, "ipc_state.json")) return 1;
        char *text = read_text(path, 1024 * 1024);
        puts(text ? text : "{\"state\":\"UNKNOWN\"}"); free(text); return 0;
    }
    if (!strcmp(o->command, "ipc-logs")) {
        if (!join_path(path, sizeof(path), o->season_dir, "ipc_worker.log")) return 1;
        FILE *f = fopen(path, "rb");
        if (!f) return 0;
        char line[2048], ring[256][2048]; int count = 0;
        while (fgets(line, sizeof(line), f)) { snprintf(ring[count % 256], sizeof(ring[0]), "%s", line); ++count; }
        fclose(f);
        int keep = o->tail < 256 ? o->tail : 256;
        for (int i = count > keep ? count - keep : 0; i < count; ++i) fputs(ring[i % 256], stdout);
        return 0;
    }
    if (!strcmp(o->command, "ipc-cmd")) {
        const char *command = o->output;
        if (!command) { fprintf(stderr, "ipc-cmd requires a command\n"); return 1; }
        const char *known[] = {"pause", "resume", "stop", "force_eval", "force_ckpt", "force_replay", "toggle_indefinite"};
        int valid = 0;
        for (size_t k = 0; k < sizeof(known) / sizeof(known[0]); ++k) if (!strcmp(command, known[k])) valid = 1;
        if (!valid) { fprintf(stderr, "Unknown IPC command\n"); return 1; }
        if (!make_dirs(o->season_dir) || !join_path(path, sizeof(path), o->season_dir, "ipc_command.json")) return 1;
        char body[256];
        snprintf(body, sizeof(body), "{\"seq\":%.0f,\"command\":\"%s\"}", now_seconds() * 1000000.0, command);
        if (!write_atomic(path, body)) return 1;
        puts(body); return 0;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) {
        usage(argv[0]);
        return 0;
    }
    Options o;
    int parsed = parse_args(argc, argv, &o);
    if (parsed <= 0) { usage(argv[0]); return parsed < 0 ? 0 : 2; }
    if (!strcmp(o.command, "network-info")) {
        tb_net_model *model = tb_net_create_618m();
        if (!model) return 1;
        printf("{\"parameters\":%llu,\"actor_parameters\":%llu,\"critic_parameters\":%llu,\"tensors\":%zu,\"valid\":%s}\n",
               (unsigned long long)tb_net_parameter_count(model),
               (unsigned long long)tb_net_actor_parameter_count(model),
               (unsigned long long)tb_net_critic_parameter_count(model),
               tb_net_tensor_count(model), tb_net_validate_layout(model) ? "true" : "false");
        tb_net_destroy(model);
        return 0;
    }
    if (!strcmp(o.command, "train") && !strcmp(o.engine, "cuda-618m")) {
#ifdef TURBINATOR_HAS_LARGE_GPU
        return run_gpu_training(&o);
#else
        fprintf(stderr, "This build has no CUDA training support\n");
        return 1;
#endif
    }
    if (!strcmp(o.command, "train") && !strcmp(o.engine, "cuda-p3000")) {
#ifdef TURBINATOR_HAS_CUDA
        return run_training(&o);
#else
        fprintf(stderr, "This build has no CUDA feature support\n");
        return 1;
#endif
    }
    if (!strcmp(o.command, "train")) return run_training(&o);
    if ((!strcmp(o.command, "evaluate") || !strcmp(o.command, "record")) &&
        !strcmp(o.engine, "cuda-618m")) {
#ifdef TURBINATOR_HAS_LARGE_GPU
        return !strcmp(o.command, "evaluate") ? run_gpu_evaluation(&o) : run_gpu_record(&o);
#else
        fprintf(stderr, "This build has no CUDA training support\n");
        return 1;
#endif
    }
    if (!strcmp(o.command, "evaluate")) return run_evaluation(&o);
    if (!strcmp(o.command, "record")) return run_record(&o);
    if (!strcmp(o.command, "benchmark")) return run_benchmark(&o);
    if (!strcmp(o.command, "tui")) return run_tui(&o);
    if (!strcmp(o.command, "ipc-status") || !strcmp(o.command, "ipc-logs") || !strcmp(o.command, "ipc-cmd")) return run_ipc(&o);
    if (!strcmp(o.command, "init-season")) {
        tb_train_config cfg; tb_train_config_default(&cfg); cfg.seed = o.seed;
        tb_sac_agent *agent = tb_sac_create(&cfg);
        if (!agent) return 1;
        int ok = make_dirs(o.season_dir) && save_checkpoint(&o, agent);
        tb_sac_destroy(agent);
        printf("{\"season_dir\":\"%s\",\"initialized\":%s}\n", o.season_dir, ok ? "true" : "false");
        return ok ? 0 : 1;
    }
    usage(argv[0]); return 2;
}
