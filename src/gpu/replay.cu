#include "turbinator/gpu/replay.h"

#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

struct tb_gpu_replay {
    tb_gpu_ops *ops;
    tb_gpu_sample_batch storage;
    size_t capacity;
    size_t size;
    size_t cursor;
    uint64_t rng;
    uint32_t *device_indices;
    size_t index_capacity;
};

static uint64_t tb_replay_rand(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static int tb_alloc_fields(tb_gpu_ops *ops, tb_gpu_sample_batch *b, size_t count) {
    if (!ops || !b || count == 0) return 0;
    memset(b, 0, sizeof(*b));
    b->count = count;
#define ALLOC_FIELD(field, dim, type) \
    b->field = (type *)tb_gpu_malloc(ops, count * (dim) * sizeof(type)); \
    if (!b->field) return 0
    ALLOC_FIELD(obs, TB_TRAIN_OBSERVATION_DIM, float);
    ALLOC_FIELD(mission, TB_TRAIN_MISSION_DIM, float);
    ALLOC_FIELD(cameras, TB_TRAIN_CAMERA_PIXELS, uint8_t);
    ALLOC_FIELD(privileged, TB_TRAIN_PRIVILEGED_DIM, float);
    ALLOC_FIELD(actions, TB_TRAIN_ACTION_DIM, float);
    ALLOC_FIELD(reference_actions, TB_TRAIN_ACTION_DIM, float);
    ALLOC_FIELD(rewards, 1, float);
    ALLOC_FIELD(next_obs, TB_TRAIN_OBSERVATION_DIM, float);
    ALLOC_FIELD(next_mission, TB_TRAIN_MISSION_DIM, float);
    ALLOC_FIELD(next_cameras, TB_TRAIN_CAMERA_PIXELS, uint8_t);
    ALLOC_FIELD(next_privileged, TB_TRAIN_PRIVILEGED_DIM, float);
    ALLOC_FIELD(dones, 1, float);
#undef ALLOC_FIELD
    return 1;
}
static void tb_free_fields(tb_gpu_ops *ops, tb_gpu_sample_batch *b) {
    if (!ops || !b) return;
    tb_gpu_free(ops, b->obs); tb_gpu_free(ops, b->mission); tb_gpu_free(ops, b->cameras);
    tb_gpu_free(ops, b->privileged); tb_gpu_free(ops, b->actions);
    tb_gpu_free(ops, b->reference_actions); tb_gpu_free(ops, b->rewards);
    tb_gpu_free(ops, b->next_obs); tb_gpu_free(ops, b->next_mission);
    tb_gpu_free(ops, b->next_cameras); tb_gpu_free(ops, b->next_privileged);
    tb_gpu_free(ops, b->dones);
    memset(b, 0, sizeof(*b));
}

extern "C" tb_gpu_replay *tb_gpu_replay_create(tb_gpu_ops *ops, size_t capacity, uint64_t seed) {
    if (!ops || capacity == 0 || capacity > UINT32_MAX) return NULL;
    tb_gpu_replay *r = (tb_gpu_replay *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->ops = ops;
    r->capacity = capacity;
    r->rng = seed ? seed : 1;
    if (!tb_alloc_fields(ops, &r->storage, capacity)) { tb_gpu_replay_destroy(r); return NULL; }
    return r;
}
extern "C" void tb_gpu_replay_destroy(tb_gpu_replay *r) {
    if (!r) return;
    tb_free_fields(r->ops, &r->storage);
    tb_gpu_free(r->ops, r->device_indices);
    free(r);
}
extern "C" size_t tb_gpu_replay_size(const tb_gpu_replay *r) { return r ? r->size : 0; }
extern "C" size_t tb_gpu_replay_capacity(const tb_gpu_replay *r) { return r ? r->capacity : 0; }

static int tb_host_valid(const tb_gpu_host_batch *h) {
    return h && h->obs && h->mission && h->cameras && h->privileged &&
        h->actions && h->reference_actions && h->rewards && h->next_obs &&
        h->next_mission && h->next_cameras && h->next_privileged && h->dones;
}
static int tb_add_segment(tb_gpu_replay *r, const tb_gpu_host_batch *h,
                          size_t host_offset, size_t ring_offset, size_t count) {
    tb_gpu_sample_batch *d = &r->storage;
#define UPLOAD_FIELD(field, dim, type) \
    if (!tb_gpu_h2d(r->ops, d->field + ring_offset * (dim), \
        h->field + host_offset * (dim), count * (dim) * sizeof(type))) return 0
    UPLOAD_FIELD(obs, TB_TRAIN_OBSERVATION_DIM, float);
    UPLOAD_FIELD(mission, TB_TRAIN_MISSION_DIM, float);
    UPLOAD_FIELD(cameras, TB_TRAIN_CAMERA_PIXELS, uint8_t);
    UPLOAD_FIELD(privileged, TB_TRAIN_PRIVILEGED_DIM, float);
    UPLOAD_FIELD(actions, TB_TRAIN_ACTION_DIM, float);
    UPLOAD_FIELD(reference_actions, TB_TRAIN_ACTION_DIM, float);
    UPLOAD_FIELD(rewards, 1, float);
    UPLOAD_FIELD(next_obs, TB_TRAIN_OBSERVATION_DIM, float);
    UPLOAD_FIELD(next_mission, TB_TRAIN_MISSION_DIM, float);
    UPLOAD_FIELD(next_cameras, TB_TRAIN_CAMERA_PIXELS, uint8_t);
    UPLOAD_FIELD(next_privileged, TB_TRAIN_PRIVILEGED_DIM, float);
    UPLOAD_FIELD(dones, 1, float);
#undef UPLOAD_FIELD
    return 1;
}
extern "C" int tb_gpu_replay_add(tb_gpu_replay *r, const tb_gpu_host_batch *host, size_t count) {
    if (!r || !tb_host_valid(host) || count == 0) return 0;
    /* Keep the newest capacity transitions if a caller supplies a huge batch. */
    size_t source_offset = count > r->capacity ? count - r->capacity : 0;
    size_t remaining = count - source_offset;
    size_t first = r->capacity - r->cursor;
    if (first > remaining) first = remaining;
    if (!tb_add_segment(r, host, source_offset, r->cursor, first)) return 0;
    if (remaining > first && !tb_add_segment(r, host, source_offset + first, 0, remaining - first)) return 0;
    r->cursor = (r->cursor + remaining) % r->capacity;
    r->size = r->size + remaining < r->capacity ? r->size + remaining : r->capacity;
    return 1;
}
extern "C" tb_gpu_sample_batch *tb_gpu_sample_batch_create(tb_gpu_ops *ops, size_t count) {
    tb_gpu_sample_batch *b = (tb_gpu_sample_batch *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    if (!tb_alloc_fields(ops, b, count)) { tb_gpu_sample_batch_destroy(ops, b); return NULL; }
    return b;
}
extern "C" void tb_gpu_sample_batch_destroy(tb_gpu_ops *ops, tb_gpu_sample_batch *b) {
    if (!b) return;
    tb_free_fields(ops, b);
    free(b);
}

__global__ static void tb_gather_float_kernel(const float *src, float *dst,
                                               const uint32_t *indices,
                                               size_t count, int dimension) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * (size_t)dimension;
    if (i < total) dst[i] = src[(size_t)indices[i / dimension] * dimension + (i % dimension)];
}
__global__ static void tb_gather_u8_kernel(const uint8_t *src, uint8_t *dst,
                                            const uint32_t *indices,
                                            size_t count, int dimension) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = count * (size_t)dimension;
    if (i < total) dst[i] = src[(size_t)indices[i / dimension] * dimension + (i % dimension)];
}
static int tb_replay_indices(tb_gpu_replay *r, size_t count) {
    if (r->index_capacity < count) {
        tb_gpu_free(r->ops, r->device_indices);
        r->device_indices = (uint32_t *)tb_gpu_malloc(r->ops, count * sizeof(uint32_t));
        if (!r->device_indices) { r->index_capacity = 0; return 0; }
        r->index_capacity = count;
    }
    uint32_t *host = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!host) return 0;
    for (size_t i = 0; i < count; ++i) host[i] = (uint32_t)(tb_replay_rand(&r->rng) % r->size);
    int ok = tb_gpu_h2d(r->ops, r->device_indices, host, count * sizeof(uint32_t));
    free(host);
    return ok;
}
extern "C" int tb_gpu_replay_sample(tb_gpu_replay *r, tb_gpu_sample_batch *out) {
    if (!r || !out || !r->size || !out->count || !tb_replay_indices(r, out->count)) return 0;
    if (cudaSetDevice(tb_gpu_device(r->ops)) != cudaSuccess) return 0;
    cudaStream_t stream = (cudaStream_t)tb_gpu_stream(r->ops);
#define GATHER_FLOAT(field, dim) { \
    size_t n = out->count * (dim); \
    tb_gather_float_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>( \
        r->storage.field, out->field, r->device_indices, out->count, (dim)); \
    if (cudaGetLastError() != cudaSuccess) return 0; }
#define GATHER_U8(field, dim) { \
    size_t n = out->count * (dim); \
    tb_gather_u8_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>( \
        r->storage.field, out->field, r->device_indices, out->count, (dim)); \
    if (cudaGetLastError() != cudaSuccess) return 0; }
    GATHER_FLOAT(obs, TB_TRAIN_OBSERVATION_DIM);
    GATHER_FLOAT(mission, TB_TRAIN_MISSION_DIM);
    GATHER_U8(cameras, TB_TRAIN_CAMERA_PIXELS);
    GATHER_FLOAT(privileged, TB_TRAIN_PRIVILEGED_DIM);
    GATHER_FLOAT(actions, TB_TRAIN_ACTION_DIM);
    GATHER_FLOAT(reference_actions, TB_TRAIN_ACTION_DIM);
    GATHER_FLOAT(rewards, 1);
    GATHER_FLOAT(next_obs, TB_TRAIN_OBSERVATION_DIM);
    GATHER_FLOAT(next_mission, TB_TRAIN_MISSION_DIM);
    GATHER_U8(next_cameras, TB_TRAIN_CAMERA_PIXELS);
    GATHER_FLOAT(next_privileged, TB_TRAIN_PRIVILEGED_DIM);
    GATHER_FLOAT(dones, 1);
#undef GATHER_FLOAT
#undef GATHER_U8
    return tb_gpu_sync(r->ops);
}
