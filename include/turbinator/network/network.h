#ifndef TURBINATOR_NETWORK_NETWORK_H
#define TURBINATOR_NETWORK_NETWORK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TB_NET_D_MODEL 1536
#define TB_NET_HEADS 24
#define TB_NET_HEAD_DIM 64
#define TB_NET_FUSION_LAYERS 12
#define TB_NET_FFN_DIM 6144
#define TB_NET_MOE_MODULES 2
#define TB_NET_MOE_EXPERTS 6
#define TB_NET_MOE_TOP_K 2
#define TB_NET_MOE_EXPERT_DIM 2360
#define TB_NET_ACTION_HORIZON 8
#define TB_NET_ACTION_DIM 15
#define TB_NET_CAMERA_COUNT 4
#define TB_NET_LIDAR_TOKENS 4
#define TB_NET_SENSOR_TOKENS 10
#define TB_NET_WORLD_DIM 2048
#define TB_NET_CRITIC_HIDDEN 2688
#define TB_NET_CRITIC_LAYERS 6
#define TB_NET_ATOMS 64
#define TB_NET_ACTOR_PARAMETERS UINT64_C(528338866)
#define TB_NET_CRITIC_PARAMETERS UINT64_C(89634176)
#define TB_NET_TOTAL_PARAMETERS UINT64_C(617973042)
#define TB_NET_MAX_TENSORS 512

typedef enum {
    TB_NET_MATRIX,
    TB_NET_VECTOR,
    TB_NET_NORM_GAIN,
    TB_NET_NORM_BIAS,
    TB_NET_EMBEDDING,
    TB_NET_CONV_KERNEL
} tb_net_tensor_kind;

typedef struct {
    char name[96];
    uint64_t offset;
    uint64_t count;
    int shape[4];
    int rank;
    int fan_in;
    tb_net_tensor_kind kind;
} tb_net_tensor_info;

typedef struct tb_net_model tb_net_model;

/* Metadata construction allocates only the tensor directory, not 2.47 GB of
 * parameter memory. Device weights and gradients are allocated by the caller. */
tb_net_model *tb_net_create_618m(void);
void tb_net_destroy(tb_net_model *model);
uint64_t tb_net_parameter_count(const tb_net_model *model);
uint64_t tb_net_actor_parameter_count(const tb_net_model *model);
uint64_t tb_net_critic_parameter_count(const tb_net_model *model);
size_t tb_net_tensor_count(const tb_net_model *model);
const tb_net_tensor_info *tb_net_tensor_at(const tb_net_model *model, size_t index);
const tb_net_tensor_info *tb_net_find_tensor(const tb_net_model *model, const char *name);
int tb_net_validate_layout(const tb_net_model *model);

/* Fill a host-side segment [offset, offset+count) for streaming to a device.
 * The seeded initialization is deterministic and independent of chunk size. */
int tb_net_initialize_slice(const tb_net_model *model, uint64_t offset,
                            float *destination, size_t count, uint64_t seed);

#ifdef __cplusplus
}
#endif
#endif
