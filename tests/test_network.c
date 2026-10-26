#include "turbinator/network/network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    tb_net_model *m = tb_net_create_618m();
    if (!m || !tb_net_validate_layout(m)) return 1;
    if (tb_net_parameter_count(m) != TB_NET_TOTAL_PARAMETERS ||
        tb_net_actor_parameter_count(m) != TB_NET_ACTOR_PARAMETERS ||
        tb_net_critic_parameter_count(m) != TB_NET_CRITIC_PARAMETERS) return 2;
    const tb_net_tensor_info *last = tb_net_tensor_at(m, tb_net_tensor_count(m) - 1);
    if (!last || last->offset + last->count != TB_NET_TOTAL_PARAMETERS) return 3;
    const tb_net_tensor_info *t = tb_net_find_tensor(m, "blocks.11.qkv.weight");
    if (!t || t->shape[0] != 4608 || t->shape[1] != 1536) return 4;
    t = tb_net_find_tensor(m, "aero_energy_moe.1.experts_gate_up.5.weight");
    if (!t || t->count != UINT64_C(1536) * 4720) return 5;
    t = tb_net_find_tensor(m, "critic.q2_net.18.weight");
    if (!t || t->count != UINT64_C(2688) * 64) return 6;
    float x[128], y[128];
    if (!tb_net_initialize_slice(m, t->offset, x, 128, 11) ||
        !tb_net_initialize_slice(m, t->offset, y, 128, 11) ||
        memcmp(x, y, sizeof(x))) return 7;
    for (int i = 0; i < 128; ++i) if (!isfinite(x[i])) return 8;
    t = tb_net_find_tensor(m, "trunk_norm.weight");
    if (!t || !tb_net_initialize_slice(m, t->offset, x, 128, 11)) return 9;
    for (int i = 0; i < 128; ++i) if (x[i] != 1.0f) return 10;
    printf("network tensors=%zu parameters=%llu actor=%llu critics=%llu\n",
           tb_net_tensor_count(m), (unsigned long long)tb_net_parameter_count(m),
           (unsigned long long)tb_net_actor_parameter_count(m),
           (unsigned long long)tb_net_critic_parameter_count(m));
    tb_net_destroy(m);
    return 0;
}
