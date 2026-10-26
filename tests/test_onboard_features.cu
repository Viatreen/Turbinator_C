#include "../src/network/onboard_features.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ static void calculate(const float *obs, const float *mission, float *result) {
    int row = (int)blockIdx.x;
    tb_onboard_tactical_prior(obs + (size_t)row * 4148, mission + (size_t)row * 10,
                              result + (size_t)row * 39, result + (size_t)row * 39 + 24);
}

int main(void) {
    enum { ROWS = 4 };
    float *obs = (float *)calloc(ROWS * 4148, sizeof(float));
    float *mission = (float *)calloc(ROWS * 10, sizeof(float));
    float result[ROWS * 39];
    if (!obs || !mission) return 1;
    for (int r = 0; r < ROWS; ++r) {
        float *o = obs + (size_t)r * 4148;
        o[2] = 8.0f; o[6] = 1.0f; o[40] = 0.6f; o[47] = 1.0f;
        for (int i = 0; i < 4; ++i) { o[16 + i] = 0.5f; o[24 + i] = 1.0f; }
    }
    obs[3] = 1.0f;
    float *o = obs + 4148, *m = mission + 10;
    o[0] = 5.0f; o[1] = 0.4f; o[3] = 2.0f; o[4] = -0.2f;
    o[48 + 2 * 128 + 3] = 5.0f;
    o[48 + 2 * 128 + 4] = 5.1f;
    m[4] = 1.0f; m[7] = 0.3f; m[8] = -0.2f; m[9] = 0.1f;
    o = obs + 2 * 4148; m = mission + 20;
    o[0] = 80.0f; o[2] = 60.0f; o[3] = 15.0f; o[5] = -0.3f;
    o[6] = 0.98f; o[7] = 0.10f; o[8] = 0.10f; o[9] = 0.05f;
    m[0] = 0.5f; m[2] = 0.5f; m[4] = 1.0f;
    m[7] = 0.2f; m[8] = 0.05f; m[9] = 0.1f;
    o = obs + 3 * 4148; m = mission + 30;
    o[0] = 200.0f; o[2] = 40.0f; o[3] = 20.0f; o[4] = 1.0f;
    o[6] = 0.9f; o[7] = 0.1f; o[8] = -0.2f; o[9] = 0.3f;
    m[4] = -0.5f; m[5] = 0.8f; m[6] = 1.0f;
    m[7] = -0.3f; m[8] = 0.1f;
    float *d_obs = NULL, *d_mission = NULL, *d_result = NULL;
    int ok = cudaMalloc((void **)&d_obs, ROWS * 4148 * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void **)&d_mission, ROWS * 10 * sizeof(float)) == cudaSuccess &&
             cudaMalloc((void **)&d_result, sizeof(result)) == cudaSuccess &&
             cudaMemcpy(d_obs, obs, ROWS * 4148 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
             cudaMemcpy(d_mission, mission, ROWS * 10 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        calculate<<<ROWS, 1>>>(d_obs, d_mission, d_result);
        ok = cudaDeviceSynchronize() == cudaSuccess &&
             cudaMemcpy(result, d_result, sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (!ok) fprintf(stderr, "CUDA onboard feature calculation failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    if (ok) {
        /* Golden values were calculated by the reviewed tensor formulas on
         * the same four hover, slalom, BWB cruise, and arena fixtures. */
        static const struct { int row, index; float expected; } golden[] = {
            {0,3,0.0180000067f}, {0,16,0.148885041f}, {0,19,0.999999881f},
            {0,20,0.116912015f}, {0,23,-0.707106769f},
            {0,24,-0.287516475f}, {0,26,0.476842999f}, {0,32,-0.382895738f},
            {0,34,0.382895738f}, {0,38,-0.800000012f},
            {1,12,0.458358645f}, {1,13,-0.313454777f}, {1,15,0.353995144f},
            {1,20,0.704012096f}, {1,21,0.718363643f},
            {1,24,0.476842999f}, {1,25,-0.178828716f}, {1,27,0.0291560888f},
            {1,32,0.382895738f}, {1,33,-0.107192054f},
            {2,8,-0.209403276f}, {2,10,0.959999979f}, {2,12,0.319995612f},
            {2,16,0.977306247f}, {2,19,-0.396981865f},
            {2,24,-0.610540390f}, {2,26,-0.610540390f},
            {2,32,0.541728497f}, {2,33,0.00769173447f},
            {3,8,-0.0806194842f}, {3,9,0.416533887f}, {3,10,0.899999976f},
            {3,12,-0.309927940f}, {3,16,0.988346934f},
            {3,24,-0.100000024f}, {3,26,-0.100000024f},
            {3,32,-0.666666687f}, {3,33,-0.345075011f},
            {3,36,1.0f}, {3,37,0.467838049f}
        };
        for (size_t i = 0; i < sizeof(golden) / sizeof(golden[0]); ++i) {
            float actual = result[golden[i].row * 39 + golden[i].index];
            if (!isfinite(actual) || fabsf(actual - golden[i].expected) > 5.0e-5f) {
                fprintf(stderr, "fixture %d index %d: expected %.9g, got %.9g\n",
                        golden[i].row, golden[i].index, golden[i].expected, actual);
                ok = 0;
            }
        }
        for (int r = 0; r < ROWS; ++r) for (int k = 24; k < 39; ++k) {
            float actual = result[r * 39 + k];
            if (!isfinite(actual) || actual < -1.00001f || actual > 1.00001f) ok = 0;
        }
        if (ok) puts("onboard tactical and prior fixtures passed");
    }
    cudaFree(d_result); cudaFree(d_mission); cudaFree(d_obs);
    free(obs); free(mission);
    return ok ? 0 : 1;
}
