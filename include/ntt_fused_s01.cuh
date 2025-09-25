#pragma once
#include "modutil.cuh"
#include "bfu8_simple.cuh"
#include "bfu8.cuh"

#ifndef TILES_PER_BLOCK_S01
#define TILES_PER_BLOCK_S01 1
#endif

__global__ void ntt_radix8_fused_s01_tile64(
    uint64_t* __restrict__ poly,
    const uint64_t* __restrict__ tlist,
    const uint32_t* __restrict__ stage_off,
    uint64_t mod, int N) // N=4096
{
    constexpr int R=8;
    constexpr int TPB = TILES_PER_BLOCK_S01;
    // 가드: blockDim.x = 64 * TPB 체크
    if (blockDim.x != 64 * TPB) return;
    const int cols    = R * TPB;               // 8*TPB
    const int tid     = threadIdx.x;           // 0..(64*TPB-1)
    const int row     = tid / cols;            // 0..7
    const int col     = tid % cols;            // 0..(8*TPB-1)
    const int sub     = col / R;               // 0..(TPB-1)
    const int lane    = col % R;               // 0..7
    const int tile0   = blockIdx.x * TPB;      // 이 블록의 첫 타일(0..63)
    const int tile    = tile0 + sub;
    if (row >= 8 || lane >= 8 || sub >= TPB || tile >= 64) return;

    extern __shared__ uint64_t sm[];
    uint64_t* sm_val = sm;                          // [8*cols] = 64*TPB
    uint64_t* sm_tw0 = sm + (R*cols);               // [8]  (s=0)
    uint64_t* sm_tw1 = sm + (R*cols) + R;           // [8*cols] (s=1)
    uint64_t* sm_bfu = sm + (R*cols) + R + (R*cols); // [8*cols] BFU scratch

    // Load poly: 연속 64*TPB 중 서브타일
    {
        int m   = R*row + lane;                     // 0..63
        int idx = tile * 64 + m;                    // tile-th chunk of 64
        sm_val[row*cols + col] = poly[idx];
    }

    // s=0 twiddle (j=1, j1=0)
    if (tid < R){
        int off0 = stage_off[0];
        sm_tw0[lane] = tlist[off0 + lane];
    }

    // s=1 twiddle (j=8, j1 = lane)  [within 64-chunk, j1 is local column lane]
    {
        int off1 = stage_off[1];
        int j1p  = lane;                               // 0..7
        sm_tw1[row*cols + col] = tlist[off1 + j1p*R + row];
    }
    __syncthreads();

    // s=0: 행 BFU - addr 기반 로드/산포 (twiddle 복원)
    {
        if (lane == 0){
            uint64_t a[R];
            // j=1, k = row + (tile*8) 가 되도록 64-타일 기준, j1=0
            // addr = 8*k + i = 8*(tile*8 + row) + i = tile*64 + 8*row + i
            const int j1p = 0;
            const int off0 = stage_off[0];
            for (int i=0;i<R;i++){
                int gaddr = tile*64 + 8*row + i;
                uint64_t v = poly[gaddr];
                uint64_t tw = tlist[off0 + j1p*R + i];
                a[i] = mod_mul(v, tw, mod);
            }
            BFU8_winograd_sequential(a, mod);
            for (int i=0;i<R;i++){
                int gaddr = tile*64 + 8*row + i;
                poly[gaddr] = a[i];
            }
        }
        __syncthreads();
    }

    // s=1: 열 BFU - addr 기반 로드/산포 (twiddle 복원) + 곱셈 레이어 4스레드 병렬
    {
        if (row == 0){
            uint64_t a[R];
            // j=8, j1=lane, k = tile, addr = 64*k + j1 + 8*i = tile*64 + lane + 8*i
            const int j1p = lane;
            const int off1 = stage_off[1];
            for (int i=0;i<R;i++){
                int gaddr = tile*64 + lane + 8*i;
                uint64_t v = poly[gaddr];
                uint64_t tw = tlist[off1 + j1p*R + i];
                a[i] = mod_mul(v, tw, mod);
                sm_bfu[i*cols + col] = a[i];
            }
            __syncthreads();
#ifdef BFU_MULT_PAR4
            if (row < 4){
                // parallel multiply layer: indices 3,6,5,7 handled by row 0,1,2,3
                const int targets[4] = {3,6,5,7};
                int t = targets[row];
                // already multiplied above; keep as-is (here for symmetry)
                sm_bfu[t*cols + col] = sm_bfu[t*cols + col];
            }
#endif
            __syncthreads();
            for (int i=0;i<R;i++) a[i] = sm_bfu[i*cols + col];
            BFU8_winograd_sequential(a, mod);
            for (int i=0;i<R;i++){
                int gaddr = tile*64 + lane + 8*i;
                poly[gaddr] = a[i];
            }
        }
        __syncthreads();
    }

    // Store
    {
        int m   = R*row + lane;
        int idx = tile * 64 + m;
        poly[idx] = sm_val[row*cols + col];
    }
}
