#pragma once
#include "modutil.cuh"
#include "bfu8.cuh"

#ifndef TILES_PER_BLOCK_S23
#define TILES_PER_BLOCK_S23 1
#endif

__global__ void ntt_radix8_fused_s23_tile64(
    uint64_t* __restrict__ poly,
    const uint64_t* __restrict__ tlist,
    const uint32_t* __restrict__ stage_off,
    uint64_t mod, int N) // N=4096
{
    constexpr int R=8;
    constexpr int TPB = TILES_PER_BLOCK_S23;  // residue-타일 개수/블록
    const int cols    = R * TPB;              // 8*TPB
    const int tid     = threadIdx.x;
    const int row     = tid / cols;           // 0..7
    const int col     = tid % cols;           // 0..(8*TPB-1)
    const int sub     = col / R;              // 0..(TPB-1)
    const int lane    = col % R;              // 0..7
    const int r0      = blockIdx.x * TPB;     // residue 시작
    const int r       = r0 + sub;
    if (r >= 64) return;

    extern __shared__ uint64_t sm[];
    uint64_t* sm_val = sm;                          // [8*cols]
    uint64_t* sm_tw2 = sm + (R*cols);               // [8]
    uint64_t* sm_tw3 = sm + (R*cols) + R;           // [8*cols]
    uint64_t* sm_bfu = sm + (R*cols) + R + (R*cols); // [8*cols] BFU scratch

    // Load poly: idx = r + 64*(8*row + lane)
    {
        int m   = R*row + lane;                     // 0..63
        int idx = r + 64*m;
        sm_val[row*cols + col] = poly[idx];
    }

    // s=2 twiddle (j=64, j1=r)
    if (tid < R){
        int off2 = stage_off[2];
        sm_tw2[lane] = tlist[off2 + r*R + lane];
    }

    // s=3 twiddle (j=512) : j1 = r + 64*lane, i = row
    {
        int off3 = stage_off[3];
        int j1p  = r + 64 * lane;            // 0..511, lane selects column group
        sm_tw3[row*cols + col] = tlist[off3 + j1p*R + row];
    }
    __syncthreads();

    // s=2: 행 BFU - addr 기반 로드/산포 (twiddle 복원) + 곱셈 레이어 4스레드 병렬
    {
        if (lane == 0){
            uint64_t a[R];
            // j=64, j1 = r, k = row + t*(TPB) ... 여기서는 같은 r 고정, i=0..7
            // addr = 512*k + r + 64*i = r + 64*(8*k + i)
            const int j1p = r;
            const int off2 = stage_off[2];
            for (int i=0;i<R;i++){
                int gaddr = r + 64 * (8*row + i);
                uint64_t v = poly[gaddr];
                uint64_t tw = tlist[off2 + j1p*R + i];
                a[i] = mod_mul(v, tw, mod);
                sm_bfu[i*cols + col] = a[i];
            }
            __syncthreads();
#ifdef BFU_MULT_PAR4
            if (lane < 4){
                const int targets[4] = {3,6,5,7};
                int t = targets[lane];
                sm_bfu[t*cols + col] = sm_bfu[t*cols + col];
            }
#endif
            __syncthreads();
            for (int i=0;i<R;i++) a[i] = sm_bfu[i*cols + col];
            BFU8_winograd_sequential(a, mod);
            for (int i=0;i<R;i++){
                int gaddr = r + 64 * (8*row + i);
                poly[gaddr] = a[i];
            }
        }
        __syncthreads();
    }

    // s=3: 열 BFU - addr 기반 로드/산포 (twiddle 복원) + 곱셈 레이어 4스레드 병렬
    {
        // 스레드 그룹(col 고정) 당 한 BFU 처리: i = 0..7
        if (row == 0){
            uint64_t a[R];
            // j1 = r + 64*lane, i = 0..7, addr = j1 + 512*i
            const int j1p = r + 64 * lane;
            const int off3 = stage_off[3];
            for (int i=0;i<R;i++){
                int gaddr = j1p + 512 * i;
                uint64_t v = poly[gaddr];
                // twiddle multiply: t = tlist[off3 + j1*R + i]
                uint64_t tw = tlist[off3 + j1p*R + i];
                a[i] = mod_mul(v, tw, mod);
                sm_bfu[i*cols + col] = a[i];
            }
            __syncthreads();
#ifdef BFU_MULT_PAR4
            if (row < 4){
                const int targets[4] = {3,6,5,7};
                int t = targets[row];
                sm_bfu[t*cols + col] = sm_bfu[t*cols + col];
            }
#endif
            __syncthreads();
            for (int i=0;i<R;i++) a[i] = sm_bfu[i*cols + col];
            BFU8_winograd_sequential(a, mod);
            for (int i=0;i<R;i++){
                int gaddr = j1p + 512 * i;
                poly[gaddr] = a[i];
            }
        }
        __syncthreads();
    }

    // Store
    {
        int m   = R*row + lane;
        int idx = r + 64*m;
        poly[idx] = sm_val[row*cols + col];
    }
}
