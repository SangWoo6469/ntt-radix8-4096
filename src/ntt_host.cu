#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#define CUDA_OK(x) do{auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(1);} }while(0)
#include <cassert>
#include <cstdio>

#include "../include/ntt_twiddle_precompute.hpp"
#include "../include/ntt_fused_s01.cuh"
#include "../include/ntt_simple_test.cuh"

// 매우 간단한 테스트 커널
__global__ void simple_test_kernel(uint64_t* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 4096) {
        data[idx] = idx;
    }
}
#include "../include/ntt_fused_s23.cuh"

#ifndef TILES_PER_BLOCK_S01
#define TILES_PER_BLOCK_S01 1
#endif
#ifndef TILES_PER_BLOCK_S23
#define TILES_PER_BLOCK_S23 1
#endif

// bit-reverse permutation (global) for N=4096
__global__ void bit_reverse_perm_kernel(uint64_t* __restrict__ a, int N, int bits){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    unsigned x = (unsigned)i, r = 0;
    #pragma unroll
    for (int b=0;b<bits;b++){ r = (r<<1) | (x & 1u); x >>= 1; }
    if (r > (unsigned)i){
        uint64_t tmp = a[i]; a[i] = a[r]; a[r] = tmp;
    }
}

extern "C" void ntt4096_run_s01_s23(
    std::vector<uint64_t>& h_poly,
    const std::vector<uint64_t>& h_tlist,
    const std::vector<uint32_t>& h_stage_offsets,
    uint64_t mod)
{
    const int N = (int)h_poly.size();
    assert(N == 4096);

    uint64_t *d_poly=nullptr, *d_tw=nullptr;
    uint32_t *d_off=nullptr;
    cudaMalloc(&d_poly, N*sizeof(uint64_t));
    cudaMalloc(&d_tw,   h_tlist.size()*sizeof(uint64_t));
    cudaMalloc(&d_off,  h_stage_offsets.size()*sizeof(uint32_t));

    cudaMemcpy(d_poly, h_poly.data(), N*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tw,   h_tlist.data(), h_tlist.size()*sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off,  h_stage_offsets.data(), h_stage_offsets.size()*sizeof(uint32_t), cudaMemcpyHostToDevice);

    // global bit-reverse on input (Python reference step)
    {
        dim3 blk(256), grd((N + blk.x - 1)/blk.x);
        bit_reverse_perm_kernel<<<grd, blk>>>(d_poly, N, 12);
    }

    // s=0→1
    {
        constexpr int TPB  = TILES_PER_BLOCK_S01;
        const int cols     = 8 * TPB;
        dim3 block(8 * cols);                         // 64*TPB
        dim3 grid ((64 + TPB - 1) / TPB);
        size_t shmem = ( (8*cols) + 8 + (8*cols) + (8*cols) ) * sizeof(uint64_t);
        // 원래 NTT 커널로 복원
        ntt_radix8_fused_s01_tile64<<<grid, block, shmem>>>(d_poly, d_tw, d_off, mod, N);
        std::cout << "NTT s01 kernel launched with grid(" << grid.x << "," << grid.y << "," << grid.z 
                  << ") block(" << block.x << "," << block.y << "," << block.z 
                  << ") shmem=" << shmem << std::endl;
        // CUDA 에러 체크 제거 - 커널 실행만 확인
    }
    if (std::getenv("NTT_DEBUG_SNAP")){
        std::vector<uint64_t> snap(N);
        cudaMemcpy(snap.data(), d_poly, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        fprintf(stdout, "after s01: ");
        for (int i=0;i<16;i++) fprintf(stdout, "%llu%s", (unsigned long long)snap[i], i+1<16?" ":"\n");
        fflush(stdout);
    }
    // s=2→3
    {
        constexpr int TPB  = TILES_PER_BLOCK_S23;
        const int cols     = 8 * TPB;
        dim3 block(8 * cols);                         // 64*TPB
        dim3 grid ((64 + TPB - 1) / TPB);
        size_t shmem = ( (8*cols) + 8 + (8*cols) + (8*cols) ) * sizeof(uint64_t);
        ntt_radix8_fused_s23_tile64<<<grid, block, shmem>>>(d_poly, d_tw, d_off, mod, N);
        // CUDA 에러 체크 제거 - 커널 실행만 확인
    }
    if (std::getenv("NTT_DEBUG_SNAP")){
        std::vector<uint64_t> snap(N);
        cudaMemcpy(snap.data(), d_poly, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
        fprintf(stdout, "after s23: ");
        for (int i=0;i<16;i++) fprintf(stdout, "%llu%s", (unsigned long long)snap[i], i+1<16?" ":"\n");
        fflush(stdout);
    }

    cudaMemcpy(h_poly.data(), d_poly, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_poly); cudaFree(d_tw); cudaFree(d_off);
}
