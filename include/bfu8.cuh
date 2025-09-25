#pragma once
#include "modutil.cuh"

// BFU8_M은 bfu8_constants.cu에서 정의됨
extern __device__ uint64_t BFU8_M[8];

// Winograd Radix-8 (파이썬 참조와 동일 단계) - 순차 처리
__device__ inline void BFU8_winograd_sequential(uint64_t a[8], uint64_t mod){
    // Step 1
    hadamard2(a[0], a[1], mod);
    hadamard2(a[2], a[3], mod);
    hadamard2(a[4], a[5], mod);
    hadamard2(a[6], a[7], mod);
    // Step 2
    hadamard2(a[4], a[6], mod);
    hadamard2(a[5], a[7], mod);
    // Step 3 (twiddle lanes) - 순차 처리
    a[3] = mod_mul(a[3], BFU8_M[3], mod);
    a[6] = mod_mul(a[6], BFU8_M[5], mod);
    a[5] = mod_mul(a[5], BFU8_M[6], mod);
    a[7] = mod_mul(a[7], BFU8_M[7], mod);
    // Step 4
    hadamard2(a[0], a[2], mod);
    hadamard2(a[1], a[3], mod);
    hadamard2(a[5], a[7], mod);
    // Step 5
    hadamard2(a[0], a[4], mod);
    hadamard2(a[1], a[5], mod);
    hadamard2(a[2], a[6], mod);
    hadamard2(a[3], a[7], mod);
}

