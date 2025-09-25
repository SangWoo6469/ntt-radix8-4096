#pragma once
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../include/modutil.cuh"

// BFU8_M은 bfu8_constants.cu에서 정의됨
extern __device__ uint64_t BFU8_M[8];

static inline uint64_t modexp64_host(uint64_t a, uint64_t e, uint64_t mod){
    __uint128_t r=1,b=a%mod; while(e){ if(e&1) r=(r*b)%mod; b=(b*b)%mod; e>>=1; } return (uint64_t)r;
}
static inline uint64_t modinv64_host(uint64_t a, uint64_t mod){
    return modexp64_host(a%mod, mod-2, mod);
}

// BFU8 M 계산 (필요 인덱스만 유효)
inline std::vector<uint64_t> compute_bfu8_m(uint64_t mod, uint64_t omega){
    std::vector<uint64_t> m(8, 1);
    uint64_t w1 = modexp64_host(omega, 1, mod);
    uint64_t w2 = modexp64_host(omega, 2, mod);
    uint64_t w3 = modexp64_host(omega, 3, mod);
    uint64_t inv2 = modinv64_host(2, mod);
    // m[0], m[1], m[2], m[4] = 1
    m[0] = 1; m[1] = 1; m[2] = 1; m[4] = 1;
    // m[3] = m[5] = w^2
    m[3] = w2;  m[5] = w2;
    // m[6] = (w^1 + w^3)/2, m[7] = (w^1 - w^3)/2
    m[6] = ( (w1 + w3) % mod ) * (__uint128_t)inv2 % mod;
    m[7] = ( (w1 + mod - w3) % mod ) * (__uint128_t)inv2 % mod;
    return m;
}

// 디바이스 업로드(그대로)
inline void upload_bfu8_m(const std::vector<uint64_t>& m){
    cudaMemcpy(BFU8_M, m.data(), sizeof(uint64_t)*8, cudaMemcpyHostToDevice);
}

// (옵션) Montgomery 1배 인코딩 후 업로드
inline void upload_bfu8_m_with_mode(std::vector<uint64_t> m, uint64_t mod, bool montgomery_const_encode){
#if defined(USE_MONTGOMERY)
    if (montgomery_const_encode){
        mont_encode_constants_inplace(m, mod); // m[i] = m[i]*R mod q
    }
#endif
    upload_bfu8_m(m);
}
