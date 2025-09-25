#pragma once
#include <stdint.h>
#include <vector>
#include <cuda_runtime.h>

//
// ===========================================================
//  Device global params
// ===========================================================
struct BarrettParams64 {
    uint64_t mod;     // q
    uint64_t mu_lo;   // floor(2^128 / q) low64
    uint64_t mu_hi;   // floor(2^128 / q) high64
};
struct MontgomeryParams64 {
    uint64_t mod;     // q (odd)
    uint64_t nprime;  // -q^{-1} mod 2^64
    uint64_t r2;      // R^2 mod q, R=2^64
};

#ifdef __CUDACC__
__device__ __constant__ BarrettParams64     __g_barrett64;
__device__ __constant__ MontgomeryParams64  __g_mont64;
#endif

//
// ===========================================================
//  Small helpers (device)
// ===========================================================
#ifdef __CUDACC__
__device__ __forceinline__ uint64_t mod_add(uint64_t a, uint64_t b, uint64_t mod){
    uint64_t c = a + b; return (c >= mod) ? (c - mod) : c;
}
__device__ __forceinline__ uint64_t mod_sub(uint64_t a, uint64_t b, uint64_t mod){
    return (a >= b) ? (a - b) : (a + mod - b);
}

// 2-point Hadamard (x,y) -> (x+y, x-y) mod q
__device__ __forceinline__ void hadamard2(uint64_t &x, uint64_t &y, uint64_t mod){
    uint64_t s = mod_add(x,y,mod);
    uint64_t d = mod_sub(x,y,mod);
    x = s; y = d;
}

__device__ __forceinline__ uint64_t mod_add_dev(uint64_t a, uint64_t b){
#if defined(USE_MONTGOMERY)
    const uint64_t mod = __g_mont64.mod;
#else
    const uint64_t mod = __g_barrett64.mod;
#endif
    uint64_t c = a + b; return (c >= mod) ? (c - mod) : c;
}
__device__ __forceinline__ uint64_t mod_sub_dev(uint64_t a, uint64_t b){
#if defined(USE_MONTGOMERY)
    const uint64_t mod = __g_mont64.mod;
#else
    const uint64_t mod = __g_barrett64.mod;
#endif
    return (a >= b) ? (a - b) : (a + mod - b);
}

//
// ===========================================================
//  mod_mul selections
// ===========================================================

// ---------- (A) 기본: 128비트 % q ----------
#if !defined(USE_MONTGOMERY) && !defined(USE_BARRETT_UMULHI)
__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t /*unused*/){
    const uint64_t mod = __g_barrett64.mod;
    unsigned __int128 x = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(x % mod);
}

#else

// ---------- (B) Montgomery (권장, % 없음) ----------
#if defined(USE_MONTGOMERY)
// returns a*b*R^{-1} mod q
__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t /*unused*/){
    const uint64_t mod    = __g_mont64.mod;
    const uint64_t nprime = __g_mont64.nprime;
    unsigned __int128 u = (unsigned __int128)a * (unsigned __int128)b;
    uint64_t lo = (uint64_t)u;
    uint64_t m  = (uint64_t)((unsigned __int128)lo * (unsigned __int128)nprime);
    unsigned __int128 t = u + (unsigned __int128)m * (unsigned __int128)mod;
    uint64_t res = (uint64_t)(t >> 64);
    if (res >= mod) res -= mod;
    return res;
}

// 도메인 헬퍼(필요시)
__device__ __forceinline__ uint64_t to_mont(uint64_t x){
    return mod_mul(x, __g_mont64.r2, 0); // x*R
}
__device__ __forceinline__ uint64_t from_mont(uint64_t x){
    return mod_mul(x, 1ull, 0);          // x/R
}

#else

// ---------- (C) Barrett (umul64hi 기반, % 없음) ----------
__device__ __forceinline__ uint64_t barrett_reduce128(uint64_t hi, uint64_t lo){
    const uint64_t mod   = __g_barrett64.mod;
    const uint64_t mu_lo = __g_barrett64.mu_lo;
    const uint64_t mu_hi = __g_barrett64.mu_hi;

    uint64_t hi_mul_mu_lo_hi = __umul64hi(hi, mu_lo);
    unsigned __int128 q128   = (unsigned __int128)hi * (unsigned __int128)mu_hi
                             + (unsigned __int128)hi_mul_mu_lo_hi;

    unsigned __int128 x  = ((unsigned __int128)hi << 64) | (unsigned __int128)lo;
    unsigned __int128 qm = q128 * (unsigned __int128)mod;
    unsigned __int128 r  = x - qm;

    uint64_t res = (uint64_t)r;
    if (res >= mod) res -= mod;
    if (res >= mod) res -= mod;
    return res;
}
__device__ __forceinline__ uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t /*unused*/){
    uint64_t lo = a * b;
    uint64_t hi = __umul64hi(a, b);
    return barrett_reduce128(hi, lo);
}
#endif
#endif
#endif // __CUDACC__

//
// ===========================================================
//  Host helpers: upload params, 상수 인코딩
// ===========================================================
// Barrett μ
static inline void compute_barrett_mu(uint64_t q, uint64_t &mu_hi, uint64_t &mu_lo){
    unsigned __int128 one = (unsigned __int128)1;
    unsigned __int128 x   = (one << 127);
    unsigned __int128 mu  = (x / q) << 1;
    unsigned __int128 rem = (x % q) << 1;
    if (rem >= q) { mu += 1; rem -= q; }
    mu_hi = (uint64_t)(mu >> 64);
    mu_lo = (uint64_t)mu;
}
static inline void upload_barrett_params(uint64_t q){
    BarrettParams64 h{};
    h.mod = q;
    compute_barrett_mu(q, h.mu_hi, h.mu_lo);
    cudaMemcpyToSymbol(__g_barrett64, &h, sizeof(BarrettParams64));
}

// Montgomery n', R^2
static inline uint64_t compute_nprime(uint64_t q){
    uint64_t inv = 1;
    inv *= 2 - q * inv;
    inv *= 2 - q * inv;
    inv *= 2 - q * inv;
    inv *= 2 - q * inv;
    inv *= 2 - q * inv;
    inv *= 2 - q * inv;
    return ~inv + 1;
}
static inline uint64_t compute_r2(uint64_t q){
    unsigned __int128 one = (unsigned __int128)1;
    unsigned __int128 r2  = ((one << 127) % q);
    r2 = (r2 << 1) % q;  // 2^128 % q
    return (uint64_t)r2;
}
static inline void upload_montgomery_params(uint64_t q){
    MontgomeryParams64 h{};
    h.mod    = q;
    h.nprime = compute_nprime(q);
    h.r2     = compute_r2(q);
    cudaMemcpyToSymbol(__g_mont64, &h, sizeof(MontgomeryParams64));
}

// 둘 다
static inline void upload_mod_params(uint64_t q){
    upload_barrett_params(q);
    upload_montgomery_params(q);
}

// ===== 상수만 Montgomery 1배 인코딩(C * R mod q) =====
static inline uint64_t compute_r1(uint64_t q){
    unsigned __int128 one = (unsigned __int128)1;
    unsigned __int128 r1  = (one << 64) % q; // R mod q
    return (uint64_t)r1;
}
static inline void mont_encode_constants_inplace(std::vector<uint64_t>& vec, uint64_t q){
    const uint64_t R1 = compute_r1(q);
    for (auto &x : vec){
        unsigned __int128 t = (unsigned __int128)x * (unsigned __int128)R1;
        x = (uint64_t)(t % q); // C*R mod q
    }
}
static inline uint64_t mont_encode_const(uint64_t c, uint64_t q){
    const uint64_t R1 = compute_r1(q);
    unsigned __int128 t = (unsigned __int128)c * (unsigned __int128)R1;
    return (uint64_t)(t % q);
}

