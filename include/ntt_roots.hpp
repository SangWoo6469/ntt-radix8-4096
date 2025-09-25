#pragma once
#include <vector>
#include <cstdint>

static inline uint64_t powmod_u128(uint64_t a, uint64_t e, uint64_t mod){
    __uint128_t r=1, b=a%mod;
    while(e){ if(e&1) r=(r*b)%mod; b=(b*b)%mod; e>>=1; }
    return (uint64_t)r;
}

// (q-1) 소인수 분해(간단 trial division)
static inline void factorize_qm1(uint64_t qm1, std::vector<uint64_t>& primes){
    if((qm1&1ull)==0){ primes.push_back(2); while((qm1&1ull)==0) qm1>>=1; }
    for(uint64_t p=3;p*p<=qm1;p+=2){
        if(qm1%p==0){ primes.push_back(p); while(qm1%p==0) qm1/=p; }
    }
    if(qm1>1) primes.push_back(qm1);
}

// 원시근 g: g^((q-1)/p) != 1 for 모든 p | (q-1)
static inline uint64_t find_primitive_root(uint64_t q){
    uint64_t qm1=q-1; std::vector<uint64_t> pf; pf.reserve(16);
    factorize_qm1(qm1, pf);
    for(uint64_t g=2;;++g){
        bool ok=true;
        for(uint64_t p:pf) if (powmod_u128(g, qm1/p, q)==1ull){ ok=false; break; }
        if(ok) return g;
    }
}

// 정확한 차수(order)를 갖는 원시근 생성: root = g^((q-1)/order)
static inline uint64_t make_exact_root(uint64_t q, uint64_t order, uint64_t g){
    // 전제: order | (q-1)
    return powmod_u128(g, (q-1)/order, q);
}

// 검증(선택)
static inline bool is_exact_order(uint64_t root, uint64_t order, uint64_t q){
    if (powmod_u128(root, order, q) != 1ull) return false;
    uint64_t k=order;
    while((k&1ull)==0){ k>>=1; if (powmod_u128(root, k, q)==1ull) return false; }
    return true;
}

// 편의 래퍼
static inline uint64_t compute_gamma_2N(uint64_t q, uint64_t N, uint64_t g){
    return make_exact_root(q, 2*N, g);   // γ: 2N-th root
}
static inline uint64_t compute_omega_N(uint64_t q, uint64_t N, uint64_t g){
    return make_exact_root(q, N, g);     // ω_N: N-th root
}
static inline uint64_t compute_bfu_omega_R(uint64_t q, uint64_t R, uint64_t g){
    return make_exact_root(q, R, g);     // BFU_ω: R-th root
}
