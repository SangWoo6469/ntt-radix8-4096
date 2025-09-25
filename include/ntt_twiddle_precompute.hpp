#pragma once
#include <vector>
#include <cstdint>

static inline uint64_t modexp64(uint64_t a, uint64_t e, uint64_t mod){
    __uint128_t r=1,b=a%mod;
    while(e){ if(e&1) r=(r*b)%mod; b=(b*b)%mod; e>>=1; }
    return (uint64_t)r;
}
static inline uint32_t bit_reverse_k(uint32_t x, uint32_t bits){
    uint32_t r=0; for(uint32_t i=0;i<bits;i++){ r=(r<<1)|(x&1u); x>>=1; } return r;
}
static inline uint32_t ilogR(uint32_t N, uint32_t R){
    uint32_t s=0,x=1; while(x<N){ x*=R; s++; } return s;
}

// Radix-8: stage s에서 j = 8^s, 각 j1(0..j-1)에 대해 lane i(0..7)를 로컬 비트리버스 순서로 저장
inline void precompute_all_tlists_radix8(
    uint32_t N, uint64_t gamma, uint64_t mod,
    std::vector<uint64_t>& out_tlist, std::vector<uint32_t>& out_stage_offsets)
{
    const uint32_t R=8, bits=3, stages=ilogR(N,R);
    out_tlist.clear(); out_stage_offsets.clear();
    out_stage_offsets.push_back(0);

    for(uint32_t s=0;s<stages;s++){
        uint32_t j=1; for(uint32_t t=0;t<s;t++) j*=R;
        uint32_t index = N / (R * j);
        uint64_t gpw = modexp64(gamma, index, mod);

        for(uint32_t j1=0;j1<j;j1++){
            uint64_t tmp[R];
            for(uint32_t i=0;i<R;i++){
                uint64_t pw = (uint64_t)(i * (2*j1 + 1));
                tmp[i] = modexp64(gpw, pw, mod);
            }
            for(uint32_t i=0;i<R;i++)
                out_tlist.push_back(tmp[bit_reverse_k(i,bits)]);
        }
        out_stage_offsets.push_back((uint32_t)out_tlist.size());
    }
}
