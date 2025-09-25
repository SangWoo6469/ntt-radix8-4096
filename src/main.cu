#include <bits/stdc++.h>
#include "../include/modutil.cuh"
#include "../include/ntt_twiddle_precompute.hpp"
#include "../include/ntt_roots.hpp"
#include "bfu8_host.hpp"
#include <cuda_runtime.h>

extern "C" void ntt4096_run_s01_s23(
    std::vector<uint64_t>&,
    const std::vector<uint64_t>&,
    const std::vector<uint32_t>&,
    uint64_t);

int main(){
    constexpr uint64_t N = 4096;
    constexpr uint64_t R = 8;     // Radix-8
    const uint64_t q = 0xFFFFFFFF00000001ull;

    // (0) 모듈러 파라미터 업로드(Barrett/Mont)
    // 호스트 초기화 함수는 modutil.cuh 내부에 CUDA 포함 시 제공됨
    upload_mod_params(q);

    // (1) g, 그리고 γ(2N), ω_N(N), BFU_ω(R) 계산
    uint64_t g          = find_primitive_root(q);
    uint64_t gamma_2N   = compute_gamma_2N(q, N, g);    // twiddle용 (2N-th)
    uint64_t omega_N    = compute_omega_N(q, N, g);     // 필요시 사용(N-th)
    uint64_t bfu_omegaR = compute_bfu_omega_R(q, R, g); // BFU m용 (R-th)

    // (선택) 검증
    // γ(2N-원시근)·BFU_ω(R-원시근) 재확인
    auto p=[](uint64_t a,uint64_t e,uint64_t q){__uint128_t r=1,b=a%q;while(e){if(e&1)r=(r*b)%q;b=(b*b)%q;e>>=1;}return (uint64_t)r;};
    assert(p(gamma_2N, 2*N, q)==1 && p(gamma_2N, N, q)!=1);
    assert(p(gamma_2N, 64, q)!=1); // s=1 index=64
    assert(p(bfu_omegaR, R, q)==1 && p(bfu_omegaR, 1, q)!=1);

    // (2) BFU8 M 계산 (+ 선택: 상수만 Montgomery 1배 인코딩)
    auto m8 = compute_bfu8_m(q, bfu_omegaR);
#if defined(USE_MONTGOMERY) && defined(MONT_CONST_ENCODE)
    constexpr bool kMontConstEncode = true;
#else
    constexpr bool kMontConstEncode = false;
#endif
    upload_bfu8_m_with_mode(m8, q, kMontConstEncode);
    
    // BFU8_M 초기화 확인
    std::cout << "BFU8_M values: ";
    for (int i = 0; i < 8; i++) {
        std::cout << m8[i] << " ";
    }
    std::cout << std::endl;

    // (3) t_list 생성: ★ gamma는 반드시 2N-원시근을 사용 ★
    std::vector<uint64_t> tlist;
    std::vector<uint32_t> stage_off;
    precompute_all_tlists_radix8((uint32_t)N, gamma_2N, q, tlist, stage_off);

#if defined(USE_MONTGOMERY) && defined(MONT_CONST_ENCODE)
    // 상수만 R배 인코딩(C*R mod q)
    mont_encode_constants_inplace(tlist, q);
#endif

    // (3.5) 디버그 출력: stage_off와 tlist 일부 확인 (환경변수로 제어)
    if (std::getenv("NTT_DEBUG_TW")){
        std::cout << "stage_off: ";
        for (size_t i=0;i<stage_off.size();++i) std::cout << stage_off[i] << (i+1<stage_off.size()? ',':'\n');
        auto print_slice = [&](const char* tag, size_t off, size_t cnt){
            std::cout << tag << ": ";
            for (size_t i=0;i<cnt && off+i<tlist.size();++i) std::cout << tlist[off+i] << (i+1<cnt? ' ': '\n');
        };
        if (stage_off.size()>=1) print_slice("tlist[s0][0:16]", stage_off[0], 16);
        if (stage_off.size()>=2) print_slice("tlist[s1][0:16]", stage_off[1], 16);
        if (stage_off.size()>=3) print_slice("tlist[s2][0:16]", stage_off[2], 16);
        if (stage_off.size()>=4) print_slice("tlist[s3][0:16]", stage_off[3], 16);
    }

    // (4) 입력 poly
    std::vector<uint64_t> poly(N);
    std::mt19937_64 rng(42);
    for (auto &x: poly) x = rng() % q;

    // (5) 벤치마크: CUDA 이벤트 기반
    int iters = 200;
    if (const char* env = std::getenv("NTT_BENCH_ITERS")) {
        int v = std::atoi(env); if (v > 0) iters = v;
    }

    // 워밍업
    for (int i=0;i<3;i++) ntt4096_run_s01_s23(poly, tlist, stage_off, q);

    cudaEvent_t evS, evE; cudaEventCreate(&evS); cudaEventCreate(&evE);
    float sum_ms=0.0f, min_ms=1e30f, max_ms=0.0f;
    size_t free0=0,total0=0; cudaMemGetInfo(&free0,&total0);

    for (int it=0; it<iters; ++it){
        // 입력을 반복마다 갱신(간단)
        for (auto &x: poly) x = (uint64_t)rand() % q;
        cudaEventRecord(evS);
        ntt4096_run_s01_s23(poly, tlist, stage_off, q);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return 1;
        }
        cudaEventRecord(evE); cudaEventSynchronize(evE);
        float ms=0.0f; cudaEventElapsedTime(&ms, evS, evE);
        sum_ms += ms; if (ms<min_ms) min_ms=ms; if (ms>max_ms) max_ms=ms;
    }

    size_t free1=0,total1=0; cudaMemGetInfo(&free1,&total1);
    double used_mb = (double)(total1 - free1) / 1048576.0;

    double avg_ms = sum_ms / std::max(1, iters);

    // (6) 요약 출력(JSON 유사)
    std::cout << "{\"N\":" << N
              << ",\"iters\":" << iters
              << ",\"avg_ms\":" << avg_ms
              << ",\"min_ms\":" << min_ms
              << ",\"max_ms\":" << max_ms
              << ",\"gpu_mem_used_MB\":" << used_mb
              << "}\n";

    // 샘플 값 출력
    for(int i=0;i<16;i++) std::cout << poly[i] << (i+1<16? ' ':'\n');
    return 0;
}
