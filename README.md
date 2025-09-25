# NTT Radix-8 4096 Implementation

CUDA 기반의 Radix-8 Number Theoretic Transform (NTT) 구현입니다. 4096 포인트 NTT를 효율적으로 계산하며, 다양한 병렬화 전략과 모듈러 연산 방법을 지원합니다.

## 주요 특징

- **Radix-8 NTT**: 4096 포인트 NTT를 Radix-8 버터플라이 유닛으로 구현
- **BFU8 (Butterfly Unit 8)**: Winograd 알고리즘을 사용한 효율적인 8-포인트 버터플라이
- **병렬화 지원**: 
  - 순차 처리 (Sequential)
  - 곱셈 레이어만 병렬 처리 (BFU_MULT_PAR4)
  - 전체 레이어 병렬 처리 (BFU_FULL_PARALLEL)
- **모듈러 연산**: Montgomery와 Barrett reduction 방법 지원
- **CUDA 최적화**: 다양한 TPB (Tiles Per Block) 설정 지원

## 프로젝트 구조

```
ntt_radix8_4096/
├── include/           # 헤더 파일들
│   ├── bfu8.cuh      # BFU8 구현
│   ├── modutil.cuh   # 모듈러 연산 유틸리티
│   ├── ntt_fused_s01.cuh  # NTT Stage 0-1 커널
│   ├── ntt_fused_s23.cuh  # NTT Stage 2-3 커널
│   ├── ntt_roots.hpp      # NTT 루트 계산
│   └── ntt_twiddle_precompute.hpp  # Twiddle factor 전처리
├── src/               # 소스 파일들
│   ├── bfu8_host.hpp  # BFU8 호스트 함수들
│   ├── main.cu        # 메인 실행 파일
│   └── ntt_host.cu    # NTT 호스트 함수들
├── bench*.sh          # 벤치마크 스크립트들
├── bench_results*.csv # 벤치마크 결과 파일들
└── CMakeLists.txt     # CMake 빌드 설정
```

## 빌드 방법

### 요구사항
- CUDA 12.0 이상
- CMake 3.10 이상
- NVIDIA GPU (Compute Capability 8.6 이상 권장)

### 빌드 명령어

```bash
# 프로젝트 디렉토리로 이동
cd ntt_radix8_4096

# 빌드 디렉토리 생성 및 CMake 설정
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

# 빌드 실행
cmake --build build -j
```

## 사용 방법

### 기본 실행

```bash
./build/ntt_demo
```

### 벤치마크 실행

다양한 벤치마크 스크립트를 사용할 수 있습니다:

```bash
# 기본 벤치마크 (순차 처리)
./bench.sh

# BFU8 순차 처리 벤치마크
./bench_bfu8_sequential.sh

# BFU8 곱셈 레이어만 병렬 처리 벤치마크
./bench_bfu8_mult_parallel.sh

# BFU8 전체 병렬 처리 벤치마크
./bench_bfu8_full_parallel.sh
```

## 벤치마크 결과

프로젝트에는 다음과 같은 벤치마크 결과가 포함되어 있습니다:

- `bench_results.csv`: 기본 NTT 벤치마크 결과
- `bench_results_bfu8_sequential.csv`: BFU8 순차 처리 결과
- `bench_results_bfu8_mult_parallel.csv`: BFU8 곱셈 레이어 병렬 처리 결과
- `bench_results_bfu8_full_parallel.csv`: BFU8 전체 병렬 처리 결과

각 결과 파일은 다음 형식으로 구성됩니다:
```
mode,tpb,avg_ms,min_ms,max_ms,gpu_mem_MB
```

## 병렬화 전략

### 1. 순차 처리 (Sequential)
- BFU8의 모든 연산을 순차적으로 처리
- 가장 안정적이고 호환성이 좋음

### 2. 곱셈 레이어 병렬 처리 (BFU_MULT_PAR4)
- BFU8 내의 곱셈 연산만 병렬 처리
- `BFU_MULT_PAR4=1` 매크로로 활성화

### 3. 전체 병렬 처리 (BFU_FULL_PARALLEL)
- BFU8의 모든 레이어를 병렬 처리
- `BFU_FULL_PARALLEL=1` 매크로로 활성화

## 성능 특징

- **메모리 효율성**: 공유 메모리를 활용한 최적화된 메모리 접근
- **병렬 처리**: CUDA 스레드 블록 내에서의 효율적인 병렬 처리
- **모듈러 연산**: Montgomery와 Barrett reduction의 성능 비교 가능
- **확장성**: 다양한 TPB 설정으로 성능 튜닝 가능

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여하기

버그 리포트나 기능 개선 제안은 GitHub Issues를 통해 해주세요.

## 참고 문헌

- Winograd, S. "On computing the discrete Fourier transform"
- Montgomery, P. L. "Modular multiplication without trial division"
- Barrett, P. "Implementing the Rivest Shamir and Adleman public key encryption algorithm on a standard digital signal processor"
