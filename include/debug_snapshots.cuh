#pragma once
#include <cstdio>

// 디버그 스냅샷을 위한 헬퍼 함수들
__device__ inline void debug_snapshot(const char* label, int block_id, int thread_id, 
                                     const uint64_t* data, int count, uint64_t mod) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {  // 첫 번째 블록의 처음 4 스레드만
        printf("%s[%d,%d]: ", label, block_id, thread_id);
        for (int i = 0; i < count && i < 8; i++) {
            printf("%llu ", (unsigned long long)data[i]);
        }
        printf("(mod %llu)\n", (unsigned long long)mod);
    }
}

// 트위들 곱 직후 스냅샷 (A)
__device__ inline void snapshot_after_twiddle_mult(const char* stage, int row, int col, 
                                                  uint64_t value, uint64_t twiddle, uint64_t result, uint64_t mod) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        printf("TWIDDLE_%s[%d,%d]: val=%llu * tw=%llu = %llu (mod %llu)\n", 
               stage, row, col, (unsigned long long)value, (unsigned long long)twiddle, 
               (unsigned long long)result, (unsigned long long)mod);
    }
}

// BFU 입력 8-튜플 스냅샷 (B)
__device__ inline void snapshot_bfu_input(const char* stage, int row, int col, 
                                         const uint64_t* bfu_input, uint64_t mod) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        printf("BFU_INPUT_%s[%d,%d]: ", stage, row, col);
        for (int i = 0; i < 8; i++) {
            printf("%llu ", (unsigned long long)bfu_input[i]);
        }
        printf("(mod %llu)\n", (unsigned long long)mod);
    }
}

// BFU 출력 8-튜플 스냅샷 (C)
__device__ inline void snapshot_bfu_output(const char* stage, int row, int col, 
                                          const uint64_t* bfu_output, uint64_t mod) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        printf("BFU_OUTPUT_%s[%d,%d]: ", stage, row, col);
        for (int i = 0; i < 8; i++) {
            printf("%llu ", (unsigned long long)bfu_output[i]);
        }
        printf("(mod %llu)\n", (unsigned long long)mod);
    }
}

// 스토어 직전 스냅샷 (D)
__device__ inline void snapshot_before_store(const char* stage, int row, int col, 
                                            uint64_t value, int global_addr, uint64_t mod) {
    if (blockIdx.x == 0 && threadIdx.x < 4) {
        printf("STORE_%s[%d,%d]: val=%llu -> addr=%d (mod %llu)\n", 
               stage, row, col, (unsigned long long)value, global_addr, (unsigned long long)mod);
    }
}
