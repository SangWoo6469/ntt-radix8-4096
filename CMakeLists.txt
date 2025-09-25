cmake_minimum_required(VERSION 3.20)
project(ntt_radix8_4096 CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 원하는 아키텍처로 설정
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

add_library(ntt_core
  src/ntt_host.cu
)

add_executable(ntt_demo
  src/main.cu
)

# 128-bit 정수 디바이스 지원(+ 최적화)
target_compile_options(ntt_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
target_compile_options(ntt_demo PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# ▼ 여기서 모드/TPB/상수 인코딩을 컴파일타임으로 선택 가능
# 예) -DUSE_MONTGOMERY=1 -DTILES_PER_BLOCK_S01=2 -DTILES_PER_BLOCK_S23=2 -DMONT_CONST_ENCODE=1
# 기본값은 모두 off(Barrett 기본, TPB=1, 상수 인코딩 off)

target_link_libraries(ntt_demo PRIVATE ntt_core)
