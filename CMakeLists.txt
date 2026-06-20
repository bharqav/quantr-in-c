cmake_minimum_required(VERSION 3.14)
project(baremetal_inference_engine C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

find_package(OpenMP REQUIRED)

if(MSVC)
    add_compile_options(/O2 /W4 /arch:AVX2 /Zi)
else()
    add_compile_options(-O3 -g -Wall -Wextra -march=haswell)
endif()

set(CORE_SOURCES
    src/model.c
    src/kernels.c
    src/runtime.c
    src/sampling.c
    src/quant.c
    src/tokenizer.c
    src/benchmark.c
    src/threadpool.c
    src/gguf.c
    src/baremetal.c
    src/server.c
    src/speculative.c
)

add_executable(inference src/main.c ${CORE_SOURCES})
target_link_libraries(inference PRIVATE OpenMP::OpenMP_C)
if(WIN32)
    target_link_libraries(inference PRIVATE ws2_32 psapi)
else()
    target_link_libraries(inference PRIVATE m)
endif()

# Unit tests executable
enable_testing()
add_executable(tests tests/tests.c ${CORE_SOURCES})
target_link_libraries(tests PRIVATE OpenMP::OpenMP_C)
if(WIN32)
    target_link_libraries(tests PRIVATE ws2_32 psapi)
else()
    target_link_libraries(tests PRIVATE m)
endif()

# Fuzzing test executable
add_executable(fuzz_gguf tests/fuzz_gguf.c src/gguf.c)

add_test(NAME UnitTests COMMAND tests)
add_test(NAME FuzzTest COMMAND fuzz_gguf)

