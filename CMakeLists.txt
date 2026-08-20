cmake_minimum_required(VERSION 3.14)
project(baremetal_inference_engine C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

include_directories(include)

find_package(OpenMP)

if(MSVC)
    add_compile_options(/O2 /W3 /Zi)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE)
else()
    add_compile_options(-O3 -g -Wall -Wextra)
endif()

set(CORE_SOURCES
    src/model.c
    src/kernels.c
    src/runtime.c
    src/sampling.c
    src/quant.c
    src/tokenizer.c
    src/context_window.c
    src/vm_terminal.c
    src/benchmark.c
    src/threadpool.c
    src/gguf.c
    src/baremetal.c
    src/server.c
    src/speculative.c
)

add_executable(inference src/main.c ${CORE_SOURCES})
if(OpenMP_C_FOUND)
    target_link_libraries(inference PRIVATE OpenMP::OpenMP_C)
endif()
if(WIN32)
    target_link_libraries(inference PRIVATE ws2_32 psapi)
else()
    target_link_libraries(inference PRIVATE m)
endif()

# Launcher executable
add_executable(quantr src/launcher.c)
if(WIN32)
    target_link_libraries(quantr PRIVATE ws2_32 psapi)
else()
    target_link_libraries(quantr PRIVATE m)
endif()

# Shared and static libraries
add_library(baremetal_shared SHARED ${CORE_SOURCES})
set_target_properties(baremetal_shared PROPERTIES OUTPUT_NAME "baremetal")
if(OpenMP_C_FOUND)
    target_link_libraries(baremetal_shared PRIVATE OpenMP::OpenMP_C)
endif()
if(WIN32)
    target_link_libraries(baremetal_shared PRIVATE ws2_32 psapi)
else()
    target_link_libraries(baremetal_shared PRIVATE m)
endif()

add_library(baremetal_static STATIC ${CORE_SOURCES})
set_target_properties(baremetal_static PROPERTIES OUTPUT_NAME "baremetal")

# Unit tests executable
enable_testing()
add_executable(tests tests/tests.c ${CORE_SOURCES})
if(OpenMP_C_FOUND)
    target_link_libraries(tests PRIVATE OpenMP::OpenMP_C)
endif()
if(WIN32)
    target_link_libraries(tests PRIVATE ws2_32 psapi)
else()
    target_link_libraries(tests PRIVATE m)
endif()

# Fuzzing test executable
add_executable(fuzz_gguf tests/fuzz_gguf.c src/gguf.c)
if(WIN32)
    target_link_libraries(fuzz_gguf PRIVATE ws2_32 psapi)
else()
    target_link_libraries(fuzz_gguf PRIVATE m)
endif()

add_test(NAME UnitTests COMMAND $<TARGET_FILE:tests>)
add_test(NAME FuzzTest COMMAND $<TARGET_FILE:fuzz_gguf>)
