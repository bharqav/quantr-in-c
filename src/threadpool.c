#include "threadpool.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#endif

void threadpool_init(ThreadPool* p, int threads, int deterministic) {
    p->threads = threads > 0 ? threads : 1;
    p->deterministic = deterministic;
}

void threadpool_destroy(ThreadPool* p) {
    (void)p;
}

void threadpool_bind_affinity(ThreadPool* p) {
#ifdef _OPENMP
    if (!p || p->threads <= 1) return;
    #pragma omp parallel num_threads(p->threads)
    {
        int tid = omp_get_thread_num();
#ifdef _WIN32
        DWORD_PTR mask = (DWORD_PTR)1 << (tid % (sizeof(DWORD_PTR) * 8));
        SetThreadAffinityMask(GetCurrentThread(), mask);
#elif defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid % CPU_SETSIZE, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }
#else
    (void)p;
#endif
}

void threadpool_parallel_for(ThreadPool* p, int start, int end, void (*fn)(int, void*), void* ctx) {
    if (p->deterministic || p->threads <= 1) {
        for (int i = start; i < end; i++) {
            fn(i, ctx);
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for num_threads(p->threads) schedule(static)
#endif
    for (int i = start; i < end; i++) {
        fn(i, ctx);
    }
}
