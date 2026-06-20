#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef struct {
    int threads;
    int deterministic;
} ThreadPool;

void threadpool_init(ThreadPool* p, int threads, int deterministic);
void threadpool_destroy(ThreadPool* p);
void threadpool_bind_affinity(ThreadPool* p);
void threadpool_parallel_for(ThreadPool* p, int start, int end, void (*fn)(int, void*), void* ctx);

#endif
