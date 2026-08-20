#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGUF_MAGIC 0x46554747 // "GGUF"
#define GGUF_VERSION 3

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// These match llama.cpp GGML types
enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // ... we care mostly about Q4_K and Q8_0
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_I8   = 16,
    GGML_TYPE_I16  = 17,
    GGML_TYPE_I32  = 18,
    GGML_TYPE_I64  = 19,
};

typedef struct {
    char* name;
    uint32_t type;
    uint32_t n_dims;
    uint64_t ne[4]; // dimensions
    uint64_t offset;
    void* data; // pointer into memory mapped file
} gguf_tensor;

typedef struct {
    char* name;
    uint32_t type;
    void* value; // raw value pointer
    uint64_t len; // for strings and arrays
    uint32_t arr_type;
} gguf_kv;

typedef struct {
    void* data;
    size_t size;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    gguf_kv* kv;
    gguf_tensor* tensors;
    uint64_t alignment;
    uint64_t data_offset;
} gguf_context;

// Parse a GGUF file from a memory mapped buffer
gguf_context* gguf_init_from_buffer(void* buffer, size_t size);
void gguf_free(gguf_context* ctx);

// Lookups
gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name);
uint32_t gguf_get_val_u32(gguf_context* ctx, const char* key, uint32_t def);
float gguf_get_val_f32(gguf_context* ctx, const char* key, float def);
const char* gguf_get_val_str(gguf_context* ctx, const char* key);
int gguf_get_val_str_array(gguf_context* ctx, const char* key, char*** arr_out, uint64_t* len_out);

#endif // GGUF_H
