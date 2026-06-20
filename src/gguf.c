#include "gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static int read_u32(const uint8_t** ptr, const uint8_t* end, uint32_t* out) {
    if (*ptr + 4 > end) return 0;
    memcpy(out, *ptr, 4);
    *ptr += 4;
    return 1;
}

static int read_u64(const uint8_t** ptr, const uint8_t* end, uint64_t* out) {
    if (*ptr + 8 > end) return 0;
    memcpy(out, *ptr, 8);
    *ptr += 8;
    return 1;
}

static char* read_string(const uint8_t** ptr, const uint8_t* end) {
    uint64_t len;
    if (!read_u64(ptr, end, &len)) return NULL;
    if (*ptr + len > end) return NULL;
    char* str = (char*)malloc(len + 1);
    if (!str) return NULL;
    memcpy(str, *ptr, len);
    str[len] = '\0';
    *ptr += len;
    return str;
}

static size_t gguf_type_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

gguf_context* gguf_init_from_buffer(void* buffer, size_t size) {
    const uint8_t* ptr = (const uint8_t*)buffer;
    const uint8_t* end = ptr + size;

    if (ptr + 4 > end) {
        fprintf(stderr, "GGUF error: buffer too small for magic\n");
        return NULL;
    }
    uint32_t magic;
    if (!read_u32(&ptr, end, &magic)) return NULL;
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF error: invalid magic 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        return NULL;
    }

    uint32_t version;
    if (!read_u32(&ptr, end, &version)) return NULL;
    printf("GGUF: magic OK, version = %u\n", version);

    gguf_context* ctx = (gguf_context*)calloc(1, sizeof(gguf_context));
    ctx->data = buffer;
    ctx->size = size;
    ctx->version = version;
    
    if (!read_u64(&ptr, end, &ctx->tensor_count)) goto error;
    if (!read_u64(&ptr, end, &ctx->kv_count)) goto error;

    ctx->kv = (gguf_kv*)calloc(ctx->kv_count, sizeof(gguf_kv));
    if (!ctx->kv) goto error;

    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        ctx->kv[i].name = read_string(&ptr, end);
        if (!ctx->kv[i].name) goto error;
        
        uint32_t type;
        if (!read_u32(&ptr, end, &type)) goto error;
        ctx->kv[i].type = type;

        if (ctx->kv[i].type == GGUF_TYPE_STRING) {
            ctx->kv[i].value = read_string(&ptr, end);
            if (!ctx->kv[i].value) goto error;
        } else if (ctx->kv[i].type == GGUF_TYPE_ARRAY) {
            uint32_t arr_type;
            if (!read_u32(&ptr, end, &arr_type)) goto error;
            uint64_t arr_len;
            if (!read_u64(&ptr, end, &arr_len)) goto error;
            ctx->kv[i].len = arr_len;
            
            if (arr_type == GGUF_TYPE_STRING) {
                char** arr = (char**)malloc(arr_len * sizeof(char*));
                if (!arr) goto error;
                for (uint64_t j = 0; j < arr_len; j++) {
                    arr[j] = read_string(&ptr, end);
                    if (!arr[j]) {
                        free(arr);
                        goto error;
                    }
                }
                ctx->kv[i].value = arr;
            } else {
                size_t el_sz = gguf_type_size(arr_type);
                if (ptr + arr_len * el_sz > end) goto error;
                void* arr = malloc(arr_len * el_sz);
                if (!arr) goto error;
                memcpy(arr, ptr, arr_len * el_sz);
                ctx->kv[i].value = arr;
                ptr += arr_len * el_sz;
            }
        } else {
            size_t el_sz = gguf_type_size(ctx->kv[i].type);
            if (ptr + el_sz > end) goto error;
            ctx->kv[i].value = malloc(el_sz);
            if (!ctx->kv[i].value) goto error;
            memcpy(ctx->kv[i].value, ptr, el_sz);
            ptr += el_sz;
        }
    }

    ctx->alignment = 32;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->kv[i].name, "general.alignment") == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_UINT32) {
                ctx->alignment = *(uint32_t*)ctx->kv[i].value;
            } else if (ctx->kv[i].type == GGUF_TYPE_UINT64) {
                ctx->alignment = *(uint64_t*)ctx->kv[i].value;
            }
        }
    }

    ctx->tensors = (gguf_tensor*)calloc(ctx->tensor_count, sizeof(gguf_tensor));
    if (!ctx->tensors && ctx->tensor_count > 0) goto error;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        ctx->tensors[i].name = read_string(&ptr, end);
        if (!ctx->tensors[i].name) goto error;
        
        if (!read_u32(&ptr, end, &ctx->tensors[i].n_dims)) goto error;
        
        for (uint32_t j = 0; j < ctx->tensors[i].n_dims; j++) {
            if (!read_u64(&ptr, end, &ctx->tensors[i].ne[j])) goto error;
        }
        
        if (!read_u32(&ptr, end, &ctx->tensors[i].type)) goto error;
        if (!read_u64(&ptr, end, &ctx->tensors[i].offset)) goto error;
    }

    size_t header_size = ptr - (const uint8_t*)buffer;
    size_t data_offset = header_size + (ctx->alignment - (header_size % ctx->alignment)) % ctx->alignment;
    ctx->data_offset = data_offset;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        size_t tensor_start = data_offset + ctx->tensors[i].offset;
        /* Basic sanity check: tensor offset shouldn't exceed file size */
        if (tensor_start > size) goto error; 
        ctx->tensors[i].data = (void*)((uint8_t*)buffer + tensor_start);
    }

    return ctx;

error:
    fprintf(stderr, "GGUF error: file is truncated or malformed\n");
    gguf_free(ctx);
    return NULL;
}

void gguf_free(gguf_context* ctx) {
    if (!ctx) return;
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->kv_count; i++) {
            if (ctx->kv[i].name) free(ctx->kv[i].name);
            if (ctx->kv[i].value) free(ctx->kv[i].value);
        }
        free(ctx->kv);
    }
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            if (ctx->tensors[i].name) free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
    }
    free(ctx);
}

gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name) {
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

uint32_t gguf_get_val_u32(gguf_context* ctx, const char* key, uint32_t def) {
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_UINT32) {
                return *(uint32_t*)ctx->kv[i].value;
            }
        }
    }
    return def;
}

float gguf_get_val_f32(gguf_context* ctx, const char* key, float def) {
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_FLOAT32) {
                return *(float*)ctx->kv[i].value;
            }
        }
    }
    return def;
}

const char* gguf_get_val_str(gguf_context* ctx, const char* key) {
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_STRING) {
                return (const char*)ctx->kv[i].value;
            }
        }
    }
    return NULL;
}

int gguf_get_val_str_array(gguf_context* ctx, const char* key, char*** arr_out, uint64_t* len_out) {
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_ARRAY) {
                *arr_out = (char**)ctx->kv[i].value;
                *len_out = ctx->kv[i].len;
                return 0;
            }
        }
    }
    return -1;
}
