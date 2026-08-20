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
    if (len > (size_t)(end - *ptr)) return NULL;
    if (len > 1024 * 1024) return NULL; /* Safety cap: max 1MB string */
    char* str = (char*)malloc((size_t)len + 1);
    if (!str) return NULL;
    memcpy(str, *ptr, (size_t)len);
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
        return NULL;
    }
    uint32_t magic;
    if (!read_u32(&ptr, end, &magic)) return NULL;
    if (magic != GGUF_MAGIC) {
        return NULL;
    }

    uint32_t version;
    if (!read_u32(&ptr, end, &version)) return NULL;

    gguf_context* ctx = (gguf_context*)calloc(1, sizeof(gguf_context));
    if (!ctx) return NULL;
    ctx->data = buffer;
    ctx->size = size;
    ctx->version = version;
    
    if (!read_u64(&ptr, end, &ctx->tensor_count)) goto error;
    if (!read_u64(&ptr, end, &ctx->kv_count)) goto error;

    if (ctx->kv_count > 65536) goto error; /* Sanity limit on KV entries */
    if (ctx->tensor_count > 65536) goto error; /* Sanity limit on tensors */

    ctx->kv = (gguf_kv*)calloc(ctx->kv_count, sizeof(gguf_kv));
    if (!ctx->kv && ctx->kv_count > 0) goto error;

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
            if (arr_len > 1000000) goto error; /* Sanity limit */
            ctx->kv[i].len = arr_len;
            ctx->kv[i].arr_type = arr_type;
            
            if (arr_type == GGUF_TYPE_STRING) {
                char** arr = (char**)calloc(arr_len, sizeof(char*));
                if (!arr && arr_len > 0) goto error;
                ctx->kv[i].value = arr;
                for (uint64_t j = 0; j < arr_len; j++) {
                    arr[j] = read_string(&ptr, end);
                    if (!arr[j]) goto error;
                }
            } else {
                size_t el_sz = gguf_type_size(arr_type);
                if (el_sz == 0) goto error;
                if (arr_len > (size_t)(end - ptr) / el_sz) goto error;
                void* arr = malloc((size_t)arr_len * el_sz);
                if (!arr && arr_len > 0) goto error;
                if (arr) memcpy(arr, ptr, (size_t)arr_len * el_sz);
                ctx->kv[i].value = arr;
                ptr += (size_t)arr_len * el_sz;
            }
        } else {
            size_t el_sz = gguf_type_size(ctx->kv[i].type);
            if (el_sz == 0) goto error;
            if ((size_t)(end - ptr) < el_sz) goto error;
            ctx->kv[i].value = malloc(el_sz);
            if (!ctx->kv[i].value) goto error;
            memcpy(ctx->kv[i].value, ptr, el_sz);
            ptr += el_sz;
        }
    }

    ctx->alignment = 32;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (ctx->kv[i].name && strcmp(ctx->kv[i].name, "general.alignment") == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_UINT32 && ctx->kv[i].value) {
                ctx->alignment = *(uint32_t*)ctx->kv[i].value;
            } else if (ctx->kv[i].type == GGUF_TYPE_UINT64 && ctx->kv[i].value) {
                ctx->alignment = *(uint64_t*)ctx->kv[i].value;
            }
        }
    }
    if (ctx->alignment == 0) ctx->alignment = 32;

    ctx->tensors = (gguf_tensor*)calloc(ctx->tensor_count, sizeof(gguf_tensor));
    if (!ctx->tensors && ctx->tensor_count > 0) goto error;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        ctx->tensors[i].name = read_string(&ptr, end);
        if (!ctx->tensors[i].name) goto error;
        
        if (!read_u32(&ptr, end, &ctx->tensors[i].n_dims)) goto error;
        if (ctx->tensors[i].n_dims > 4) goto error;
        
        for (uint32_t j = 0; j < ctx->tensors[i].n_dims; j++) {
            if (!read_u64(&ptr, end, &ctx->tensors[i].ne[j])) goto error;
        }
        
        if (!read_u32(&ptr, end, &ctx->tensors[i].type)) goto error;
        if (!read_u64(&ptr, end, &ctx->tensors[i].offset)) goto error;
    }

    size_t header_size = (size_t)(ptr - (const uint8_t*)buffer);
    size_t rem = header_size % ctx->alignment;
    size_t pad = rem ? (ctx->alignment - rem) : 0;
    size_t data_offset = header_size + pad;
    ctx->data_offset = data_offset;

    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        size_t tensor_start = data_offset + ctx->tensors[i].offset;
        if (tensor_start > size) goto error; 
        ctx->tensors[i].data = (void*)((uint8_t*)buffer + tensor_start);
    }

    return ctx;

error:
    gguf_free(ctx);
    return NULL;
}

void gguf_free(gguf_context* ctx) {
    if (!ctx) return;
    if (ctx->kv) {
        for (uint64_t i = 0; i < ctx->kv_count; i++) {
            if (ctx->kv[i].name) {
                free(ctx->kv[i].name);
                ctx->kv[i].name = NULL;
            }
            if (ctx->kv[i].value) {
                if (ctx->kv[i].type == GGUF_TYPE_ARRAY && ctx->kv[i].arr_type == GGUF_TYPE_STRING) {
                    char** arr = (char**)ctx->kv[i].value;
                    for (uint64_t j = 0; j < ctx->kv[i].len; j++) {
                        if (arr[j]) free(arr[j]);
                    }
                }
                free(ctx->kv[i].value);
                ctx->kv[i].value = NULL;
            }
        }
        free(ctx->kv);
        ctx->kv = NULL;
    }
    if (ctx->tensors) {
        for (uint64_t i = 0; i < ctx->tensor_count; i++) {
            if (ctx->tensors[i].name) free(ctx->tensors[i].name);
        }
        free(ctx->tensors);
        ctx->tensors = NULL;
    }
    free(ctx);
}

gguf_tensor* gguf_find_tensor(gguf_context* ctx, const char* name) {
    if (!ctx || !ctx->tensors || !name) return NULL;
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (ctx->tensors[i].name && strcmp(ctx->tensors[i].name, name) == 0) {
            return &ctx->tensors[i];
        }
    }
    return NULL;
}

uint32_t gguf_get_val_u32(gguf_context* ctx, const char* key, uint32_t def) {
    if (!ctx || !ctx->kv || !key) return def;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (ctx->kv[i].name && strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_UINT32 && ctx->kv[i].value) {
                return *(uint32_t*)ctx->kv[i].value;
            }
        }
    }
    return def;
}

float gguf_get_val_f32(gguf_context* ctx, const char* key, float def) {
    if (!ctx || !ctx->kv || !key) return def;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (ctx->kv[i].name && strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_FLOAT32 && ctx->kv[i].value) {
                return *(float*)ctx->kv[i].value;
            }
        }
    }
    return def;
}

const char* gguf_get_val_str(gguf_context* ctx, const char* key) {
    if (!ctx || !ctx->kv || !key) return NULL;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (ctx->kv[i].name && strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_STRING && ctx->kv[i].value) {
                return (const char*)ctx->kv[i].value;
            }
        }
    }
    return NULL;
}

int gguf_get_val_str_array(gguf_context* ctx, const char* key, char*** arr_out, uint64_t* len_out) {
    if (!ctx || !ctx->kv || !key || !arr_out || !len_out) return -1;
    for (uint64_t i = 0; i < ctx->kv_count; i++) {
        if (ctx->kv[i].name && strcmp(ctx->kv[i].name, key) == 0) {
            if (ctx->kv[i].type == GGUF_TYPE_ARRAY && ctx->kv[i].value) {
                *arr_out = (char**)ctx->kv[i].value;
                *len_out = ctx->kv[i].len;
                return 0;
            }
        }
    }
    return -1;
}
