#include "model.h"
#include "gguf.h"
#include "quant.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

static size_t ggml_type_row_size(uint32_t type, size_t ne0) {
    switch (type) {
        case GGML_TYPE_F32: return ne0 * 4;
        case GGML_TYPE_F16: return ne0 * 2;
        case GGML_TYPE_Q8_0: return (ne0 / 32) * sizeof(block_q8_0);
        case GGML_TYPE_Q4_K: return (ne0 / 256) * sizeof(block_q4_K);
        case GGML_TYPE_Q5_K: return (ne0 / 256) * 176;
        case GGML_TYPE_Q6_K: return (ne0 / 256) * 210;
        default: return ne0 * 4;
    }
}

size_t checked_mul(size_t a, size_t b, const char* label) {
    if (a == 0 || b == 0) {
        return 0;
    }
    if (a > SIZE_MAX / b) {
        fprintf(stderr, "size overflow while computing %s\n", label);
        exit(EXIT_FAILURE);
    }
    return a * b;
}

size_t config_weight_floats(const Config* p) {
    const size_t dim = (size_t)p->dim;
    const size_t hidden_dim = (size_t)p->hidden_dim;
    const size_t n_layers = (size_t)p->n_layers;
    const size_t vocab_size = (size_t)p->vocab_size;
    size_t total = 0;
    total += checked_mul(vocab_size, dim, "token_embedding_table");
    total += checked_mul(n_layers, dim, "rms_att_weight");
    total += checked_mul(n_layers, dim, "rms_ffn_weight");
    total += checked_mul(checked_mul(n_layers, dim, "wq"), dim, "wq");
    total += checked_mul(checked_mul(n_layers, dim, "wk"), dim, "wk");
    total += checked_mul(checked_mul(n_layers, dim, "wv"), dim, "wv");
    total += checked_mul(checked_mul(n_layers, dim, "wo"), dim, "wo");
    total += checked_mul(checked_mul(n_layers, hidden_dim, "w1"), dim, "w1");
    total += checked_mul(checked_mul(n_layers, dim, "w2"), hidden_dim, "w2");
    total += checked_mul(checked_mul(n_layers, hidden_dim, "w3"), dim, "w3");
    total += dim;
    total += checked_mul(vocab_size, dim, "wcls");
    return total;
}

size_t run_state_floats(const Config* p) {
    const size_t dim = (size_t)p->dim;
    const size_t hidden_dim = (size_t)(p->hidden_dim > 0 ? p->hidden_dim : 6144);
    const size_t n_heads = (size_t)p->n_heads;
    const size_t head_dim = dim / (n_heads > 0 ? n_heads : 1);
    const size_t q_dim = n_heads * head_dim > dim ? n_heads * head_dim * 2 : dim * 4;
    const size_t vocab_size = (size_t)p->vocab_size;

    size_t total = 0;
    total += dim * 8;
    total += hidden_dim * 4;
    total += q_dim * 4;
    total += checked_mul(n_heads, (size_t)p->seq_len, "att");
    total += vocab_size + 4096;
    /* KV cache is allocated separately — not included in main buffer */
    return total;
}

static size_t kv_cache_bytes_per_element(int kv_type) {
    switch (kv_type) {
        case 1: return 2;  /* FP16 */
        case 2: return 1;  /* Q8_0: ~1 byte per element (block overhead averaged) */
        default: return 4; /* FP32 */
    }
}

void malloc_run_state(RunState* s, const Config* p) {
    memset(s, 0, sizeof(*s));
    size_t total = run_state_floats(p);
    s->memory = (float*)calloc(total, sizeof(float));
    if (s->memory == NULL) {
        fprintf(stderr, "failed to allocate run state (%zu floats)\n", total);
        exit(EXIT_FAILURE);
    }
    s->memory_floats = total;

    const size_t dim = (size_t)p->dim;
    const size_t n_heads = (size_t)p->n_heads;
    const size_t head_dim = dim / (n_heads > 0 ? n_heads : 1);
    const size_t q_dim = n_heads * head_dim > dim ? n_heads * head_dim * 2 : dim * 4;
    const size_t hidden_dim = (size_t)(p->hidden_dim > 0 ? p->hidden_dim : 6144);

    float* ptr = s->memory;
    s->x = ptr; ptr += dim * 2;
    s->xb = ptr; ptr += dim * 2;
    s->xb2 = ptr; ptr += dim * 2;
    s->hb = ptr; ptr += hidden_dim * 2;
    s->hb2 = ptr; ptr += hidden_dim * 2;
    s->q = ptr; ptr += q_dim;
    s->k = ptr; ptr += q_dim;
    s->v = ptr; ptr += q_dim;
    s->att = ptr; ptr += (size_t)p->n_heads * (size_t)p->seq_len;
    s->logits = ptr;

    /* Allocate KV cache separately (default FP32, can be overridden) */
    s->kv_cache_type = 0; /* FP32 default */
    size_t kv_elements = (size_t)p->n_layers * (size_t)p->seq_len * dim;
    size_t bytes_per_elem = kv_cache_bytes_per_element(s->kv_cache_type);
    s->kv_memory_bytes = 2 * kv_elements * bytes_per_elem;
    s->kv_memory = calloc(1, s->kv_memory_bytes);
    if (s->kv_memory == NULL) {
        fprintf(stderr, "failed to allocate KV cache (%zu bytes)\n", s->kv_memory_bytes);
        exit(EXIT_FAILURE);
    }
    s->key_cache = s->kv_memory;
    s->value_cache = (char*)s->kv_memory + kv_elements * bytes_per_elem;
}

void malloc_run_state_kv(RunState* s, const Config* p, int kv_type) {
    malloc_run_state(s, p);
    if (kv_type == s->kv_cache_type) return;

    /* Reallocate KV cache with the requested quantization type */
    free(s->kv_memory);
    s->kv_cache_type = kv_type;
    size_t dim = (size_t)p->dim;
    size_t kv_elements = (size_t)p->n_layers * (size_t)p->seq_len * dim;
    size_t bytes_per_elem = kv_cache_bytes_per_element(kv_type);
    s->kv_memory_bytes = 2 * kv_elements * bytes_per_elem;
    s->kv_memory = calloc(1, s->kv_memory_bytes);
    if (s->kv_memory == NULL) {
        fprintf(stderr, "failed to allocate quantized KV cache (%zu bytes)\n", s->kv_memory_bytes);
        exit(EXIT_FAILURE);
    }
    s->key_cache = s->kv_memory;
    s->value_cache = (char*)s->kv_memory + kv_elements * bytes_per_elem;
}

void free_run_state(RunState* s) {
    free(s->memory);
    free(s->kv_memory);
    memset(s, 0, sizeof(*s));
}

#ifdef _WIN32
static void* map_file_readonly_path(const char* path, size_t* out_size, HANDLE* out_file_handle, HANDLE* out_map_handle) {
    HANDLE hfile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if (hfile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "CreateFileA failed for %s (error %lu)\n", path, GetLastError());
        return NULL;
    }
    LARGE_INTEGER size;
    if (!GetFileSizeEx(hfile, &size)) {
        CloseHandle(hfile);
        return NULL;
    }
    *out_size = (size_t)size.QuadPart;
    HANDLE hmap = CreateFileMappingA(hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (hmap == NULL) {
        fprintf(stderr, "CreateFileMappingA failed (error %lu)\n", GetLastError());
        CloseHandle(hfile);
        return NULL;
    }
    void* data = MapViewOfFile(hmap, FILE_MAP_READ, 0, 0, 0);
    if (data == NULL) {
        fprintf(stderr, "MapViewOfFile failed (error %lu)\n", GetLastError());
        CloseHandle(hmap);
        CloseHandle(hfile);
        return NULL;
    }
    *out_file_handle = hfile;
    *out_map_handle = hmap;
    printf("Successfully memory-mapped %s (%llu bytes)\n", path, (unsigned long long)*out_size);
    return data;
}
#else
static void* map_file_readonly(int fd, size_t* out_size) {
    struct stat st;
    if (fstat(fd, &st) != 0) {
        return NULL;
    }
    *out_size = (size_t)st.st_size;
    void* data = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        return NULL;
    }
    return data;
}
#endif

static void map_weights(Transformer* t) {
    Config* p = &t->config;
    float* ptr = (float*)((char*)t->mapped_data + sizeof(Config));
    t->weights.token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * (size_t)p->dim;
    
    t->weights.rms_att_weight = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.rms_ffn_weight = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.wq = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.wk = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.wv = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.wo = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.w1 = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.w2 = (float**)malloc(p->n_layers * sizeof(float*));
    t->weights.w3 = (float**)malloc(p->n_layers * sizeof(float*));

    for (int l = 0; l < p->n_layers; l++) {
        t->weights.rms_att_weight[l] = ptr; ptr += (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.rms_ffn_weight[l] = ptr; ptr += (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.wq[l] = ptr; ptr += (size_t)p->dim * (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.wk[l] = ptr; ptr += (size_t)p->dim * (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.wv[l] = ptr; ptr += (size_t)p->dim * (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.wo[l] = ptr; ptr += (size_t)p->dim * (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.w1[l] = ptr; ptr += (size_t)p->hidden_dim * (size_t)p->dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.w2[l] = ptr; ptr += (size_t)p->dim * (size_t)p->hidden_dim;
    }
    for (int l = 0; l < p->n_layers; l++) {
        t->weights.w3[l] = ptr; ptr += (size_t)p->hidden_dim * (size_t)p->dim;
    }

    t->weights.rms_final_weight = ptr;
    ptr += (size_t)p->dim;
    t->weights.wcls = ptr;
}

static void load_gguf_transformer(Transformer* t) {
    gguf_context* ctx = gguf_init_from_buffer(t->mapped_data, t->mapped_size);
    if (!ctx) {
        fprintf(stderr, "failed to parse GGUF\n");
        exit(EXIT_FAILURE);
    }
    t->gguf_ctx = ctx;

    Config* p = &t->config;
    const char* arch_val = gguf_get_val_str(ctx, "general.architecture");
    char arch[64] = "qwen2";
    if (arch_val) {
        strncpy(arch, arch_val, sizeof(arch) - 1);
        arch[sizeof(arch) - 1] = '\0';
    }
    printf("Detected GGUF architecture: %s\n", arch);

    char key_buf[128];
    sprintf(key_buf, "%s.embedding_length", arch);
    p->dim = gguf_get_val_u32(ctx, key_buf, 0);

    sprintf(key_buf, "%s.block_count", arch);
    p->n_layers = gguf_get_val_u32(ctx, key_buf, 0);

    sprintf(key_buf, "%s.attention.head_count", arch);
    p->n_heads = gguf_get_val_u32(ctx, key_buf, 0);

    sprintf(key_buf, "%s.attention.head_count_kv", arch);
    p->n_kv_heads = gguf_get_val_u32(ctx, key_buf, p->n_heads);

    sprintf(key_buf, "%s.expert_count", arch);
    p->expert_count = gguf_get_val_u32(ctx, key_buf, 0);

    sprintf(key_buf, "%s.expert_used_count", arch);
    p->expert_used_count = gguf_get_val_u32(ctx, key_buf, 0);

    sprintf(key_buf, "%s.expert_feed_forward_length", arch);
    uint32_t exp_ffn = gguf_get_val_u32(ctx, key_buf, 0);
    if (p->expert_count > 0 && exp_ffn > 0) {
        p->hidden_dim = exp_ffn;
    } else {
        sprintf(key_buf, "%s.feed_forward_length", arch);
        p->hidden_dim = gguf_get_val_u32(ctx, key_buf, 0);
    }

    sprintf(key_buf, "%s.rope.freq_base", arch);
    p->rope_freq_base = gguf_get_val_f32(ctx, key_buf, 1000000.0f);

    // Cap runtime KV cache allocation to 2048 tokens to stay within strict <4GB budget
    p->seq_len = 2048;

    char** tokens_arr = NULL;
    uint64_t tokens_len = 0;
    if (gguf_get_val_str_array(ctx, "tokenizer.ggml.tokens", &tokens_arr, &tokens_len) == 0) {
        p->vocab_size = (int)tokens_len;
    } else {
        p->vocab_size = 152064; // default Qwen vocab size
    }

    TransformerWeights* w = &t->weights;
    
    // Standard layers
    gguf_tensor* tok = gguf_find_tensor(ctx, "token_embd.weight");
    if (tok) {
        w->token_embedding_table = tok->data;
        w->token_embd_type = tok->type;
        w->token_embd_row_bytes = ggml_type_row_size(tok->type, p->dim);
    } else {
        w->token_embd_type = 0;
        w->token_embd_row_bytes = (size_t)p->dim * sizeof(float);
    }
    
    gguf_tensor* norm = gguf_find_tensor(ctx, "output_norm.weight");
    if (norm) w->rms_final_weight = norm->data;

    gguf_tensor* wcls = gguf_find_tensor(ctx, "output.weight");
    if (wcls) {
        w->wcls = wcls->data;
        w->wcls_type = wcls->type;
    } else {
        w->wcls = w->token_embedding_table;
        w->wcls_type = w->token_embd_type;
    }

    // Allocate array of pointers for layers if they were contiguous, but in GGUF they are separate.
    w->rms_att_weight = (float**)malloc(p->n_layers * sizeof(float*));
    w->rms_ffn_weight = (float**)malloc(p->n_layers * sizeof(float*));
    w->wq = (float**)malloc(p->n_layers * sizeof(float*));
    w->wk = (float**)malloc(p->n_layers * sizeof(float*));
    w->wv = (float**)malloc(p->n_layers * sizeof(float*));
    w->wo = (float**)malloc(p->n_layers * sizeof(float*));
    w->w1 = (float**)malloc(p->n_layers * sizeof(float*));
    w->w2 = (float**)malloc(p->n_layers * sizeof(float*));
    w->w3 = (float**)malloc(p->n_layers * sizeof(float*));
    w->wq_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->wk_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->wv_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->wo_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->exp_w1_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->exp_w2_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->exp_w3_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->w1_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->w2_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));
    w->w3_type = (uint32_t*)calloc(p->n_layers, sizeof(uint32_t));

    if (p->expert_count > 0) {
        w->ffn_gate_inp = (float**)malloc(p->n_layers * sizeof(float*));
        w->expert_w1 = (float***)malloc(p->n_layers * sizeof(float**));
        w->expert_w2 = (float***)malloc(p->n_layers * sizeof(float**));
        w->expert_w3 = (float***)malloc(p->n_layers * sizeof(float**));
        for (int l = 0; l < p->n_layers; l++) {
            w->expert_w1[l] = (float**)malloc(p->expert_count * sizeof(float*));
            w->expert_w2[l] = (float**)malloc(p->expert_count * sizeof(float*));
            w->expert_w3[l] = (float**)malloc(p->expert_count * sizeof(float*));
        }
    }

    char name_buf[256];
    for (int l = 0; l < p->n_layers; l++) {
        sprintf(name_buf, "blk.%d.attn_norm.weight", l);
        gguf_tensor* ts = gguf_find_tensor(ctx, name_buf);
        if (ts) w->rms_att_weight[l] = ts->data;
        
        sprintf(name_buf, "blk.%d.ffn_norm.weight", l);
        ts = gguf_find_tensor(ctx, name_buf);
        if (ts) w->rms_ffn_weight[l] = ts->data;

        sprintf(name_buf, "blk.%d.attn_q.weight", l);
        ts = gguf_find_tensor(ctx, name_buf);
        if (ts) { w->wq[l] = ts->data; w->wq_type[l] = ts->type; }

        sprintf(name_buf, "blk.%d.attn_k.weight", l);
        ts = gguf_find_tensor(ctx, name_buf);
        if (ts) { w->wk[l] = ts->data; w->wk_type[l] = ts->type; }

        sprintf(name_buf, "blk.%d.attn_v.weight", l);
        ts = gguf_find_tensor(ctx, name_buf);
        if (ts) { w->wv[l] = ts->data; w->wv_type[l] = ts->type; }

        sprintf(name_buf, "blk.%d.attn_output.weight", l);
        ts = gguf_find_tensor(ctx, name_buf);
        if (ts) { w->wo[l] = ts->data; w->wo_type[l] = ts->type; }

        if (p->expert_count > 0) {
            sprintf(name_buf, "blk.%d.ffn_gate_inp.weight", l);
            ts = gguf_find_tensor(ctx, name_buf);
            if (ts) w->ffn_gate_inp[l] = ts->data;

            sprintf(name_buf, "blk.%d.ffn_up_exps.weight", l);
            gguf_tensor* ts_up = gguf_find_tensor(ctx, name_buf);
            sprintf(name_buf, "blk.%d.ffn_down_exps.weight", l);
            gguf_tensor* ts_down = gguf_find_tensor(ctx, name_buf);
            sprintf(name_buf, "blk.%d.ffn_gate_exps.weight", l);
            gguf_tensor* ts_gate = gguf_find_tensor(ctx, name_buf);

            if (ts_up && ts_down && ts_gate) {
                w->exp_w1_type[l] = ts_up->type;
                w->exp_w2_type[l] = ts_down->type;
                w->exp_w3_type[l] = ts_gate->type;
                size_t slice_up = ggml_type_row_size(ts_up->type, ts_up->ne[0]) * ts_up->ne[1];
                size_t slice_down = ggml_type_row_size(ts_down->type, ts_down->ne[0]) * ts_down->ne[1];
                size_t slice_gate = ggml_type_row_size(ts_gate->type, ts_gate->ne[0]) * ts_gate->ne[1];
                for (int e = 0; e < p->expert_count; e++) {
                    w->expert_w1[l][e] = (float*)((char*)ts_up->data + (size_t)e * slice_up);
                    w->expert_w2[l][e] = (float*)((char*)ts_down->data + (size_t)e * slice_down);
                    w->expert_w3[l][e] = (float*)((char*)ts_gate->data + (size_t)e * slice_gate);
                }
            } else {
                for (int e = 0; e < p->expert_count; e++) {
                    sprintf(name_buf, "blk.%d.ffn_up_exps.%d.weight", l, e);
                    ts = gguf_find_tensor(ctx, name_buf);
                    if (ts) { w->expert_w1[l][e] = ts->data; w->exp_w1_type[l] = ts->type; }
                    
                    sprintf(name_buf, "blk.%d.ffn_down_exps.%d.weight", l, e);
                    ts = gguf_find_tensor(ctx, name_buf);
                    if (ts) { w->expert_w2[l][e] = ts->data; w->exp_w2_type[l] = ts->type; }
                    
                    sprintf(name_buf, "blk.%d.ffn_gate_exps.%d.weight", l, e);
                    ts = gguf_find_tensor(ctx, name_buf);
                    if (ts) { w->expert_w3[l][e] = ts->data; w->exp_w3_type[l] = ts->type; }
                }
            }
        } else {
            sprintf(name_buf, "blk.%d.ffn_up.weight", l);
            ts = gguf_find_tensor(ctx, name_buf);
            if (ts) { w->w1[l] = ts->data; w->w1_type[l] = ts->type; }

            sprintf(name_buf, "blk.%d.ffn_down.weight", l);
            ts = gguf_find_tensor(ctx, name_buf);
            if (ts) { w->w2[l] = ts->data; w->w2_type[l] = ts->type; }

            sprintf(name_buf, "blk.%d.ffn_gate.weight", l);
            ts = gguf_find_tensor(ctx, name_buf);
            if (ts) { w->w3[l] = ts->data; w->w3_type[l] = ts->type; }
        }
    }
}

void load_transformer(Transformer* t, const char* model_path) {
    memset(t, 0, sizeof(*t));
#ifdef _WIN32
    HANDLE hfile = INVALID_HANDLE_VALUE;
    HANDLE hmap = NULL;
    t->mapped_data = map_file_readonly_path(model_path, &t->mapped_size, &hfile, &hmap);
    t->map_handle = (void*)hmap;
    t->mapped_fd = (int)(intptr_t)hfile;
#else
    t->mapped_fd = open(model_path, O_RDONLY);
    if (t->mapped_fd < 0) {
        fprintf(stderr, "open model failed: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    t->mapped_data = map_file_readonly(t->mapped_fd, &t->mapped_size);
#endif
    if (t->mapped_data == NULL) {
        fprintf(stderr, "map model failed\n");
        exit(EXIT_FAILURE);
    }

    // Check extension
    size_t len = strlen(model_path);
    if (len > 5 && strcmp(model_path + len - 5, ".gguf") == 0) {
        load_gguf_transformer(t);
    } else {
        memcpy(&t->config, t->mapped_data, sizeof(Config));
        map_weights(t);
    }
    
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->weights.rms_att_weight) {
        free(t->weights.rms_att_weight);
        free(t->weights.rms_ffn_weight);
        free(t->weights.wq);
        free(t->weights.wk);
        free(t->weights.wv);
        free(t->weights.wo);
        free(t->weights.w1);
        free(t->weights.w2);
        free(t->weights.w3);
        if (t->weights.wq_type) free(t->weights.wq_type);
        if (t->weights.wk_type) free(t->weights.wk_type);
        if (t->weights.wv_type) free(t->weights.wv_type);
        if (t->weights.wo_type) free(t->weights.wo_type);
        if (t->weights.exp_w1_type) free(t->weights.exp_w1_type);
        if (t->weights.exp_w2_type) free(t->weights.exp_w2_type);
        if (t->weights.exp_w3_type) free(t->weights.exp_w3_type);
        if (t->weights.w1_type) free(t->weights.w1_type);
        if (t->weights.w2_type) free(t->weights.w2_type);
        if (t->weights.w3_type) free(t->weights.w3_type);
        if (t->config.expert_count > 0 && t->weights.ffn_gate_inp) {
            free(t->weights.ffn_gate_inp);
            for (int l = 0; l < t->config.n_layers; l++) {
                free(t->weights.expert_w1[l]);
                free(t->weights.expert_w2[l]);
                free(t->weights.expert_w3[l]);
            }
            free(t->weights.expert_w1);
            free(t->weights.expert_w2);
            free(t->weights.expert_w3);
        }
    }

    if (t->gguf_ctx) {
        gguf_free((gguf_context*)t->gguf_ctx);
    }
    free_run_state(&t->state);
    if (t->mapped_data != NULL) {
#ifdef _WIN32
        UnmapViewOfFile(t->mapped_data);
        if (t->map_handle != NULL) {
            CloseHandle((HANDLE)t->map_handle);
        }
        if (t->mapped_fd != 0) {
            CloseHandle((HANDLE)(intptr_t)t->mapped_fd);
        }
#else
        munmap(t->mapped_data, t->mapped_size);
        if (t->mapped_fd >= 0) {
            close(t->mapped_fd);
        }
#endif
    }
}

void write_dummy_model(const char* path) {
    Config c = {64, 128, 4, 4, 4, 256, 128, 10000.0f, 0, 0};
    size_t n_floats = config_weight_floats(&c);
    float* weights = (float*)malloc(sizeof(float) * n_floats);
    if (weights == NULL) {
        fprintf(stderr, "malloc dummy weights failed\n");
        exit(EXIT_FAILURE);
    }
    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < n_floats; i++) {
        weights[i] = (((float)rand() / (float)RAND_MAX) - 0.5f) * 0.02f;
    }
    FILE* f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "open output failed\n");
        free(weights);
        exit(EXIT_FAILURE);
    }
    fwrite(&c, sizeof(Config), 1, f);
    fwrite(weights, sizeof(float), n_floats, f);
    fclose(f);
    free(weights);
}
