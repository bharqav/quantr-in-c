#include "tokenizer.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf.h"

static char* dup_str(const char* s) {
    size_t n = strlen(s);
    char* d = (char*)malloc(n + 1);
    if (d == NULL) return NULL;
    memcpy(d, s, n + 1);
    return d;
}

int tokenizer_load(Tokenizer* t, const char* path, TokenizerType type_hint) {
    memset(t, 0, sizeof(*t));
    t->type = type_hint;
    FILE* f = fopen(path, "r");
    if (f == NULL) return -1;
    int cap = 1024;
    t->vocab = (char**)calloc((size_t)cap, sizeof(char*));
    if (t->vocab == NULL) {
        fclose(f);
        return -1;
    }
    char line[1024];
    while (fgets(line, sizeof(line), f) != NULL) {
        size_t n = strlen(line);
        while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[--n] = '\0';
        }
        if (n == 0) continue;
        if (t->vocab_size >= cap) {
            cap *= 2;
            char** nv = (char**)realloc(t->vocab, sizeof(char*) * (size_t)cap);
            if (nv == NULL) {
                fclose(f);
                return -1;
            }
            t->vocab = nv;
        }
        t->vocab[t->vocab_size++] = dup_str(line);
    }
    fclose(f);

    for (int i = 0; i < 256; i++) t->bucket_head[i] = -1;
    t->bucket_next = (int*)malloc(sizeof(int) * t->vocab_size);
    for (int i = 0; i < t->vocab_size; i++) {
        unsigned char first = (unsigned char)t->vocab[i][0];
        t->bucket_next[i] = t->bucket_head[first];
        t->bucket_head[first] = i;
    }

    return t->vocab_size > 0 ? 0 : -1;
}

int tokenizer_load_gguf(Tokenizer* t, gguf_context* ctx) {
    memset(t, 0, sizeof(*t));
    t->type = TOKENIZER_BPE;
    
    char** tokens = NULL;
    uint64_t len = 0;
    if (gguf_get_val_str_array(ctx, "tokenizer.ggml.tokens", &tokens, &len) == 0 && tokens != NULL) {
        t->vocab_size = (int)len;
        t->vocab = tokens;
    } else {
        return -1;
    }
    
    char** merges = NULL;
    uint64_t merges_len = 0;
    if (gguf_get_val_str_array(ctx, "tokenizer.ggml.merges", &merges, &merges_len) == 0) {
        t->merges_size = (int)merges_len;
        t->merges = merges;
    }
    
    for (int i = 0; i < 256; i++) t->bucket_head[i] = -1;
    t->bucket_next = (int*)malloc(sizeof(int) * t->vocab_size);
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab[i] != NULL && t->vocab[i][0] != '\0') {
            unsigned char first = (unsigned char)t->vocab[i][0];
            t->bucket_next[i] = t->bucket_head[first];
            t->bucket_head[first] = i;
        } else {
            t->bucket_next[i] = -1;
        }
    }

    return t->vocab_size > 0 ? 0 : -1;
}

void tokenizer_free(Tokenizer* t) {
    if (t->bucket_next) {
        free(t->bucket_next);
    }
    memset(t, 0, sizeof(*t));
}

// Fast bucket-indexed greedy BPE approximation via Longest Prefix Match
int tokenizer_encode(const Tokenizer* t, const char* text, int* out_tokens, int max_tokens) {
    int n = 0;
    const char* p = text;
    while (*p != '\0' && n < max_tokens) {
        int best_len = 0;
        int best_id = -1;
        unsigned char first = (unsigned char)*p;
        size_t rem_len = strlen(p);
        
        for (int i = t->bucket_head[first]; i >= 0; i = t->bucket_next[i]) {
            size_t vlen = strlen(t->vocab[i]);
            if (vlen > 0 && vlen <= rem_len && (int)vlen > best_len) {
                if (strncmp(t->vocab[i], p, vlen) == 0) {
                    best_len = (int)vlen;
                    best_id = i;
                }
            }
        }
        if (best_id >= 0) {
            out_tokens[n++] = best_id;
            p += best_len;
        } else {
            // fallback byte
            out_tokens[n++] = 1;
            p++;
        }
    }
    return n;
}

void tokenizer_decode_append(const Tokenizer* t, int token, char* out_text, int out_cap) {
    if (token < 0 || token >= t->vocab_size || out_text == NULL || out_cap <= 0) return;
    const char* v = t->vocab[token];
    if (v == NULL) return;
    size_t vlen = strlen(v);
    
    // Ignore special byte-tokens that start with <0x and end with >
    if (vlen == 6 && v[0] == '<' && v[1] == '0' && v[2] == 'x' && v[5] == '>') {
        return; // Skip raw byte tokens in simplified decoder
    }
    
    // Skip special control tokens in output string
    if (strcmp(v, "<s>") == 0 || strcmp(v, "</s>") == 0 ||
        strcmp(v, "<|im_start|>") == 0 || strcmp(v, "<|im_end|>") == 0 ||
        strcmp(v, "<|endoftext|>") == 0 || strcmp(v, "<unk>") == 0) {
        return;
    }

    size_t cur = strlen(out_text);
    
    // Check if token begins with SentencePiece space token "\xe2\x96\x81" (Unicode U+2581)
    const char* src = v;
    while (*src != '\0') {
        if ((unsigned char)src[0] == 0xe2 && (unsigned char)src[1] == 0x96 && (unsigned char)src[2] == 0x81) {
            if (cur + 1 < (size_t)out_cap) {
                out_text[cur++] = ' ';
                out_text[cur] = '\0';
            }
            src += 3;
        } else {
            if (cur + 1 < (size_t)out_cap) {
                out_text[cur++] = *src;
                out_text[cur] = '\0';
            }
            src++;
        }
    }
}

int tokenizer_is_eos(const Tokenizer* t, int token) {
    if (token < 0 || token >= t->vocab_size || t->vocab == NULL) return 1;
    const char* v = t->vocab[token];
    if (v == NULL) return 1;
    if (strcmp(v, "</s>") == 0 ||
        strcmp(v, "<|im_end|>") == 0 ||
        strcmp(v, "<|endoftext|>") == 0 ||
        strcmp(v, "<eos>") == 0 ||
        strcmp(v, "<|end_of_text|>") == 0) {
        return 1;
    }
    return 0;
}
