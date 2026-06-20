#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "gguf.h"

typedef enum {
    TOKENIZER_NONE = 0,
    TOKENIZER_BPE = 1,
    TOKENIZER_SENTENCEPIECE = 2
} TokenizerType;

typedef struct {
    TokenizerType type;
    int vocab_size;
    char** vocab;
    int merges_size;
    char** merges;
    int bucket_head[256];
    int* bucket_next;
} Tokenizer;

int tokenizer_load(Tokenizer* t, const char* path, TokenizerType type_hint);
int tokenizer_load_gguf(Tokenizer* t, gguf_context* ctx);
void tokenizer_free(Tokenizer* t);
int tokenizer_encode(const Tokenizer* t, const char* text, int* out_tokens, int max_tokens);
void tokenizer_decode_append(const Tokenizer* t, int token, char* out_text, int out_cap);
int tokenizer_is_eos(const Tokenizer* t, int token);

#endif
