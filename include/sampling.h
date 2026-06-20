#ifndef SAMPLING_H
#define SAMPLING_H

#include "types.h"

typedef struct {
    float* recent_tokens;
    int recent_capacity;
    int recent_count;
} SamplingState;

typedef struct {
    int depth;            /**< Object {} nesting depth */
    int array_depth;      /**< Array [] nesting depth */
    int in_string;        /**< Inside string literal */
    int escape;           /**< Character escaping active */
    int token_count;      /**< Tokens emitted in JSON generation */
    int complete;         /**< Whether top-level JSON is fully closed */
} JsonGrammarState;

void json_grammar_init(JsonGrammarState* g);
void json_grammar_update(JsonGrammarState* g, const char* token_piece);
void json_grammar_filter_logits(float* logits, int vocab_size, const JsonGrammarState* g);

void sampling_state_init(SamplingState* s, int capacity);
void sampling_state_free(SamplingState* s);
void sampling_state_push(SamplingState* s, int token);

int sample_next(float* logits, int vocab_size, const RuntimeOptions* opt, SamplingState* state);

#endif
