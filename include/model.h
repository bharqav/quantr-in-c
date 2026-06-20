#ifndef MODEL_H
#define MODEL_H

#include "types.h"

size_t checked_mul(size_t a, size_t b, const char* label);
size_t config_weight_floats(const Config* p);
size_t run_state_floats(const Config* p);

void malloc_run_state(RunState* s, const Config* p);
void malloc_run_state_kv(RunState* s, const Config* p, int kv_type);
void free_run_state(RunState* s);

void load_transformer(Transformer* t, const char* model_path);
void free_transformer(Transformer* t);
void write_dummy_model(const char* path);

#endif
