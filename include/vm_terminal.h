#ifndef VM_TERMINAL_H
#define VM_TERMINAL_H

#include "model.h"
#include "runtime.h"
#include "tokenizer.h"
#include "context_window.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    Transformer* transformer;
    RuntimeContext* runtime;
    Tokenizer* tokenizer;
    const char* model_path;
    int max_steps_per_turn;
    int system_prompt_enabled;
    const char* system_prompt;
} VMTerminalConfig;

/* Initialize and run the interactive Quantr Virtual Machine Terminal */
int vm_terminal_run(VMTerminalConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* VM_TERMINAL_H */
