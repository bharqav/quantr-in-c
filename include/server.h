/**
 * @file server.h
 * @brief Embedded OpenAI-compatible HTTP server for Baremetal AI Engine.
 *
 * Exposes POST /v1/chat/completions with SSE streaming and GET /v1/models.
 */

#ifndef SERVER_H
#define SERVER_H

#include "runtime.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* host;              /**< Bind address (e.g. "0.0.0.0") */
    int port;                      /**< Listen port (e.g. 8080) */
    RuntimeContext* rt;            /**< Initialized runtime context */
    Transformer* transformer;      /**< Loaded transformer model */
    Tokenizer* tokenizer;          /**< Loaded tokenizer (may be NULL) */
    RuntimeOptions* options;       /**< Runtime options for sampling */
} ServerConfig;

/**
 * @brief Start the HTTP server (blocking). Returns 0 on clean shutdown, -1 on error.
 */
int server_start(const ServerConfig* config);

#ifdef __cplusplus
}
#endif

#endif /* SERVER_H */
