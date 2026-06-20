/**
 * @file server.c
 * @brief Zero-dependency OpenAI-compatible HTTP/1.1 server with SSE streaming.
 *
 * Endpoints:
 *   POST /v1/chat/completions  - Generate with SSE streaming
 *   GET  /v1/models            - List available models
 *   GET  /health               - Health check
 *
 * Cross-platform: Winsock2 on Windows, POSIX sockets on Linux/macOS.
 */

#include "server.h"
#include "sampling.h"
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define CLOSE_SOCKET closesocket
#define SOCKET_ERROR_VAL INVALID_SOCKET
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
typedef int socket_t;
#define CLOSE_SOCKET close
#define SOCKET_ERROR_VAL (-1)
#define INVALID_SOCKET (-1)
#endif

static volatile int g_running = 1;

static void signal_handler(int sig) {
    (void)sig;
    g_running = 0;
}

/* ─── Minimal JSON helpers ─── */

static const char* json_find_key(const char* json, const char* key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    return p;
}

static int json_get_int(const char* json, const char* key, int def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    return atoi(p);
}

static float json_get_float(const char* json, const char* key, float def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    return strtof(p, NULL);
}

static int json_get_bool(const char* json, const char* key, int def) {
    const char* p = json_find_key(json, key);
    if (!p) return def;
    if (strncmp(p, "true", 4) == 0) return 1;
    if (strncmp(p, "false", 5) == 0) return 0;
    return def;
}

/* Extract the last "content" string value from the messages array */
static int json_get_last_content(const char* json, char* out, int out_cap) {
    const char* last_content = NULL;
    const char* search = json;
    while ((search = strstr(search, "\"content\"")) != NULL) {
        last_content = search;
        search += 9;
    }
    if (!last_content) return -1;

    const char* p = last_content + 9;
    while (*p == ' ' || *p == ':' || *p == '\t') p++;
    if (*p != '"') return -1;
    p++;

    int i = 0;
    while (*p && *p != '"' && i < out_cap - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
                case 'n': out[i++] = '\n'; break;
                case 't': out[i++] = '\t'; break;
                case '"': out[i++] = '"'; break;
                case '\\': out[i++] = '\\'; break;
                default: out[i++] = *p; break;
            }
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return i;
}

/* ─── HTTP helpers ─── */

static int send_all(socket_t sock, const char* data, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(sock, data + sent, len - sent, 0);
        if (n <= 0) return -1;
        sent += n;
    }
    return sent;
}

static void send_response(socket_t sock, int status, const char* content_type,
                          const char* body, int body_len) {
    char header[512];
    const char* status_text = status == 200 ? "OK" : status == 404 ? "Not Found" : "Bad Request";
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, status_text, content_type, body_len);
    send_all(sock, header, hlen);
    if (body_len > 0) {
        send_all(sock, body, body_len);
    }
}

static void send_sse_start(socket_t sock) {
    const char* header =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "\r\n";
    send_all(sock, header, (int)strlen(header));
}

static void send_sse_token(socket_t sock, const char* token_str, const char* model_name) {
    /* Escape token string for JSON */
    char escaped[512];
    int ei = 0;
    for (int i = 0; token_str[i] && ei < 500; i++) {
        switch (token_str[i]) {
            case '"': escaped[ei++] = '\\'; escaped[ei++] = '"'; break;
            case '\\': escaped[ei++] = '\\'; escaped[ei++] = '\\'; break;
            case '\n': escaped[ei++] = '\\'; escaped[ei++] = 'n'; break;
            case '\r': escaped[ei++] = '\\'; escaped[ei++] = 'r'; break;
            case '\t': escaped[ei++] = '\\'; escaped[ei++] = 't'; break;
            default: escaped[ei++] = token_str[i]; break;
        }
    }
    escaped[ei] = '\0';

    char buf[1024];
    int len = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-baremetal\",\"object\":\"chat.completion.chunk\","
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
        "\"finish_reason\":null}]}\n\n",
        model_name, escaped);
    send_all(sock, buf, len);
}

static void send_sse_done(socket_t sock, const char* model_name) {
    char buf[512];
    int len = snprintf(buf, sizeof(buf),
        "data: {\"id\":\"chatcmpl-baremetal\",\"object\":\"chat.completion.chunk\","
        "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{},"
        "\"finish_reason\":\"stop\"}]}\n\n"
        "data: [DONE]\n\n",
        model_name);
    send_all(sock, buf, len);
}

/* ─── Request handling ─── */

static void handle_health(socket_t sock) {
    const char* body = "{\"status\":\"ok\"}";
    send_response(sock, 200, "application/json", body, (int)strlen(body));
}

static void handle_models(socket_t sock) {
    const char* body =
        "{\"object\":\"list\",\"data\":["
        "{\"id\":\"baremetal-default\",\"object\":\"model\","
        "\"owned_by\":\"baremetal\",\"permission\":[]}"
        "]}";
    send_response(sock, 200, "application/json", body, (int)strlen(body));
}

static void handle_chat_completions(socket_t sock, const char* body,
                                     const ServerConfig* cfg) {
    /* Parse request */
    char prompt[4096] = {0};
    if (json_get_last_content(body, prompt, sizeof(prompt)) < 0) {
        const char* err = "{\"error\":{\"message\":\"No content in messages\",\"type\":\"invalid_request_error\"}}";
        send_response(sock, 400, "application/json", err, (int)strlen(err));
        return;
    }

    int max_tokens = json_get_int(body, "max_tokens", 128);
    float temperature = json_get_float(body, "temperature", cfg->options->temperature);
    int stream = json_get_bool(body, "stream", 0);

    if (max_tokens > 2048) max_tokens = 2048;
    if (max_tokens < 1) max_tokens = 1;

    /* Setup sampling */
    RuntimeOptions gen_opt = *cfg->options;
    gen_opt.temperature = temperature;
    if (strstr(body, "\"json_object\"") != NULL || strstr(body, "\"response_format\"") != NULL) {
        gen_opt.json_mode = 1;
    }

    SamplingState samp;
    sampling_state_init(&samp, 128);

    /* Tokenize and prefill prompt */
    int prompt_tokens[1024];
    int n_prompt = 0;
    int token = 1;
    int pos = 0;

    if (cfg->tokenizer) {
        n_prompt = tokenizer_encode(cfg->tokenizer, prompt, prompt_tokens, 1024);
        prefill_runtime(cfg->rt, cfg->transformer, prompt_tokens, n_prompt);
        for (int i = 0; i < n_prompt; i++) {
            sampling_state_push(&samp, prompt_tokens[i]);
        }
        if (n_prompt > 0) {
            token = prompt_tokens[n_prompt - 1];
            pos = n_prompt;
        }
    }

    if (stream) {
        /* SSE streaming mode */
        send_sse_start(sock);

        for (int i = 0; i < max_tokens; i++) {
            float* logits = forward_runtime(cfg->rt, cfg->transformer, token, pos % cfg->transformer->config.seq_len);
            int next = sample_next(logits, cfg->transformer->config.vocab_size, &gen_opt, &samp);
            sampling_state_push(&samp, next);
            token = next;
            pos++;

            if (cfg->tokenizer) {
                char out[256] = {0};
                tokenizer_decode_append(cfg->tokenizer, next, out, (int)sizeof(out));
                send_sse_token(sock, out, "baremetal-default");
            } else {
                char out[32];
                snprintf(out, sizeof(out), "%d ", next);
                send_sse_token(sock, out, "baremetal-default");
            }
        }
        send_sse_done(sock, "baremetal-default");
    } else {
        /* Non-streaming: collect full response */
        char response[65536] = {0};
        int resp_len = 0;

        for (int i = 0; i < max_tokens; i++) {
            float* logits = forward_runtime(cfg->rt, cfg->transformer, token, pos % cfg->transformer->config.seq_len);
            int next = sample_next(logits, cfg->transformer->config.vocab_size, &gen_opt, &samp);
            sampling_state_push(&samp, next);
            token = next;
            pos++;

            if (cfg->tokenizer) {
                char out[256] = {0};
                tokenizer_decode_append(cfg->tokenizer, next, out, (int)sizeof(out));
                int out_len = (int)strlen(out);
                if (resp_len + out_len < (int)sizeof(response) - 1) {
                    memcpy(response + resp_len, out, out_len);
                    resp_len += out_len;
                }
            }
        }
        response[resp_len] = '\0';

        /* Escape response for JSON */
        char escaped_resp[65536];
        int ei = 0;
        for (int i = 0; i < resp_len && ei < (int)sizeof(escaped_resp) - 2; i++) {
            switch (response[i]) {
                case '"': escaped_resp[ei++] = '\\'; escaped_resp[ei++] = '"'; break;
                case '\\': escaped_resp[ei++] = '\\'; escaped_resp[ei++] = '\\'; break;
                case '\n': escaped_resp[ei++] = '\\'; escaped_resp[ei++] = 'n'; break;
                case '\r': escaped_resp[ei++] = '\\'; escaped_resp[ei++] = 'r'; break;
                default: escaped_resp[ei++] = response[i]; break;
            }
        }
        escaped_resp[ei] = '\0';

        char json_resp[131072];
        int json_len = snprintf(json_resp, sizeof(json_resp),
            "{\"id\":\"chatcmpl-baremetal\",\"object\":\"chat.completion\","
            "\"model\":\"baremetal-default\","
            "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
            "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
            escaped_resp, n_prompt, max_tokens, n_prompt + max_tokens);
        send_response(sock, 200, "application/json", json_resp, json_len);
    }

    sampling_state_free(&samp);
}

static void handle_request(socket_t client, const ServerConfig* cfg) {
    char buf[65536] = {0};
    int total = 0;

    /* Read full request */
    while (total < (int)sizeof(buf) - 1) {
        int n = recv(client, buf + total, (int)sizeof(buf) - 1 - total, 0);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';
        /* Check if we have the full request (double CRLF found) */
        if (strstr(buf, "\r\n\r\n")) {
            /* For POST, check Content-Length */
            const char* cl = strstr(buf, "Content-Length:");
            if (cl) {
                int content_len = atoi(cl + 15);
                const char* body_start = strstr(buf, "\r\n\r\n");
                if (body_start) {
                    int header_len = (int)(body_start - buf) + 4;
                    int body_received = total - header_len;
                    if (body_received >= content_len) break;
                }
            } else {
                break; /* GET request or no body */
            }
        }
    }

    if (total <= 0) return;

    /* Parse method and path */
    char method[16] = {0};
    char path[256] = {0};
    sscanf(buf, "%15s %255s", method, path);

    /* CORS preflight */
    if (strcmp(method, "OPTIONS") == 0) {
        send_response(client, 200, "text/plain", "", 0);
        return;
    }

    /* Route request */
    if (strcmp(path, "/") == 0 || strcmp(path, "/health") == 0) {
        handle_health(client);
    } else if (strcmp(path, "/v1/models") == 0) {
        handle_models(client);
    } else if (strcmp(path, "/v1/chat/completions") == 0 && strcmp(method, "POST") == 0) {
        const char* body = strstr(buf, "\r\n\r\n");
        if (body) body += 4;
        handle_chat_completions(client, body ? body : "{}", cfg);
    } else {
        const char* err = "{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}";
        send_response(client, 404, "application/json", err, (int)strlen(err));
    }
}

/* ─── Server main loop ─── */

int server_start(const ServerConfig* config) {
    if (!config) return -1;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "[SERVER] WSAStartup failed\n");
        return -1;
    }
#endif

    signal(SIGINT, signal_handler);
#ifdef SIGTERM
    signal(SIGTERM, signal_handler);
#endif

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET) {
        fprintf(stderr, "[SERVER] socket() failed\n");
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    /* Allow port reuse */
    int opt_val = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt_val, sizeof(opt_val));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((unsigned short)config->port);
    addr.sin_addr.s_addr = inet_addr(config->host);

    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[SERVER] bind() failed on %s:%d\n", config->host, config->port);
        CLOSE_SOCKET(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    if (listen(server_fd, 8) < 0) {
        fprintf(stderr, "[SERVER] listen() failed\n");
        CLOSE_SOCKET(server_fd);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    printf("[SERVER] Baremetal AI HTTP Server listening on http://%s:%d\n", config->host, config->port);
    printf("[SERVER] Endpoints:\n");
    printf("  POST /v1/chat/completions  (OpenAI-compatible, SSE streaming)\n");
    printf("  GET  /v1/models            (Model listing)\n");
    printf("  GET  /health               (Health check)\n");
    printf("[SERVER] Press Ctrl+C to stop.\n\n");
    fflush(stdout);

    while (g_running) {
        struct sockaddr_in client_addr;
#ifdef _WIN32
        int client_len = sizeof(client_addr);
#else
        socklen_t client_len = sizeof(client_addr);
#endif
        socket_t client = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client == INVALID_SOCKET) {
            if (!g_running) break;
            continue;
        }

        handle_request(client, config);
        CLOSE_SOCKET(client);
    }

    printf("\n[SERVER] Shutting down...\n");
    CLOSE_SOCKET(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
