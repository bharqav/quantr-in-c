#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <unistd.h>
#include <sys/statvfs.h>
#include <sys/sysinfo.h>
#endif

typedef struct {
    const char* key;
    const char* name;
    const char* tier;
    const char* size_str;
    const char* ram_str;
    const char* url;
    const char* filename;
} ModelCatalogEntry;

static const ModelCatalogEntry CATALOG[] = {
    {
        "smollm2-135m",
        "SmolLM2 135M Instruct",
        "Nano",
        "~85 MB",
        "<200 MB",
        "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf",
        "models/smollm2-135m-instruct-q4_k_m.gguf"
    },
    {
        "tinyllama-1.1b-q8",
        "TinyLlama 1.1B Chat (Q8_0)",
        "Compact",
        "~146 MB",
        "~450 MB",
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    },
    {
        "qwen2.5-0.5b",
        "Qwen 2.5 0.5B Instruct",
        "Balanced",
        "~350 MB",
        "~700 MB",
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    {
        "tinyllama-1.1b-q4",
        "TinyLlama 1.1B Chat (Q4_K_M)",
        "Compact",
        "~630 MB",
        "~800 MB",
        "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    {
        "qwen-moe-2.7b",
        "Qwen 1.5 MoE A2.7B Chat",
        "Performance",
        "~950 MB",
        "~1.5 GB",
        "https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/qwen1_5-moe-a2_7b-chat-q4_k_m.gguf",
        "models/qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
    }
};

static const int CATALOG_SIZE = (int)(sizeof(CATALOG) / sizeof(CATALOG[0]));
static const char* CONFIG_FILE = "quantr.cfg";

typedef struct {
    double total_ram_gb;
    double free_ram_gb;
    double free_disk_gb;
    int cpu_cores;
    int has_avx2;
    int has_avx512;
    int has_gpu;
    char gpu_name[128];
} HardwareInfo;

static int file_exists(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f) {
        fclose(f);
        return 1;
    }
    return 0;
}

static void ensure_models_dir(void) {
#ifdef _WIN32
    CreateDirectoryA("models", NULL);
#else
    mkdir("models", 0755);
#endif
}

static void probe_hardware(HardwareInfo* hw) {
    memset(hw, 0, sizeof(*hw));

#ifdef _WIN32
    /* RAM Status */
    MEMORYSTATUSEX mem;
    mem.dwLength = sizeof(mem);
    if (GlobalMemoryStatusEx(&mem)) {
        hw->total_ram_gb = (double)mem.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
        hw->free_ram_gb = (double)mem.ullAvailPhys / (1024.0 * 1024.0 * 1024.0);
    }

    /* Disk Space */
    ULARGE_INTEGER freeBytes, totalBytes, totalFreeBytes;
    if (GetDiskFreeSpaceExA(".", &freeBytes, &totalBytes, &totalFreeBytes)) {
        hw->free_disk_gb = (double)freeBytes.QuadPart / (1024.0 * 1024.0 * 1024.0);
    }

    /* CPU Cores */
    SYSTEM_INFO sys;
    GetSystemInfo(&sys);
    hw->cpu_cores = (int)sys.dwNumberOfProcessors;

    /* GPU Detection (NVIDIA / Vulkan) */
    HMODULE hCuda = LoadLibraryA("nvcuda.dll");
    if (hCuda) {
        hw->has_gpu = 1;
        strncpy(hw->gpu_name, "NVIDIA CUDA Acceleration Supported", sizeof(hw->gpu_name) - 1);
        FreeLibrary(hCuda);
    } else {
        HMODULE hVulkan = LoadLibraryA("vulkan-1.dll");
        if (hVulkan) {
            hw->has_gpu = 1;
            strncpy(hw->gpu_name, "Vulkan GPU Acceleration Supported", sizeof(hw->gpu_name) - 1);
            FreeLibrary(hVulkan);
        } else {
            strncpy(hw->gpu_name, "No Discrete GPU detected (CPU SIMD active)", sizeof(hw->gpu_name) - 1);
        }
    }
#else
    hw->cpu_cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    struct sysinfo si;
    if (sysinfo(&si) == 0) {
        hw->total_ram_gb = (double)si.totalram * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
        hw->free_ram_gb = (double)si.freeram * si.mem_unit / (1024.0 * 1024.0 * 1024.0);
    }
    struct statvfs st;
    if (statvfs(".", &st) == 0) {
        hw->free_disk_gb = (double)st.f_bavail * st.f_frsize / (1024.0 * 1024.0 * 1024.0);
    }
    strncpy(hw->gpu_name, "Standard Compute Device", sizeof(hw->gpu_name) - 1);
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    hw->has_avx2 = __builtin_cpu_supports("avx2");
    hw->has_avx512 = __builtin_cpu_supports("avx512f");
#else
    hw->has_avx2 = 1;
#endif
#endif
}

static int download_selected_model(const ModelCatalogEntry* entry) {
    printf("\n\033[1;36m[DOWNLOAD] Fetching %s (%s)...\033[0m\n", entry->name, entry->size_str);
    printf("\033[90mSource URL: %s\033[0m\n", entry->url);
    printf("\033[90mSaving to:  %s\033[0m\n\n", entry->filename);
    fflush(stdout);

    ensure_models_dir();

#ifdef _WIN32
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "powershell -Command \"[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; "
        "Write-Host 'Connecting to HuggingFace repository...'; "
        "$wc = New-Object System.Net.WebClient; "
        "$wc.DownloadFile('%s', '%s')\"",
        entry->url, entry->filename);
    
    int ret = system(cmd);
    if (ret != 0 || !file_exists(entry->filename)) {
        snprintf(cmd, sizeof(cmd), "curl -L -o \"%s\" \"%s\"", entry->filename, entry->url);
        ret = system(cmd);
    }
    return (ret == 0 && file_exists(entry->filename)) ? 0 : -1;
#else
    char cmd[2048];
    snprintf(cmd, sizeof(cmd), "curl -L -o \"%s\" \"%s\"", entry->filename, entry->url);
    int ret = system(cmd);
    return (ret == 0 && file_exists(entry->filename)) ? 0 : -1;
#endif
}

static void save_config(const char* model_path, int use_gpu, int threads) {
    FILE* f = fopen(CONFIG_FILE, "w");
    if (f) {
        fprintf(f, "model=%s\n", model_path);
        fprintf(f, "use_gpu=%d\n", use_gpu);
        fprintf(f, "threads=%d\n", threads);
        fclose(f);
    }
}

static int load_config(char* model_path, size_t max_len, int* use_gpu, int* threads) {
    FILE* f = fopen(CONFIG_FILE, "r");
    if (!f) return 0;
    char line[512];
    int loaded = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model=", 6) == 0) {
            char* p = line + 6;
            size_t l = strlen(p);
            while (l > 0 && (p[l - 1] == '\r' || p[l - 1] == '\n')) p[--l] = '\0';
            strncpy(model_path, p, max_len - 1);
            loaded = 1;
        } else if (strncmp(line, "use_gpu=", 8) == 0) {
            if (use_gpu) *use_gpu = atoi(line + 8);
        } else if (strncmp(line, "threads=", 8) == 0) {
            if (threads) *threads = atoi(line + 8);
        }
    }
    fclose(f);
    return loaded && file_exists(model_path);
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleTitleA("⚡ Quantr Baremetal AI Setup & Launcher");
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
#endif

    int force_setup = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--setup") == 0 || strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "--menu") == 0) {
            force_setup = 1;
        }
    }

    char chosen_model[512] = {0};
    int use_gpu = 0;
    int threads = 8;

    /* Check if previous configuration exists and model is present */
    if (!force_setup && load_config(chosen_model, sizeof(chosen_model), &use_gpu, &threads)) {
        if (file_exists(chosen_model)) {
            char exec_cmd[2048];
            snprintf(exec_cmd, sizeof(exec_cmd), "inference.exe --model \"%s\" --vm --threads %d %s",
                     chosen_model, threads, use_gpu ? "--gpu" : "");
            for (int i = 1; i < argc; i++) {
                strncat(exec_cmd, " ", sizeof(exec_cmd) - strlen(exec_cmd) - 1);
                strncat(exec_cmd, argv[i], sizeof(exec_cmd) - strlen(exec_cmd) - 1);
            }
            return system(exec_cmd);
        }
    }

    /* Run Interactive Hardware Diagnostic & Setup Wizard */
    HardwareInfo hw;
    probe_hardware(&hw);

    printf("\033[1;36m");
    printf("╔══════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                 ⚡ QUANTR BAREMETAL HARDWARE SETUP WIZARD                    ║\n");
    printf("║                 Zero-Dependency Contextual AI Installer                      ║\n");
    printf("╚══════════════════════════════════════════════════════════════════════════════╝\n");
    printf("\033[0m\n");

    printf("\033[90m┌─ \033[1;37mSystem Hardware Diagnostic\033[0m\033[90m ──────────────────────────────────────────────┐\033[0m\n");
    printf("\033[90m│\033[0m  \033[1;33mPhysical RAM:\033[0m      %.1f GB Total  (%.1f GB Available)                     \033[90m│\033[0m\n",
           hw.total_ram_gb, hw.free_ram_gb);
    printf("\033[90m│\033[0m  \033[1;33mDisk Space:\033[0m        %.1f GB Free Storage                                    \033[90m│\033[0m\n",
           hw.free_disk_gb);
    printf("\033[90m│\033[0m  \033[1;33mCPU Processor:\033[0m     %d Threads | SIMD: %s%s                                \033[90m│\033[0m\n",
           hw.cpu_cores, hw.has_avx512 ? "AVX-512 " : "", hw.has_avx2 ? "AVX2+FMA" : "Scalar");
    printf("\033[90m│\033[0m  \033[1;33mGPU Hardware:\033[0m      %-55s\033[90m│\033[0m\n", hw.gpu_name);
    printf("\033[90m└──────────────────────────────────────────────────────────────────────────────┘\033[0m\n\n");

    printf("\033[1;32m[Step 1/2]\033[0m Select Model Tier based on your Storage & RAM preferences:\n\n");

    for (int i = 0; i < CATALOG_SIZE; i++) {
        int present = file_exists(CATALOG[i].filename);
        printf("  \033[1;36m[%d]\033[0m \033[1m%-28s\033[0m Tier: \033[33m%-11s\033[0m Size: %-8s RAM: %-8s %s\n",
               i + 1,
               CATALOG[i].name,
               CATALOG[i].tier,
               CATALOG[i].size_str,
               CATALOG[i].ram_str,
               present ? "\033[32m[Installed]\033[0m" : "");
    }
    printf("  \033[1;36m[%d]\033[0m \033[1mCustom GGUF Model\033[0m            (Specify local file path)\n", CATALOG_SIZE + 1);

    printf("\nEnter choice [1-%d] (Default is 1): ", CATALOG_SIZE + 1);
    fflush(stdout);

    char line[256];
    int choice = 1;
    if (fgets(line, sizeof(line), stdin) != NULL) {
        int v = atoi(line);
        if (v >= 1 && v <= CATALOG_SIZE + 1) {
            choice = v;
        }
    }

    if (choice == CATALOG_SIZE + 1) {
        printf("\nEnter local path to .gguf model: ");
        fflush(stdout);
        if (fgets(chosen_model, sizeof(chosen_model), stdin) != NULL) {
            size_t l = strlen(chosen_model);
            while (l > 0 && (chosen_model[l - 1] == '\r' || chosen_model[l - 1] == '\n' || chosen_model[l - 1] == ' ')) {
                chosen_model[--l] = '\0';
            }
        }
    } else {
        const ModelCatalogEntry* selected = &CATALOG[choice - 1];
        strncpy(chosen_model, selected->filename, sizeof(chosen_model) - 1);
        if (!file_exists(selected->filename)) {
            if (download_selected_model(selected) != 0) {
                fprintf(stderr, "\033[31m[ERROR] Failed to download model. Please check internet connectivity.\033[0m\n");
                printf("Press Enter to exit...");
                getchar();
                return 1;
            }
        }
    }

    /* Compute Target & Offloading Choice */
    printf("\n\033[1;32m[Step 2/2]\033[0m Select Compute Target & Offloading:\n");
    printf("  \033[1;36m[1]\033[0m \033[1mPure CPU Baremetal (AVX2/AVX-512)\033[0m  - Zero external dependencies\n");
    printf("  \033[1;36m[2]\033[0m \033[1mGPU Offload (CUDA / Vulkan)\033[0m        - Offload matrix operations to GPU\n");
    printf("\nEnter choice [1-2] (Default is 1): ");
    fflush(stdout);

    int comp_choice = 1;
    if (fgets(line, sizeof(line), stdin) != NULL) {
        int v = atoi(line);
        if (v == 2) comp_choice = 2;
    }
    use_gpu = (comp_choice == 2);
    threads = hw.cpu_cores > 0 ? hw.cpu_cores : 8;

    /* Save preferences */
    save_config(chosen_model, use_gpu, threads);

    printf("\n\033[32m[Ready]\033[0m Launching Quantr Virtual Machine Terminal...\n\n");

    char exec_cmd[2048];
    snprintf(exec_cmd, sizeof(exec_cmd), "inference.exe --model \"%s\" --vm --threads %d %s",
             chosen_model, threads, use_gpu ? "--gpu" : "");
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--setup") != 0 && strcmp(argv[i], "--config") != 0) {
            strncat(exec_cmd, " ", sizeof(exec_cmd) - strlen(exec_cmd) - 1);
            strncat(exec_cmd, argv[i], sizeof(exec_cmd) - strlen(exec_cmd) - 1);
        }
    }

    return system(exec_cmd);
}
