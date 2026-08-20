CC ?= gcc
INCLUDES := -Iinclude
CFLAGS ?= -O3 -Wall -Wextra -std=c99
LDFLAGS ?= -lm

UNAME_S := $(shell uname -s 2>/dev/null || echo Windows_NT)

ifeq ($(OS),Windows_NT)
    LDFLAGS += -lpsapi -lws2_32
    EXE := .exe
    LIB_SHARED := baremetal.dll
    LIB_STATIC := libbaremetal.a
    OPENMP_FLAG ?= -fopenmp
    RPATH_FLAG :=
    ifeq ($(findstring sh,$(SHELL)),sh)
        RM := rm -f
    else
        RM := del /Q /F
    endif
    RUN := ./
else ifeq ($(UNAME_S),Darwin)
    EXE :=
    LIB_SHARED := libbaremetal.dylib
    LIB_STATIC := libbaremetal.a
    RM := rm -f
    RUN := ./
    RPATH_FLAG := -Wl,-rpath,@loader_path -Wl,-rpath,.
    ifneq ($(wildcard /opt/homebrew/opt/libomp/include),)
        INCLUDES += -I/opt/homebrew/opt/libomp/include
        LDFLAGS += -L/opt/homebrew/opt/libomp/lib -lomp
        OPENMP_FLAG ?= -Xpreprocessor -fopenmp
    else ifneq ($(wildcard /usr/local/opt/libomp/include),)
        INCLUDES += -I/usr/local/opt/libomp/include
        LDFLAGS += -L/usr/local/opt/libomp/lib -lomp
        OPENMP_FLAG ?= -Xpreprocessor -fopenmp
    else
        OPENMP_FLAG ?=
    endif
else
    EXE :=
    LIB_SHARED := libbaremetal.so
    LIB_STATIC := libbaremetal.a
    RM := rm -f
    RUN := ./
    OPENMP_FLAG ?= -fopenmp
    RPATH_FLAG := -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,.
endif

ALL_CFLAGS := $(CFLAGS) $(INCLUDES) $(OPENMP_FLAG)
ALL_LDFLAGS := $(LDFLAGS) $(OPENMP_FLAG)

TARGET := inference
TARGET_LAUNCHER := quantr
TARGET_TESTS := tests
TARGET_EXAMPLE := minimal_embed

CORE_SRC := src/model.c \
            src/kernels.c \
            src/runtime.c \
            src/sampling.c \
            src/quant.c \
            src/tokenizer.c \
            src/context_window.c \
            src/vm_terminal.c \
            src/benchmark.c \
            src/threadpool.c \
            src/gguf.c \
            src/baremetal.c \
            src/server.c \
            src/speculative.c

SRC := src/main.c $(CORE_SRC)
TEST_SRC := tests/tests.c $(CORE_SRC)

.PHONY: all clean smoke test fuzz benchmark run setup-model lib example amalgamate install

all: $(TARGET)$(EXE) $(TARGET_LAUNCHER)$(EXE) $(TARGET_TESTS)$(EXE) $(LIB_SHARED) $(LIB_STATIC) $(TARGET_EXAMPLE)$(EXE)

amalgamate:
	python scripts/amalgamate.py

$(TARGET)$(EXE): $(SRC)
	$(CC) $(ALL_CFLAGS) $(SRC) -o $(TARGET)$(EXE) $(ALL_LDFLAGS)

$(TARGET_LAUNCHER)$(EXE): src/launcher.c
	$(CC) $(ALL_CFLAGS) src/launcher.c -o $(TARGET_LAUNCHER)$(EXE) $(ALL_LDFLAGS)

$(TARGET_TESTS)$(EXE): $(TEST_SRC)
	$(CC) $(ALL_CFLAGS) $(TEST_SRC) -o $(TARGET_TESTS)$(EXE) $(ALL_LDFLAGS)

$(LIB_SHARED): $(CORE_SRC)
	$(CC) $(ALL_CFLAGS) -shared -fPIC $(CORE_SRC) -o $(LIB_SHARED) $(ALL_LDFLAGS)

$(LIB_STATIC): $(CORE_SRC)
	$(CC) $(ALL_CFLAGS) -c $(CORE_SRC)
	ar rcs $(LIB_STATIC) *.o
	-$(RM) *.o

$(TARGET_EXAMPLE)$(EXE): examples/minimal_embed.c $(LIB_SHARED)
	$(CC) $(ALL_CFLAGS) examples/minimal_embed.c -L. -lbaremetal $(RPATH_FLAG) -o $(TARGET_EXAMPLE)$(EXE) $(ALL_LDFLAGS)

lib: $(LIB_SHARED) $(LIB_STATIC)

example: $(TARGET_EXAMPLE)$(EXE)

test: $(TARGET_TESTS)$(EXE)
	$(RUN)$(TARGET_TESTS)$(EXE)

cross-verify: $(TARGET_TESTS)$(EXE) $(TARGET)$(EXE)
	$(RUN)$(TARGET_TESTS)$(EXE)
	python scripts/cross_verify.py

setup-model:
	python scripts/setup_model.py

run: $(TARGET)$(EXE)
	$(RUN)$(TARGET)$(EXE)

smoke: $(TARGET)$(EXE)
	$(RUN)$(TARGET)$(EXE) --init-dummy dummy.bin
	$(RUN)$(TARGET)$(EXE) --model dummy.bin --steps 8 --temperature 0.7 --prompt-token 1 --backend avx2 --threads 4 --prompt "hello"
	-$(RM) dummy.bin

benchmark: $(TARGET)$(EXE)
	python scripts/benchmark_suite.py

clean:
	-$(RM) $(TARGET)$(EXE) $(TARGET_LAUNCHER)$(EXE) $(TARGET_TESTS)$(EXE) $(TARGET_EXAMPLE)$(EXE) $(LIB_SHARED) $(LIB_STATIC) *.o dummy.bin test_dummy.bin quantr.cfgo
