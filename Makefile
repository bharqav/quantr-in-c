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
    SHARED_FLAG := -shared
    RPATH_FLAG :=
    RM := rm -f
    RUN := ./
else ifeq ($(UNAME_S),Darwin)
    EXE :=
    LIB_SHARED := libbaremetal.dylib
    LIB_STATIC := libbaremetal.a
    RM := rm -f
    RUN := ./
    SHARED_FLAG := -dynamiclib
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
    SHARED_FLAG := -shared
    OPENMP_FLAG ?= -fopenmp
    RPATH_FLAG := -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,.
endif

ALL_CFLAGS := $(CFLAGS) $(INCLUDES) $(OPENMP_FLAG) -fPIC
ALL_LDFLAGS := $(LDFLAGS) $(OPENMP_FLAG)

TARGET := inference
TARGET_LAUNCHER := quantr
TARGET_TESTS := run_tests
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

CORE_OBJS := $(CORE_SRC:.c=.o)

.PHONY: all clean smoke test fuzz benchmark run setup-model lib example amalgamate install

all: $(TARGET)$(EXE) $(TARGET_LAUNCHER)$(EXE) $(TARGET_TESTS)$(EXE) $(LIB_SHARED) $(LIB_STATIC) $(TARGET_EXAMPLE)$(EXE)

amalgamate:
	python scripts/amalgamate.py

src/%.o: src/%.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

tests/%.o: tests/%.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

examples/%.o: examples/%.c
	$(CC) $(ALL_CFLAGS) -c $< -o $@

$(TARGET)$(EXE): src/main.o $(CORE_OBJS)
	$(CC) $(ALL_CFLAGS) src/main.o $(CORE_OBJS) -o $(TARGET)$(EXE) $(ALL_LDFLAGS)

$(TARGET_LAUNCHER)$(EXE): src/launcher.o
	$(CC) $(ALL_CFLAGS) src/launcher.o -o $(TARGET_LAUNCHER)$(EXE) $(ALL_LDFLAGS)

$(TARGET_TESTS)$(EXE): tests/tests.o $(CORE_OBJS)
	$(CC) $(ALL_CFLAGS) tests/tests.o $(CORE_OBJS) -o $(TARGET_TESTS)$(EXE) $(ALL_LDFLAGS)

$(LIB_SHARED): $(CORE_OBJS)
	$(CC) $(ALL_CFLAGS) $(SHARED_FLAG) $(CORE_OBJS) -o $(LIB_SHARED) $(ALL_LDFLAGS)

$(LIB_STATIC): $(CORE_OBJS)
	ar rcs $(LIB_STATIC) $(CORE_OBJS)

$(TARGET_EXAMPLE)$(EXE): examples/minimal_embed.o $(LIB_STATIC)
	$(CC) $(ALL_CFLAGS) examples/minimal_embed.o $(LIB_STATIC) -o $(TARGET_EXAMPLE)$(EXE) $(ALL_LDFLAGS)

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
	-$(RM) $(TARGET)$(EXE) $(TARGET_LAUNCHER)$(EXE) $(TARGET_TESTS)$(EXE) $(TARGET_EXAMPLE)$(EXE) $(LIB_SHARED) $(LIB_STATIC) src/*.o tests/*.o examples/*.o *.o dummy.bin test_dummy.bin quantr.cfgo
