CC := gcc
CFLAGS := -O3 -Wall -Wextra -std=c99 -fopenmp -Iinclude
LDFLAGS := -lm -fopenmp

ifeq ($(OS),Windows_NT)
	LDFLAGS += -lpsapi -lws2_32
	EXE := .exe
	LIB_SHARED := baremetal.dll
	LIB_STATIC := libbaremetal.a
	RM := del /Q /F
	RUN := .\\
else
	EXE :=
	LIB_SHARED := libbaremetal.so
	LIB_STATIC := libbaremetal.a
	RM := rm -f
	RUN := ./
endif

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
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)$(EXE) $(LDFLAGS)

$(TARGET_LAUNCHER)$(EXE): src/launcher.c
	$(CC) $(CFLAGS) src/launcher.c -o $(TARGET_LAUNCHER)$(EXE) $(LDFLAGS)

$(TARGET_TESTS)$(EXE): $(TEST_SRC)
	$(CC) $(CFLAGS) $(TEST_SRC) -o $(TARGET_TESTS)$(EXE) $(LDFLAGS)

$(LIB_SHARED): $(CORE_SRC)
	$(CC) $(CFLAGS) -shared -fPIC $(CORE_SRC) -o $(LIB_SHARED) $(LDFLAGS)

$(LIB_STATIC): $(CORE_SRC)
	$(CC) $(CFLAGS) -c $(CORE_SRC)
	ar rcs $(LIB_STATIC) *.o
	-$(RM) *.o

$(TARGET_EXAMPLE)$(EXE): examples/minimal_embed.c $(LIB_SHARED)
	$(CC) $(CFLAGS) examples/minimal_embed.c -L. -lbaremetal -o $(TARGET_EXAMPLE)$(EXE) $(LDFLAGS)

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
	$(RUN)$(TARGET)$(EXE) --model dummy.bin --steps 8 --temperature 0.7 --prompt-token 1 --backend avx2 --threads 4
	-$(RM) dummy.bin

benchmark: $(TARGET)$(EXE)
	python scripts/benchmark_suite.py

clean:
	-$(RM) $(TARGET)$(EXE) $(TARGET_LAUNCHER)$(EXE) $(TARGET_TESTS)$(EXE) $(TARGET_EXAMPLE)$(EXE) $(LIB_SHARED) $(LIB_STATIC) *.o dummy.bin test_dummy.bin quantr.cfgo
