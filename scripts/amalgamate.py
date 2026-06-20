#!/usr/bin/env python3
"""
Baremetal AI Inference Engine - Single-File Amalgamation Generator.

Combines all modular headers and C source files into a single, zero-dependency
SQLite-style amalgamation (dist/baremetal.h and dist/baremetal.c).

Usage:
    python scripts/amalgamate.py
"""

import os
import re

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIST_DIR = os.path.join(BASE_DIR, "dist")

HEADERS = [
    "include/types.h",
    "include/gguf.h",
    "include/quant.h",
    "include/kernels.h",
    "include/model.h",
    "include/threadpool.h",
    "include/sampling.h",
    "include/tokenizer.h",
    "include/runtime.h",
    "include/benchmark.h",
    "include/server.h",
    "include/speculative.h",
    "include/baremetal.h"
]

SOURCES = [
    "src/quant.c",
    "src/kernels.c",
    "src/threadpool.c",
    "src/sampling.c",
    "src/tokenizer.c",
    "src/gguf.c",
    "src/model.c",
    "src/runtime.c",
    "src/benchmark.c",
    "src/server.c",
    "src/speculative.c",
    "src/baremetal.c"
]

def clean_content(filepath, is_header=False):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    out = []
    for line in lines:
        # Strip local header includes (e.g. #include "types.h")
        if re.match(r'^\s*#\s*include\s*["<](types|gguf|quant|kernels|model|threadpool|sampling|tokenizer|runtime|benchmark|server|speculative|baremetal)\.h[">]', line):
            continue
        # Strip include guards from intermediate headers if needed
        out.append(line)
    return "".join(out)

def amalgamate():
    os.makedirs(DIST_DIR, exist_ok=True)
    out_h_path = os.path.join(DIST_DIR, "baremetal.h")
    out_c_path = os.path.join(DIST_DIR, "baremetal.c")

    print(f"Generating Amalgamated Header: {out_h_path}")
    with open(out_h_path, "w", encoding="utf-8") as out_h:
        out_h.write("/**\n")
        out_h.write(" * @file baremetal.h\n")
        out_h.write(" * @brief Amalgamated single-header for Baremetal AI Inference Engine.\n")
        out_h.write(" * Zero external dependencies. C99 Standard.\n")
        out_h.write(" */\n\n")
        out_h.write("#ifndef BAREMETAL_AMALGAMATION_H\n")
        out_h.write("#define BAREMETAL_AMALGAMATION_H\n\n")
        out_h.write("#include <stddef.h>\n")
        out_h.write("#include <stdint.h>\n")
        out_h.write("#include <stdio.h>\n")
        out_h.write("#include <stdlib.h>\n")
        out_h.write("#include <string.h>\n")
        out_h.write("#include <math.h>\n\n")

        for h in HEADERS:
            p = os.path.join(BASE_DIR, h)
            out_h.write(f"\n/* ─── Header: {h} ─── */\n")
            out_h.write(clean_content(p, is_header=True))

        out_h.write("\n#endif /* BAREMETAL_AMALGAMATION_H */\n")

    print(f"Generating Amalgamated Source: {out_c_path}")
    with open(out_c_path, "w", encoding="utf-8") as out_c:
        out_c.write("/**\n")
        out_c.write(" * @file baremetal.c\n")
        out_c.write(" * @brief Amalgamated single-file implementation for Baremetal AI Inference Engine.\n")
        out_c.write(" * Compile with: gcc baremetal.c -O3 -fopenmp -lm\n")
        out_c.write(" */\n\n")
        out_c.write('#include "baremetal.h"\n\n')

        for s in SOURCES:
            p = os.path.join(BASE_DIR, s)
            out_c.write(f"\n/* ─── Source: {s} ─── */\n")
            out_c.write(clean_content(p, is_header=False))

    print(" Amalgamation complete! Outputs saved in dist/")

if __name__ == "__main__":
    amalgamate()
