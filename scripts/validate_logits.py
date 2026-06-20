#!/usr/bin/env python3
"""
Baremetal AI vs. llama.cpp Mathematical Correctness Validation Harness.

This script ensures that the Baremetal engine produces identical deterministic
greedy output (T=0.0) compared to the reference implementation in llama.cpp.

Requires:
    pip install llama-cpp-python
"""

import sys
import argparse
from typing import List

try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python is required for validation.")
    print("Please install with: pip install llama-cpp-python")
    sys.exit(1)

# Import the baremetal python wrapper
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from bindings.python.baremetal import BaremetalEngine

def run_llama_cpp(model_path: str, prompt: str, max_tokens: int) -> List[str]:
    print("[llama.cpp] Loading model...")
    # Suppress verbose llama.cpp logs by capturing stdout or using verbose=False
    llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
    
    print(f"[llama.cpp] Generating {max_tokens} tokens (Greedy/T=0)...")
    res = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_k=1,
        echo=False
    )
    
    text = res['choices'][0]['text']
    return text

def run_baremetal(model_path: str, prompt: str, max_tokens: int) -> str:
    print("\n[Baremetal] Loading model...")
    engine = BaremetalEngine(model_path=model_path, threads=8, temperature=0.0)
    
    print(f"[Baremetal] Generating {max_tokens} tokens (Greedy/T=0)...")
    text = engine.generate(prompt, max_tokens=max_tokens)
    return text

def main():
    parser = argparse.ArgumentParser(description="Validate Baremetal AI output against llama.cpp")
    parser.add_argument("model", help="Path to the GGUF model file")
    parser.add_argument("--prompt", default="The history of artificial intelligence dates back to", help="Input prompt")
    parser.add_argument("--tokens", type=int, default=32, help="Number of tokens to generate")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        sys.exit(1)

    print("=========================================================")
    print("   BAREMETAL AI MATHEMATICAL CORRECTNESS VALIDATION      ")
    print("=========================================================\n")

    # 1. Run reference
    llama_text = run_llama_cpp(model_path, args.prompt, args.tokens)
    
    # 2. Run baremetal engine
    baremetal_text = run_baremetal(model_path, args.prompt, args.tokens)
    
    print("\n=========================================================")
    print("                   VALIDATION RESULTS                    ")
    print("=========================================================")
    
    print("\n[llama.cpp Output]")
    print(repr(llama_text))
    
    print("\n[Baremetal Output]")
    print(repr(baremetal_text))
    
    print("\n---")
    
    if llama_text == baremetal_text:
        print("✅ SUCCESS: Outputs match perfectly! Mathematical correctness verified.")
        sys.exit(0)
    else:
        print("❌ FAILED: Outputs diverge. Check quantization rounding or kernel bugs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
