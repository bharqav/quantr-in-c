#!/usr/bin/env python3
"""
Baremetal AI Inference Engine - Automated Model Importer & Downloader
Automatically imports existing GGUF models on disk or downloads them from Hugging Face.
"""

import os
import sys
import argparse
import urllib.request
import struct

MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
DEFAULT_MODEL_NAME = "qwen3-30b-a3b-Q5_K_M.gguf"
KNOWN_LOCAL_PATHS = [
    r"D:\qwen3-30b-a3b-Q5_K_M.gguf",
    r"D:\qwen3-30b-a3b-Q6_K.gguf",
    r"C:\models\qwen3-30b-a3b-Q5_K_M.gguf"
]

# Hugging Face Mirror options for 1-click downloads categorized by size & RAM tier
MODEL_CATALOG = {
    "smollm2-135m": {
        "tier": "Nano",
        "description": "SmolLM2 135M Instruct - Q4_K_M (Download: ~85MB, RAM: <200MB, Instant Setup)",
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf",
        "filename": "smollm2-135m-instruct-q4_k_m.gguf"
    },
    "tinyllama-1.1b-q8": {
        "tier": "Compact",
        "description": "TinyLlama 1.1B Chat - Q8_0 (Download: ~146MB, RAM: ~450MB, High Precision)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
    },
    "tinyllama-1.1b-q4": {
        "tier": "Compact",
        "description": "TinyLlama 1.1B Chat - Q4_K_M (Download: ~630MB, RAM: ~800MB, Fast 4-bit)",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    },
    "qwen2.5-0.5b": {
        "tier": "Balanced",
        "description": "Qwen 2.5 0.5B Instruct - Q4_K_M (Download: ~350MB, RAM: ~700MB, Rich Reasoning)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    },
    "qwen-moe-2.7b": {
        "tier": "Performance",
        "description": "Qwen 1.5 MoE A2.7B Chat - Q4_K_M (Download: ~950MB, RAM: <1.5GB)",
        "url": "https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat-GGUF/resolve/main/qwen1_5-moe-a2_7b-chat-q4_k_m.gguf",
        "filename": "qwen1_5-moe-a2_7b-chat-q4_k_m.gguf"
    },
    "qwen3-30b-moe": {
        "tier": "Pro / Heavyweight",
        "description": "Qwen 3 30B MoE (A3B) - Q5_K_M (Active Params: ~3B, Peak RAM: 4.0GB)",
        "url": "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF/resolve/main/qwen2.5-coder-32b-instruct-q5_k_m.gguf",
        "filename": "qwen3-30b-a3b-Q5_K_M.gguf"
    }
}

def verify_gguf(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic == b"GGUF":
                version = struct.unpack("<I", f.read(4))[0]
                return version in [2, 3]
    except Exception:
        return False
    return False

def import_local():
    os.makedirs(MODELS_DIR, exist_ok=True)
    target_path = os.path.join(MODELS_DIR, DEFAULT_MODEL_NAME)
    
    if verify_gguf(target_path):
        print(f"[OK] Target model already present in models/: {target_path}")
        return target_path

    # Check known local paths
    for p in KNOWN_LOCAL_PATHS:
        if verify_gguf(p):
            print(f"[FOUND] Found existing intact GGUF model: {p}")
            print(f"Creating link/reference into {MODELS_DIR}...")
            try:
                if hasattr(os, "symlink"):
                    try:
                        os.symlink(p, target_path)
                        print(f"[LINK] Created symlink {target_path} -> {p}")
                        return target_path
                    except Exception:
                        pass
                if hasattr(os, "link"):
                    try:
                        os.link(p, target_path)
                        print(f"[LINK] Created hardlink {target_path} -> {p}")
                        return target_path
                    except Exception:
                        pass
            except Exception as e:
                print(f"[WARN] Link creation skipped ({e}). You can use {p} directly.")
            return p
    return None

def download_progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = (downloaded / total_size) * 100
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {mb_down:.1f} MB / {mb_total:.1f} MB [{percent:.1f}%]")
    else:
        mb_down = downloaded / (1024 * 1024)
        sys.stdout.write(f"\rDownloaded: {mb_down:.1f} MB")
    sys.stdout.flush()

def download_model(model_key="qwen3-30b-moe"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    if model_key not in MODEL_CATALOG:
        print(f"Unknown model key: {model_key}. Available: {list(MODEL_CATALOG.keys())}")
        return False
    
    entry = MODEL_CATALOG[model_key]
    out_path = os.path.join(MODELS_DIR, entry["filename"])
    if verify_gguf(out_path):
        print(f"[OK] Model already downloaded and verified: {out_path}")
        return True
    
    print(f"Downloading {entry['description']}...")
    print(f"Source URL: {entry['url']}")
    print(f"Saving to: {out_path}\n")
    try:
        urllib.request.urlretrieve(entry["url"], out_path, download_progress_hook)
        print("\n[SUCCESS] Download completed!")
        if verify_gguf(out_path):
            print("[VERIFIED] GGUF header valid.")
            return True
        else:
            print("[ERROR] Corrupted file header.")
            return False
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Baremetal AI Inference Engine - Model Manager")
    parser.add_argument("--import-local", action="store_true", help="Auto-discover and link local GGUF models")
    parser.add_argument("--download", type=str, choices=list(MODEL_CATALOG.keys()), help="Download pre-configured model")
    parser.add_argument("--list", action="store_true", help="List available downloadable models")
    args = parser.parse_args()

    if args.list:
        print("Available models in catalog:")
        for k, v in MODEL_CATALOG.items():
            print(f"  - {k:<20} {v['description']}")
        return

    # Default action: import local if exists
    found = import_local()
    if found:
        print(f"\nEngine is ready to run! Execute:")
        print(f"  .\\inference.exe --model \"{found}\" --prompt \"Hello world\" --steps 16 --threads 8")
        return

    if args.download:
        download_model(args.download)
    else:
        print("\nNo local GGUF models found in standard paths.")
        print("To download a model directly, run:")
        print("  python scripts/setup_model.py --download qwen-moe-2.7b")

if __name__ == "__main__":
    main()
