"""
Baremetal AI Inference Engine - Lightweight Python Bindings (Pure Ctypes)
Run 30B MoE LLMs in Python with <2 GB RAM!
"""

import os
import sys
import ctypes
from typing import Callable, Optional

# Load shared library
_lib_names = ["libbaremetal.so", "libbaremetal.dylib", "baremetal.dll"]
_lib_path = None
_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

for name in _lib_names:
    candidate = os.path.join(_base_dir, name)
    if os.path.exists(candidate):
        _lib_path = candidate
        break

class BMConfig(ctypes.Structure):
    _fields_ = [
        ("num_threads", ctypes.c_int),
        ("backend", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
    ]

class BMStats(ctypes.Structure):
    _fields_ = [
        ("ttft_ms", ctypes.c_double),
        ("mean_latency_ms", ctypes.c_double),
        ("tokens_per_sec", ctypes.c_double),
        ("peak_rss_mb", ctypes.c_double),
        ("total_tokens", ctypes.c_int),
    ]

CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p)

class BaremetalEngine:
    def __init__(self, model_path: str, threads: int = 8, temperature: float = 0.7):
        if _lib_path is None or not os.path.exists(_lib_path):
            raise RuntimeError(f"Baremetal dynamic library not found. Run 'make lib' in {_base_dir} first.")
        
        self.lib = ctypes.CDLL(_lib_path)
        
        self.lib.bm_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(BMConfig)]
        self.lib.bm_create.restype = ctypes.c_void_p

        self.lib.bm_generate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, CALLBACK_TYPE, ctypes.c_void_p]
        self.lib.bm_generate.restype = ctypes.c_int

        self.lib.bm_get_stats.argtypes = [ctypes.c_void_p]
        self.lib.bm_get_stats.restype = BMStats

        self.lib.bm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.bm_destroy.restype = None

        cfg = BMConfig(
            num_threads=threads,
            backend=1, # AVX2
            temperature=temperature,
            top_k=40,
            top_p=0.9,
            repetition_penalty=1.05
        )
        self.engine = self.lib.bm_create(model_path.encode('utf-8'), ctypes.byref(cfg))
        if not self.engine:
            raise RuntimeError(f"Failed to initialize engine with model {model_path}")

    def generate(self, prompt: str, max_tokens: int = 32, callback: Optional[Callable[[str], None]] = None) -> str:
        collected = []
        def _cb(token_bytes, token_id, user_data):
            text = token_bytes.decode('utf-8', errors='replace')
            collected.append(text)
            if callback:
                callback(text)

        c_cb = CALLBACK_TYPE(_cb)
        self.lib.bm_generate(self.engine, prompt.encode('utf-8'), max_tokens, c_cb, None)
        return "".join(collected)

    def stats(self) -> dict:
        st = self.lib.bm_get_stats(self.engine)
        return {
            "ttft_ms": st.ttft_ms,
            "mean_latency_ms": st.mean_latency_ms,
            "tokens_per_sec": st.tokens_per_sec,
            "peak_rss_mb": st.peak_rss_mb,
            "total_tokens": st.total_tokens
        }

    def __del__(self):
        if hasattr(self, 'engine') and self.engine:
            self.lib.bm_destroy(self.engine)
            self.engine = None
