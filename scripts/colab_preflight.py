#!/usr/bin/env python3
"""Colab GPU preflight for BloomDepth.

Checks runtime paths, GPU availability, package imports, Hugging Face token status,
and presence of the default QAG context file before expensive model loading.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, out.strip()
    except Exception as exc:  # noqa: BLE001
        return 1, str(exc)


def _module_status(name: str) -> str:
    return "ok" if importlib.util.find_spec(name) else "missing"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contexts",
        default="data/interim/gate_v2/readiness/qag_pilot_gold_contexts.jsonl",
        help="Context JSONL expected by Phase A.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    from configs.config import CFG

    cuda_code, cuda_out = _run(["nvidia-smi"])
    gpu_ok = cuda_code == 0
    contexts_path = Path(args.contexts)
    if not contexts_path.is_absolute():
        contexts_path = CFG.paths.root / contexts_path

    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(CFG.paths.root),
        "drive_base": str(CFG.drive_sync.drive_base),
        "gpu_ok": gpu_ok,
        "nvidia_smi": cuda_out[:2000],
        "hf_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "contexts_path": str(contexts_path),
        "contexts_exists": contexts_path.exists(),
        "modules": {
            "torch": _module_status("torch"),
            "transformers": _module_status("transformers"),
            "vllm": _module_status("vllm"),
            "huggingface_hub": _module_status("huggingface_hub"),
            "docling": _module_status("docling"),
            "fasttext": _module_status("fasttext"),
        },
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("BloomDepth Colab preflight")
        print(f"- project_root: {report['project_root']}")
        print(f"- drive_base:   {report['drive_base']}")
        print(f"- gpu_ok:       {report['gpu_ok']}")
        print(f"- hf_token_set: {report['hf_token_set']}")
        print(f"- contexts:     {report['contexts_path']} ({'ok' if report['contexts_exists'] else 'missing'})")
        print("- modules:")
        for name, status in report["modules"].items():
            print(f"  - {name}: {status}")
        if gpu_ok:
            print("\n[nvidia-smi]")
            print(cuda_out)

    hard_fail = not gpu_ok
    if hard_fail:
        print("ERROR: GPU is not available. In Colab, use Runtime > Change runtime type > GPU.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
