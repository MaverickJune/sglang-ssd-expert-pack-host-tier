#!/usr/bin/env python3
"""Prepare DeepSeek-V4-Flash Expert Pack inputs on upstream SGLang main.

Upstream main moved tools/expert_pack/prepare_deepseek_pack.py into
python/sglang/srt/model_loader/expert_pack/ and dropped the blog's
`benchmark_deepseek_5090.py --validate-only` step; the same work is done by
the library functions used by the server's raw-GGUF auto-preparation path.
This driver runs those steps explicitly so each is logged and reproducible:
  1. model-meta (config.json, tokenizer.json, ...) derived from the GGUF
  2. build (or reuse) DeepSeek-V4-Flash.expert-pack + manifest beside the GGUF
  3. validate the pack with prepare_deepseek_pack.py --check-only
and prints the artifact dir / fingerprint and the manifest digests the
launch command needs.
"""
import argparse, json, subprocess, sys
from pathlib import Path

from sglang.srt.model_loader import expert_pack_runtime as rt

ap = argparse.ArgumentParser()
ap.add_argument("--gguf", type=Path, required=True)
ap.add_argument("--safety-margin-gib", type=float, default=16.0)
ap.add_argument("--check-only", action="store_true")
a = ap.parse_args()
source = a.gguf.resolve(strict=True)
artifact_dir = rt._deepseek_artifact_dir_for_source(source).resolve()
artifact_dir.mkdir(parents=True, exist_ok=True)
print(f"artifact_dir={artifact_dir}", flush=True)
model_config = rt._prepare_deepseek_model_metadata(source, artifact_dir)
print(f"model_config={model_config}", flush=True)
tools_dir = rt._expert_pack_tools_dir()
cmd = [sys.executable, str(tools_dir / "prepare_deepseek_pack.py"), "--gguf", str(source),
       "--model-config", str(model_config), "--safety-margin-gib", str(a.safety_margin_gib)]
if a.check_only:
    cmd.append("--check-only")
print("+", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=tools_dir, check=True)
pack = source.parent / "DeepSeek-V4-Flash.expert-pack"
manifest = json.loads((source.parent / "DeepSeek-V4-Flash.expert-pack.manifest.json").read_text())
out = {
    "artifact_dir": str(artifact_dir), "model_meta": str(model_config.parent),
    "pack_path": str(pack), "manifest_path": str(pack) + ".manifest.json",
    "source_path": str(source), "source_sha256": manifest["source"]["sha256"],
    "model_identity_sha256": manifest["model"]["model_identity_sha256"],
    "config_sha256": manifest["model"]["config_sha256"],
    "pack_size": manifest.get("pack_size"), "object_stride": manifest.get("object_stride"),
    "num_layers": manifest["model"]["num_layers"], "num_routed_experts": manifest["model"]["num_routed_experts"],
    "top_k": manifest["model"]["top_k"],
}
(artifact_dir / "launch_inputs.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
