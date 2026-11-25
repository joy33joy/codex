"""
Export the trained Ganyu adapter so it can run outside Tinker (e.g. on a Mac
with llama.cpp or Ollama), attached to Qwen3.8-27B or the abliterated copy
huihui-ai/Huihui-Qwen3.8-27B-abliterated (same architecture).

Steps:
  1. Download the adapter from Tinker (address in adapter.txt, written by
     train.py).
  2. Convert it to standard PEFT format (export/peft_adapter/).
  3. Convert that to GGUF, the format llama.cpp and Ollama load
     (export/ganyu-lora-f16.gguf).

None of this needs the 56 GB base model: only its config and the names and
shapes of its weights, which are read from Hugging Face with small partial
downloads.

Run:  python ganyu/export_adapter.py --llama-cpp /path/to/llama.cpp
      (a clone of https://github.com/ggml-org/llama.cpp, for step 3)
"""

import argparse
import json
import struct
import subprocess
import sys
import urllib.request
from pathlib import Path

from common import ADAPTER_FILE, HERE

BASE_MODEL = "Qwen/Qwen3.8-27B"  # config and weight names are identical in the abliterated copy
EXPORT = HERE / "export"
CONFIG_FILES = ["config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json", "generation_config.json"]


def hf_url(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def fetch(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    request = urllib.request.Request(url)
    if byte_range:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1]}")
    with urllib.request.urlopen(request) as response:
        return response.read()


def remote_weight_shapes(repo: str, index: dict) -> dict[str, tuple[int, ...]]:
    """Names and shapes of every weight, read from the safetensors headers only."""
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(set(index["weight_map"].values())):
        url = hf_url(repo, shard)
        header_len = struct.unpack("<Q", fetch(url, (0, 7)))[0]
        header = json.loads(fetch(url, (8, 8 + header_len - 1)))
        for name, info in header.items():
            if name != "__metadata__":
                shapes[name] = tuple(info["shape"])
    return shapes


def fuse_split_qkv(peft_dir: Path, out_dir: Path) -> None:
    """Combine Tinker's separate in_proj_q/k/v adapters into one in_proj_qkv adapter.

    The model stores the linear-attention Q, K and V projections as ONE weight
    (in_proj_qkv, rows in the order Q|K|V); Tinker trains three separate LoRA
    adapters. llama.cpp can only attach an adapter to a weight that exists, so:
        A_qkv = [A_q; A_k; A_v]            (stacked: rank 3r)
        B_qkv = block_diag(B_q, B_k, B_v)
    gives B_qkv @ A_qkv = [B_q A_q; B_k A_k; B_v A_v], exactly the three updates.
    llama.cpp scales each adapter by alpha / (its own rank), so a rank-3r piece
    would be scaled 3x too weakly; B_qkv is multiplied by 3 to compensate.
    """
    import re
    import shutil

    import torch
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(peft_dir / "adapter_model.safetensors"))
    fused: dict[str, torch.Tensor] = {}
    prefixes = sorted({re.sub(r"\.in_proj_[qkv]\.lora_[AB]\.weight$", "", k)
                       for k in tensors if re.search(r"\.in_proj_[qkv]\.lora_[AB]\.weight$", k)})
    for k, v in tensors.items():
        if not re.search(r"\.in_proj_[qkv]\.lora_[AB]\.weight$", k):
            fused[k] = v
    for p in prefixes:
        a = [tensors[f"{p}.in_proj_{x}.lora_A.weight"] for x in "qkv"]
        b = [tensors[f"{p}.in_proj_{x}.lora_B.weight"] for x in "qkv"]
        fused[f"{p}.in_proj_qkv.lora_A.weight"] = torch.cat(a, dim=0).contiguous()
        fused[f"{p}.in_proj_qkv.lora_B.weight"] = (torch.block_diag(*b) * len(b)).contiguous()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    save_file(fused, str(out_dir / "adapter_model.safetensors"))
    shutil.copy(peft_dir / "adapter_config.json", out_dir / "adapter_config.json")
    print(f"    Combined q/k/v adapters into in_proj_qkv for {len(prefixes)} layers: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llama-cpp", type=Path, help="path to a llama.cpp clone (for the GGUF step)")
    args = parser.parse_args()

    from tinker_cookbook import weights
    from tinker_cookbook.weights import _adapter

    raw_dir, peft_dir = EXPORT / "tinker_adapter", EXPORT / "peft_adapter"
    config_dir = EXPORT / "base_config"
    gguf_file = EXPORT / "ganyu-lora-f16.gguf"

    # 1. Download from Tinker.
    if not (raw_dir / "adapter_model.safetensors").exists():
        print(f"1/3 Downloading the adapter from Tinker ({ADAPTER_FILE.read_text().strip()}) ...")
        weights.download(tinker_path=ADAPTER_FILE.read_text().strip(), output_dir=str(raw_dir))
    else:
        print(f"1/3 Adapter already downloaded: {raw_dir}")

    # 2. Convert to PEFT, with the base model's config but without its weights.
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in CONFIG_FILES:
        if not (config_dir / name).exists():
            (config_dir / name).write_bytes(fetch(hf_url(BASE_MODEL, name)))
    index = json.loads((config_dir / "model.safetensors.index.json").read_text())
    shapes = remote_weight_shapes(BASE_MODEL, index)
    print(f"2/3 Read {len(shapes)} weight names from {BASE_MODEL} (headers only); converting to PEFT ...")
    _adapter.get_model_state_keys = lambda _dir: set(shapes)
    _adapter.get_model_state_shapes = lambda _dir: dict(shapes)
    if peft_dir.exists():
        import shutil
        shutil.rmtree(peft_dir)
    weights.build_lora_adapter(base_model=str(config_dir), adapter_path=str(raw_dir), output_path=str(peft_dir))
    adapter_config = json.loads((peft_dir / "adapter_config.json").read_text())
    adapter_config["base_model_name_or_path"] = BASE_MODEL
    (peft_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2))
    print(f"    PEFT adapter: {peft_dir}")

    # 3. Convert to GGUF for llama.cpp / Ollama.
    if not args.llama_cpp:
        print("3/3 Skipped (no --llama-cpp given).")
        return
    fused_dir = EXPORT / "peft_adapter_fused"
    fuse_split_qkv(peft_dir, fused_dir)
    print("3/3 Converting to GGUF ...")
    subprocess.run([sys.executable, str(args.llama_cpp / "convert_lora_to_gguf.py"), str(fused_dir),
                    "--base-model-id", BASE_MODEL, "--outtype", "f16", "--outfile", str(gguf_file)],
                   check=True)
    print(f"\nDone: {gguf_file} ({gguf_file.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
