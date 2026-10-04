"""Fold a trained LoRA adapter into its base model, ready to convert to GGUF for llama.cpp.

    python scripts/merge_qwen_lora.py --base models/Qwen3.5-0.8B \
        --adapter models/qwen3.5-0.8b-schemes-lora --output models/qwen3.5-0.8b-schemes-merged
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

_args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_args.add_argument("--base", default="models/qwen-0.8b")
_args.add_argument("--adapter", default="models/qwen-0.8b-synalign-lora")
_args.add_argument("--output", default="models/qwen-0.8b-synalign-merged")
_args = _args.parse_args()
BASE, ADAPTER, OUT = _args.base, _args.adapter, _args.output

tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, local_files_only=True)

base = AutoModelForCausalLM.from_pretrained(
    BASE,
    trust_remote_code=True,
    local_files_only=True,
    device_map="auto",
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

model = PeftModel.from_pretrained(base, ADAPTER)
model = model.merge_and_unload()

model.save_pretrained(OUT, safe_serialization=True)
tokenizer.save_pretrained(OUT)

print("Merged model saved to:", OUT)
