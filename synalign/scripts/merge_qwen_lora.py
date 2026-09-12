import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "models/qwen-0.8b"
ADAPTER = "models/qwen-0.8b-synalign-lora"
OUT = "models/qwen-0.8b-synalign-merged"

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
