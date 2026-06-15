from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"

# Which domain pack to load. Change this one line to switch domains.
ACTIVE_DOMAIN = "welfare_demo"

RANDOM_SEED = 42
N_SYNTHETIC_USERS = 300

# Disjoint user-level splits (fractions must sum to 1.0).
SPLIT_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}

# Which assistant to audit by default.
#   "naive"  -> non-LLM baseline (instant, no dependencies)
#   "qwen"   -> Qwen3-0.6B via transformers (slow on CPU, needs torch)
#   "ollama" -> Qwen3-0.6B via a running Ollama server
ASSISTANT_BACKEND = "naive"

# A full audit is N_SYNTHETIC_USERS x 5 conditions cases. With a 0.6B model on
# CPU that is far too many, so LLM runs default to this smaller sample.
N_USERS_LLM = 20
