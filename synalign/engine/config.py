from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DOMAINS_DIR = ROOT_DIR / "domains"
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"

# Which domain pack to load. Change this one line to switch domains.
ACTIVE_DOMAIN = os.environ.get("SYNALIGN_DOMAIN", "welfare_demo")

RANDOM_SEED = 42
N_SYNTHETIC_USERS = 300

# Disjoint user-level splits (fractions must sum to 1.0).
SPLIT_FRACTIONS = {"train": 0.6, "dev": 0.2, "test": 0.2}

# Which assistant to audit and serve by default.
#   "naive"        -> non-LLM baseline (instant, no dependencies)
#   "rules_only"   -> no model: shows the matching rules word for word
#   "llama_cpp"    -> a quantised model served by llama.cpp (scripts/serve_models.py)
#   "transformers" -> a local Hugging Face model (needs torch; slow on CPU)
#   "ollama"       -> a model served by a running Ollama server
ASSISTANT_BACKEND = os.environ.get("SYNALIGN_ASSISTANT_BACKEND", "naive")

# Where the invented users come from.
#   "sampler"    -> drawn from the ranges in the domain's profile_schema.json
#                   (the users every saved run was tested on)
#   "population" -> drawn from the population file the schema points to
USER_SOURCES = ("sampler", "population")
USER_SOURCE = os.environ.get("SYNALIGN_USER_SOURCE", "sampler")

# A full audit is N_SYNTHETIC_USERS x 5 conditions cases. With a model on CPU
# that is far too many, so model runs default to this smaller sample.
N_USERS_LLM = 20
