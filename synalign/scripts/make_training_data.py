import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import ACTIVE_DOMAIN
from engine.domain_pack import DomainPack
from engine.training_data import create_sft_data, create_preference_data


if __name__ == "__main__":
    pack = DomainPack(ACTIVE_DOMAIN)
    create_sft_data(pack)
    create_preference_data(pack)
    print("\nIMPORTANT: have a domain expert review sft_train.jsonl before any training run.")
