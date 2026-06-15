import importlib
import json
from pathlib import Path

import yaml

from engine.config import DOMAINS_DIR


class DomainPack:
    """Loads everything from domains/<name>/ and exposes it to the engine."""

    def __init__(self, name: str):
        self.name = name
        self.path: Path = DOMAINS_DIR / name

        self.profile_schema = json.loads((self.path / "profile_schema.json").read_text(encoding="utf-8"))
        self.aliases = yaml.safe_load((self.path / "aliases.yaml").read_text(encoding="utf-8"))
        self.perturbations = yaml.safe_load((self.path / "perturbations.yaml").read_text(encoding="utf-8"))
        self.eval_config = yaml.safe_load((self.path / "eval_config.yaml").read_text(encoding="utf-8"))

        module = importlib.import_module(f"domains.{name}.ground_truth")
        self.build_ground_truth = module.build_ground_truth

        self.documents_dir: Path = self.path / "documents"

    # Convenience accessors -------------------------------------------------

    def entity_aliases(self) -> dict[str, list[str]]:
        return self.aliases.get("entities", {})

    def field_aliases(self) -> dict[str, list[str]]:
        return self.aliases.get("fields", {})

    def forbidden_claims(self) -> dict[str, list[str]]:
        claims = self.aliases.get("forbidden_claims", {})
        return {cid: spec["patterns"] for cid, spec in claims.items()}
