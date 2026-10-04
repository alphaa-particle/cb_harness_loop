import importlib
import json
import os
from pathlib import Path

import pandas as pd
import yaml

from engine.config import DOMAINS_DIR, ROOT_DIR


class DomainPack:
    """Loads everything from domains/<name>/ and exposes it to the engine.

    Every pack needs documents/, aliases.yaml, eval_config.yaml and prompts.yaml.
    The invented-user audit (engine/audit.py) also needs profile_schema.json,
    perturbations.yaml and ground_truth.py; a pack without them can still be
    searched, served and evaluated on its own question files.
    """

    def __init__(self, name: str, corpus_path: str | Path | None = None):
        self.name = name
        self.path: Path = DOMAINS_DIR / name

        schema = self.path / "profile_schema.json"
        self.profile_schema = json.loads(schema.read_text(encoding="utf-8")) if schema.is_file() else {}
        self.aliases = yaml.safe_load((self.path / "aliases.yaml").read_text(encoding="utf-8"))
        # Scheme names written by scripts/build_corpus.py from the scheme records.
        entities = self.path / "documents" / "entities.json"
        if entities.is_file():
            self.aliases.setdefault("entities", {}).update(json.loads(entities.read_text(encoding="utf-8")))
        perturbations = self.path / "perturbations.yaml"
        self.perturbations = (yaml.safe_load(perturbations.read_text(encoding="utf-8"))
                              if perturbations.is_file() else {})
        self.eval_config = yaml.safe_load((self.path / "eval_config.yaml").read_text(encoding="utf-8"))
        self.prompts = yaml.safe_load((self.path / "prompts.yaml").read_text(encoding="utf-8"))
        # Optional: Hindi and Hinglish words mapped to the English words the documents use.
        glossary = self.path / "glossary.yaml"
        self.glossary = yaml.safe_load(glossary.read_text(encoding="utf-8")) if glossary.is_file() else {}
        if os.environ.get("SYNALIGN_RETRIEVAL_METHOD"):
            self.eval_config["retrieval"]["method"] = os.environ["SYNALIGN_RETRIEVAL_METHOD"]
        if os.environ.get("SYNALIGN_RETRIEVAL_MIN_SCORE"):
            self.eval_config["retrieval"]["min_score"] = float(os.environ["SYNALIGN_RETRIEVAL_MIN_SCORE"])
        if os.environ.get("SYNALIGN_RETRIEVAL_TOP_K"):
            self.eval_config["retrieval"]["top_k"] = int(os.environ["SYNALIGN_RETRIEVAL_TOP_K"])

        self.build_ground_truth = None
        if (self.path / "ground_truth.py").is_file():
            self.build_ground_truth = importlib.import_module(f"domains.{name}.ground_truth").build_ground_truth

        self.documents_dir: Path = self.path / "documents"
        override = corpus_path or os.environ.get("SYNALIGN_CORPUS_PATH")
        self.corpus_path = Path(override).expanduser().resolve() if override else self.documents_dir

    # Convenience accessors -------------------------------------------------

    def entity_aliases(self) -> dict[str, list[str]]:
        return self.aliases.get("entities", {})

    def field_aliases(self) -> dict[str, list[str]]:
        return self.aliases.get("fields", {})

    def load_population(self) -> pd.DataFrame:
        """The table of ready-made people named in profile_schema.json, one per row."""
        spec = self.profile_schema.get("population")
        if not spec:
            raise ValueError(f"Domain {self.name!r} has no 'population' block in profile_schema.json")
        path = (ROOT_DIR / spec["file"]).resolve()
        if not path.is_file():
            raise ValueError(f"Population file not found: {path}")
        # Values such as "None" are real categories here, not blanks.
        return pd.read_csv(path, keep_default_na=False)

    def uses_own_documents(self) -> bool:
        """False once the corpus has been swapped for one the pack's rules do not describe."""
        return self.corpus_path == self.documents_dir.resolve()

    def forbidden_claims(self) -> dict[str, list[str]]:
        claims = self.aliases.get("forbidden_claims", {})
        return {cid: spec["patterns"] for cid, spec in claims.items()}
