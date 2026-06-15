import numpy as np

from engine.config import RANDOM_SEED


def make_synthetic_users(schema: dict, n: int, seed: int = RANDOM_SEED) -> list[dict]:
    """Generate n user profiles from a declarative schema.

    Values can be None when the schema gives the field a missing_prob —
    meaning the user genuinely does not know this fact about themselves.
    """
    rng = np.random.default_rng(seed)
    fields = schema["fields"]
    users: list[dict] = []

    for user_id in range(n):
        profile: dict = {"user_id": user_id}

        for name, spec in fields.items():
            sampler = spec["sampler"]
            kind = sampler["kind"]

            if kind == "uniform_int":
                value = int(rng.integers(sampler["low"], sampler["high"] + 1))

            elif kind == "normal_int":
                raw = rng.normal(sampler["mean"], sampler["std"])
                raw = float(np.clip(raw, sampler["min"], sampler["max"]))
                step = sampler.get("round_to", 1)
                value = int(raw // step * step)

            elif kind == "choice":
                options = sampler["options"]
                p = sampler.get("p")
                value = str(rng.choice(options, p=p))

            elif kind == "conditional_bool":
                parent = profile[sampler["depends_on"]]
                prob = sampler["true_prob"].get(parent, 0.5)
                value = bool(rng.random() < prob)

            else:
                raise ValueError(f"Unknown sampler kind: {kind}")

            # The user may simply not know this fact.
            if rng.random() < spec.get("missing_prob", 0.0):
                value = None

            profile[name] = value

        users.append(profile)

    return users
