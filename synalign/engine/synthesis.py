import numpy as np

from engine.config import RANDOM_SEED


def _from_population(row: dict, source: dict):
    """One profile value taken from a population row, as the schema's mapping says."""
    value = row[source["column"]]
    if "map" in source:
        if value not in source["map"]:
            raise ValueError(f"No mapping for {source['column']}={value!r}")
        return source["map"][value]
    if isinstance(value, str):
        return value
    value = float(value) / source.get("divide_by", 1)
    step = source.get("round_to", 1)
    return max(int(round(value / step) * step), source.get("min", 0))


def make_synthetic_users(schema: dict, n: int, seed: int = RANDOM_SEED, population=None) -> list[dict]:
    """Generate n user profiles from a declarative schema.

    Values can be None when the schema gives the field a missing_prob —
    meaning the user genuinely does not know this fact about themselves.

    With `population` (a table of ready-made people), n distinct rows are
    drawn and the fields listed under schema["population"]["fields"] are taken
    from them; every other field is still sampled as before.
    """
    rng = np.random.default_rng(seed)
    fields = schema["fields"]
    users: list[dict] = []

    rows, mapping, id_column = None, {}, None
    if population is not None:
        if n > len(population):
            raise ValueError(f"Asked for {n} users but the population has {len(population)}")
        mapping = schema["population"]["fields"]
        id_column = schema["population"].get("id_column")
        rows = population.iloc[rng.choice(len(population), n, replace=False)].to_dict("records")

    for user_id in range(n):
        profile: dict = {"user_id": user_id}
        if id_column:
            profile[id_column] = int(rows[user_id][id_column])

        for name, spec in fields.items():
            sampler = spec["sampler"]
            kind = sampler["kind"]

            if name in mapping:
                value = _from_population(rows[user_id], mapping[name])

            elif kind == "uniform_int":
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
