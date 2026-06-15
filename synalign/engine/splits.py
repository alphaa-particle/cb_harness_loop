import numpy as np

from engine.config import RANDOM_SEED, SPLIT_FRACTIONS


def assign_splits(user_ids: list[int], seed: int = RANDOM_SEED) -> dict[int, str]:
    """Deterministically split users into disjoint train/dev/test groups."""
    rng = np.random.default_rng(seed + 7)
    shuffled = list(user_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_FRACTIONS["train"])
    n_dev = int(n * SPLIT_FRACTIONS["dev"])

    assignment: dict[int, str] = {}
    for i, uid in enumerate(shuffled):
        if i < n_train:
            assignment[uid] = "train"
        elif i < n_train + n_dev:
            assignment[uid] = "dev"
        else:
            assignment[uid] = "test"
    return assignment
