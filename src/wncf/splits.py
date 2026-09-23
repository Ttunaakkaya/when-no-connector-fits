"""Family-level data splits, fixed before any model scoring.

A whole family (one shape template with all its size, aspect and mirror variants, and any later
perturbations) stays in one split, so near-duplicates never straddle development, calibration and
test. Development is for implementation and pilot choices, calibration only for rejection
thresholds, and test is used once with everything frozen.
"""

import random

SPLIT_SEED = 20260922
SPLIT_SIZES = {"dev": 4, "calib": 7, "test": 7}  # 18 families, about 20% / 40% / 40%

# Families whose shape templates the eight-shape reconstruction used (decisions.md D12). The split
# stands as recorded; results are reported separately for exposed and unexposed query families.
EXPOSED_FAMILIES = frozenset({"cross", "square", "rect", "trapezoid", "circle", "obround", "polygon"})


def family_splits(families) -> dict[str, str]:
    order = sorted(set(families))
    if len(order) != sum(SPLIT_SIZES.values()):
        raise ValueError(f"expected {sum(SPLIT_SIZES.values())} families, got {len(order)}")
    random.Random(SPLIT_SEED).shuffle(order)
    out, i = {}, 0
    for split, n in SPLIT_SIZES.items():
        for fam in order[i:i + n]:
            out[fam] = split
        i += n
    return out
