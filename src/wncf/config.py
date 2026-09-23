"""The frozen input configuration, in one importable place.

Every scoring path for the core experiment reads this, so the driver cannot drift from the frozen setup.
Changing any value invalidates cached scores, because the cache key covers all of it.
"""

FROZEN = {
    "model": "7b",
    "quant": "nf4",
    "grouping": "flat",
    "chat": "qwen_1_5",
    "prompt": "controlled",
    "views": ("v1", "v2"),
    "render_settings_sha": "7de39322d57d",
}

# Predeclared sensitivity arms: calibration blocks only, thresholds recalibrated
# inside the arm, never able to change the frozen primary configuration. The appearance arm swaps the
# neutral renders for an imitation of the paper's photos; the colours match scripts/replica_3dprint.py.
PAPER_LOOK = {"background": "#141414", "peg": "#d8322e", "socket": "#8fdca4", "floor": "#0b0b0b"}
# Each arm overrides exactly one thing: the appearance arm the renders, the prompt arm the prompt.
ARM_OVERRIDES = {
    "appearance": {"renders": "data/renders/arm_appearance"},
    "prompt": {"prompt": "paper"},
    "int8": {"quant": "int8"},
}
ARM_RENDERS = {a: o["renders"] for a, o in ARM_OVERRIDES.items() if "renders" in o}
ARM_SPLITS = frozenset({"calib"})


def arm_config(arm: str | None) -> dict:
    """The frozen configuration with one arm's override applied (None: the primary configuration)."""
    overrides = {k: v for k, v in ARM_OVERRIDES.get(arm, {}).items() if k != "renders"} if arm else {}
    return {**FROZEN, **overrides}


def arm_image_path(path: str, arm: str | None) -> str:
    """Map a primary render path to the arm's copy of the same image (same file name, other look)."""
    renders = ARM_RENDERS.get(arm) if arm else None
    return f"{renders}/{path.rsplit('/', 1)[-1]}" if renders else path
