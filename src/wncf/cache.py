"""Append-only score cache: one JSON line per peg-candidate model call.

A cached score is reused only when everything that could change the model's output matches:
checkpoint and revision, precision, image grouping, chat wrapper, the fully rendered prompt, the
exact bytes of all four images in order, and the software versions. The key is the SHA-256 of that
record, so a changed render, prompt or library silently misses the cache instead of reusing a stale
score. Lines are appended and flushed one at a time, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

from wncf import REPO_ROOT

CACHE = REPO_ROOT / "cache" / "scores.jsonl"


def file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def frozen_identity(arm: str | None = None) -> dict:
    """Everything except the images that determines a score, under the frozen configuration or one arm.

    The arm's name is deliberately not part of the identity: an arm that changes only the images
    (appearance) keeps the primary identity, and one that changes the prompt changes prompt_sha."""
    from importlib.metadata import version

    import wncf
    from wncf.config import arm_config
    from wncf.scorer_llava import render_prompt

    cfg = arm_config(arm)
    model_id, revision = wncf.MODELS[cfg["model"]]
    prompt = render_prompt(cfg["prompt"], cfg["chat"])
    return {
        "model_id": model_id, "revision": revision, "quant": cfg["quant"], "grouping": cfg["grouping"],
        "chat": cfg["chat"], "prompt_id": cfg["prompt"],
        "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(), "views": list(cfg["views"]),
        "versions": {p: version(p) for p in ("torch", "transformers", "bitsandbytes")},
    }


def pair_key(base: dict, image_paths: list) -> tuple[str, list[str]]:
    """Cache key for one peg-candidate call, from the four image files in prompt order."""
    shas = [file_sha(REPO_ROOT / p) for p in image_paths]
    return cache_key({**base, "image_sha": shas}), shas


class ScoreCache:
    def __init__(self, path: Path = CACHE):
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        self._incomplete: tuple[int, bytes, bytes] | None = None
        self._needs_newline = False
        if self.path.exists():
            content = self.path.read_bytes()
            lines = content.splitlines(keepends=True)
            offset = 0
            for index, line in enumerate(lines):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # A writer killed mid-record can leave one unterminated JSON prefix.
                        # Completed lines and corruption in the middle must still fail loudly.
                        if index != len(lines) - 1 or line.endswith((b"\n", b"\r")) or not line.lstrip().startswith(b"{"):
                            raise
                        self._incomplete = (offset, line, content)
                        warnings.warn(f"Ignoring incomplete final cache record in {self.path}; "
                                      "it will be backed up before the next append", RuntimeWarning, stacklevel=2)
                        break
                    if row["key"] in self.rows and self.rows[row["key"]] != row:
                        raise ValueError(f"conflicting duplicate cache key {row['key']}")
                    self.rows[row["key"]] = row
                offset += len(line)
            self._needs_newline = bool(content) and not content.endswith((b"\n", b"\r"))

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def add(self, key: str, row: dict) -> None:
        if key in self.rows:
            return
        row = {"key": key, **row}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._incomplete is not None:
            offset, tail, original = self._incomplete
            if self.path.read_bytes() != original:
                raise RuntimeError("cache changed during recovery; reload before appending")
            backup = self.path.with_name(self.path.name + "." + hashlib.sha256(tail).hexdigest()[:12] + ".incomplete")
            backup.write_bytes(tail)
            with self.path.open("r+b") as stream:
                stream.truncate(offset)
            self._incomplete = None
            self._needs_newline = False
        with self.path.open("a", encoding="utf-8") as f:
            if self._needs_newline:
                f.write("\n")
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        self._needs_newline = False
        self.rows[key] = row
