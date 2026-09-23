"""Environment check: one matching and one non-matching query on synthetic images.

Done when the model returns Yes/No probabilities and VRAM + latency are logged.
Appends one JSON line per run to results/env_check.jsonl.

    uv run python scripts/smoke_llava.py --model 0.5b --quant fp16
    uv run python scripts/smoke_llava.py --model 7b --quant nf4
"""

import argparse
import json
import platform
import time
from importlib.metadata import version

import wncf
import torch
from PIL import Image, ImageDraw

from wncf.scorer_llava import LlavaScorer

S = 512


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (S, S), "white")
    return img, ImageDraw.Draw(img)


def rect_peg(oblique: bool) -> Image.Image:
    img, d = _canvas()
    if oblique:  # the plug's end face seen at ~30 deg, plus the body behind it
        d.polygon([(150, 200), (360, 200), (400, 300), (190, 300)], fill=(150, 150, 150), outline="black", width=4)
        d.rectangle([190, 300, 400, 420], fill=(120, 120, 120), outline="black", width=4)
    else:
        d.rectangle([146, 206, 366, 306], fill=(160, 160, 160), outline="black", width=6)
        d.rectangle([166, 226, 346, 256], fill=(40, 40, 40))
    return img


def rect_hole(oblique: bool) -> Image.Image:
    img, d = _canvas()
    d.rectangle([60, 60, 452, 452], fill=(200, 200, 200), outline="black", width=4)
    if oblique:
        d.polygon([(150, 210), (360, 210), (395, 290), (185, 290)], fill=(20, 20, 20))
    else:
        d.rectangle([140, 200, 372, 312], fill=(20, 20, 20))
        d.rectangle([166, 262, 346, 290], fill=(90, 90, 90))
    return img


def round_hole(oblique: bool) -> Image.Image:
    img, d = _canvas()
    d.rectangle([60, 60, 452, 452], fill=(200, 200, 200), outline="black", width=4)
    box = [196, 216, 316, 296] if oblique else [196, 196, 316, 316]
    d.ellipse(box, fill=(20, 20, 20))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="0.5b", choices=sorted(wncf.MODELS))
    ap.add_argument("--quant", default="fp16", choices=["nf4", "int8", "fp16"])
    ap.add_argument("--grouping", default="nested", choices=["nested", "flat"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    scorer = LlavaScorer(model=args.model, quant=args.quant, grouping=args.grouping)
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 1024**3

    peg = [rect_peg(False), rect_peg(True)]
    queries = {
        "rect_peg->rect_hole (expect yes)": peg + [rect_hole(False), rect_hole(True)],
        "rect_peg->round_hole (expect no)": peg + [round_hole(False), round_hole(True)],
    }
    scorer.score(queries[next(iter(queries))])  # warm-up: CUDA kernels, bnb dequant setup

    results = {}
    for name, imgs in queries.items():
        s = scorer.score(imgs)
        results[name] = s.as_dict()
        print(
            f"{name:34s} answer={s.answer:5s} p_yes={s.p_yes:.3f} "
            f"mass(yes+no)={s.mass_yes + s.mass_no:.3f} top={s.top_token!r}@{s.top_prob:.3f} "
            f"tokens={s.n_input_tokens} latency={s.latency_s:.2f}s peak={s.peak_vram_gb:.2f}GB"
        )

    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": scorer.model_id,
        "revision": scorer.revision,
        "quant": args.quant,
        "grouping": args.grouping,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "load_s": round(load_s, 1),
        "weights_gb": round(weights_gb, 2),
        "yes_ids": scorer.yes_ids,
        "no_ids": scorer.no_ids,
        "versions": {
            "python": platform.python_version(),
            **{p: version(p) for p in ["torch", "transformers", "accelerate", "bitsandbytes"]},
            "cuda": torch.version.cuda,
        },
        "prompt": scorer.build_prompt("paper"),
        "results": results,
    }
    out = wncf.REPO_ROOT / "results" / "env_check.jsonl"
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"load {load_s:.1f}s, weights {weights_gb:.2f}GB -> logged to {out.relative_to(wncf.REPO_ROOT)}")


if __name__ == "__main__":
    main()
