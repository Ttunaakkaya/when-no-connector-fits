"""Stage 4 implementation check: does the model actually see the images?

    uv run python scripts/diagnose_vision.py

Run after the pilot found p_yes nearly constant across fitting and non-fitting pairs. Dev parts only.
1. Free description: can the model name each of the four shapes it is given?
2. Blank control: does p_yes move when every image is replaced by a blank white image?
3. Chat format: the llava-hf template (images first, no system turn) instead of the qwen_1_5 wrapper.
"""

import torch
from PIL import Image

from wncf import REPO_ROOT
from wncf.scorer_llava import CHAT, IMAGE, PROMPTS, LlavaScorer

R = REPO_ROOT / "data" / "renders" / "procedural"
PAIRS = [("cross_01", "cross_01"), ("cross_01", "tube_03"), ("ellipse_03", "ellipse_03"), ("ellipse_03", "dshape_04")]


def imgs(peg, sock):
    return [Image.open(R / f"{peg}_peg_{v}.png").convert("RGB") for v in ("v1", "v2")] + \
           [Image.open(R / f"{sock}_socket_{v}.png").convert("RGB") for v in ("v1", "v2")]


@torch.inference_mode()
def generate(s, images, text, max_new_tokens=120):
    batch = [images] if s.grouping == "nested" else images
    x = s.processor(images=batch, text=text, return_tensors="pt").to(s.device, torch.float16)
    out = s.model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
    return s.processor.tokenizer.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def p_yes_with_text(s, images, text):
    s_build = s.build_prompt
    s.build_prompt = lambda _pid: text
    try:
        return s.score(images, "paper")
    finally:
        s.build_prompt = s_build


def main():
    s = LlavaScorer(model="7b", quant="nf4", grouping="nested")
    describe = CHAT.format(body=" ".join(f"{IMAGE}" for _ in range(4)) +
                           " Describe the main shape in each of the four images in one short line each, "
                           "numbered 1 to 4.")
    hf_template = ("<|im_start|>user " + IMAGE * 4 + "\n" + PROMPTS["paper"].replace(f"{IMAGE} ", "")
                   + "<|im_end|><|im_start|>assistant\n")
    blank = [Image.new("RGB", (768, 768), "white")] * 4

    for grouping in ("nested", "flat"):
        s.grouping = grouping
        print(f"\n=== {grouping}: description of {PAIRS[1]} ===")
        print(generate(s, imgs(*PAIRS[1]), describe))

    s.grouping = "nested"
    print("\n=== p_yes: qwen_1_5 wrapper (as piloted) | llava-hf template | answer text ===")
    for peg, sock in PAIRS:
        a = s.score(imgs(peg, sock), "paper")
        b = p_yes_with_text(s, imgs(peg, sock), hf_template)
        g = generate(s, imgs(peg, sock), s.build_prompt("paper"), max_new_tokens=8)
        print(f"{peg:11s} -> {sock:11s}  {a.p_yes:.3f} ({a.answer})  |  {b.p_yes:.3f} ({b.answer})  |  {g!r}")
    a = s.score(blank, "paper")
    print(f"{'blank':11s} -> {'blank':11s}  {a.p_yes:.3f} ({a.answer})")


if __name__ == "__main__":
    main()
