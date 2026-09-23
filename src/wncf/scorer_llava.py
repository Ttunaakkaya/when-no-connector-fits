"""LLaVA-OneVision Yes/No scorer.

p_yes = P(yes) / (P(yes) + P(no)), where P(yes) and P(no) sum the softmax over the full
vocabulary at the first generated position across every single-token spelling of the
answer. The prompt is built by hand rather than with the HF chat template because the
paper interleaves text and images, and the template renders all images first.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import wncf  # noqa: F401  (pins HF caches before transformers is imported)
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaOnevisionForConditionalGeneration

from wncf import MODELS

IMAGE = "<image>"

# Yajima et al. (IROS 2025), Sec. IV-A "Prompt to VLM": each <imageN> placeholder precedes its sentence.
PROMPTS = {
    "paper": (
        f"{IMAGE} This is a cross-sectional image of a peg. "
        f"{IMAGE} This is another image of a peg from a different angle. "
        f"{IMAGE} This is a cross-sectional image of a hole. "
        f"{IMAGE} This is another image of a hole from a different angle. "
        "Can the peg in images 1 and 2 be perfectly inserted into the hole in images 3 and 4? "
        "Please answer with only yes or no."
    ),
    # Same structure, but states the task contract (master plan section 3) and the shared scale.
    "controlled": (
        f"{IMAGE} This is a top-down image of a peg's cross-section. "
        f"{IMAGE} This is another image of the same peg from a 30-degree angle. "
        f"{IMAGE} This is a top-down image of a hole. "
        f"{IMAGE} This is another image of the same hole from a 30-degree angle. "
        "All four images share the same scale. The peg is centered over the hole, may be turned by "
        "0, 90, 180 or 270 degrees, and is then pushed straight down. Consider only shape and size. "
        "Can the peg in images 1 and 2 be fully inserted into the hole in images 3 and 4? "
        "Please answer with only yes or no."
    ),
    # Stage-4 probes (dev only): the paper's single-view variant, and a shape-comparison question
    # that asks for recognition of shape and size without reasoning about insertion.
    "paper_v1": (
        f"{IMAGE} This is a cross-sectional image of a peg. "
        f"{IMAGE} This is a cross-sectional image of a hole. "
        "Can the peg in image 1 be perfectly inserted into the hole in image 2? "
        "Please answer with only yes or no."
    ),
    "shape": (
        f"{IMAGE} This is a top-down image of a peg's cross-section. "
        f"{IMAGE} This is another image of the same peg from a 30-degree angle. "
        f"{IMAGE} This is a top-down image of a hole. "
        f"{IMAGE} This is another image of the same hole from a 30-degree angle. "
        "All four images share the same scale. Is the opening of the hole in images 3 and 4 the same "
        "shape and size as the cross-section of the peg in images 1 and 2, possibly turned by 90 degrees? "
        "Please answer with only yes or no."
    ),
}
# Which rendered views fill each prompt's image slots: peg views, then the same socket views.
PROMPT_VIEWS = {"paper": ("v1", "v2"), "controlled": ("v1", "v2"), "paper_v1": ("v1",), "shape": ("v1", "v2")}

# ChatML "qwen_1_5" conversation template that LLaVA-OneVision was trained with (LLaVA-NeXT repo).
# The llava-hf chat template differs: images first, "user " + space, no system turn. See docs/decisions.md.
CHAT = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{body}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def render_prompt(prompt_id: str, chat: str = "qwen_1_5") -> str:
    """The full text the processor receives, without loading the model (the cache key uses it)."""
    body = PROMPTS[prompt_id]
    return CHAT.format(body=body) if chat == "qwen_1_5" else hf_template(body)


def hf_template(body: str) -> str:
    """What the llava-hf chat template renders for one user turn: all images first, then the text."""
    n = body.count(IMAGE)
    text = body.replace(f"{IMAGE} ", "")
    return f"<|im_start|>user {IMAGE * n}\n{text}<|im_end|><|im_start|>assistant\n"

YES_VARIANTS = ["Yes", "yes", "YES", " Yes", " yes", " YES"]
NO_VARIANTS = ["No", "no", "NO", " No", " no", " NO"]


def answer_token_ids(tokenizer, variants: list[str]) -> list[int]:
    """IDs of the variants that encode to exactly one token (multi-token spellings are dropped)."""
    ids = set()
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            ids.add(enc[0])
    return sorted(ids)


@dataclass
class Score:
    p_yes: float  # normalized: P(yes) / (P(yes) + P(no))
    mass_yes: float  # raw softmax mass on yes tokens
    mass_no: float
    answer: str  # "yes" / "no" / "other": what greedy decoding would emit first
    top_token: str
    top_prob: float  # raw probability of that emitted token: the paper's confidence p(o_m), used by B0
    n_input_tokens: int
    latency_s: float  # preprocessing + forward pass
    prep_s: float  # the processor's share of latency_s (CPU image preprocessing and tokenisation)
    peak_vram_gb: float

    def as_dict(self) -> dict:
        return asdict(self)


class LlavaScorer:
    def __init__(self, model: str = "7b", quant: str = "nf4", grouping: str = "nested", chat: str = "qwen_1_5",
                 device: str = "cuda:0"):
        # grouping="nested" passes the query's images as one multi-image sample: each image is encoded at base
        # resolution (~730 tokens). "flat" makes the processor treat every image as a single-image sample and
        # tile it at high resolution (anyres, ~3.7k tokens at 512 px). Freeze one choice on development data.
        if grouping not in ("nested", "flat"):
            raise ValueError(f"unknown grouping '{grouping}' (nested | flat)")
        if chat not in ("qwen_1_5", "hf"):
            raise ValueError(f"unknown chat format '{chat}' (qwen_1_5 | hf)")
        self.model_key = model
        self.model_id, self.revision = MODELS[model]
        self.quant = quant
        self.grouping = grouping
        self.chat = chat
        self.device = device

        self.processor = _from_pretrained(AutoProcessor, self.model_id, revision=self.revision)
        self.model = _from_pretrained(
            LlavaOnevisionForConditionalGeneration,
            self.model_id,
            revision=self.revision,
            dtype=torch.float16,
            quantization_config=_quant_config(quant),
            device_map=device,
            low_cpu_mem_usage=True,
        ).eval()

        tok = self.processor.tokenizer
        self.yes_ids = answer_token_ids(tok, YES_VARIANTS)
        self.no_ids = answer_token_ids(tok, NO_VARIANTS)
        assert self.yes_ids and self.no_ids and not set(self.yes_ids) & set(self.no_ids)

    def build_prompt(self, prompt_id: str) -> str:
        return render_prompt(prompt_id, self.chat)

    @torch.inference_mode()
    def score(self, images: list[Image.Image], prompt_id: str = "paper") -> Score:
        prompt = self.build_prompt(prompt_id)
        n_slots = prompt.count(IMAGE)
        if len(images) != n_slots:
            raise ValueError(f"prompt '{prompt_id}' has {n_slots} image slots, got {len(images)} images")

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        batch = [images] if self.grouping == "nested" else images
        inputs = self.processor(images=batch, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        prep = time.perf_counter() - t0
        out = self.model(**inputs, logits_to_keep=1, use_cache=False)
        probs = out.logits[0, -1].float().softmax(-1)

        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        mass_yes = probs[self.yes_ids].sum().item()
        mass_no = probs[self.no_ids].sum().item()
        top = int(probs.argmax())
        answer = "yes" if top in self.yes_ids else "no" if top in self.no_ids else "other"
        return Score(
            p_yes=mass_yes / (mass_yes + mass_no) if mass_yes + mass_no > 0 else float("nan"),
            mass_yes=mass_yes,
            mass_no=mass_no,
            answer=answer,
            top_token=self.processor.tokenizer.decode([top]),
            top_prob=probs[top].item(),
            n_input_tokens=int(inputs["input_ids"].shape[1]),
            latency_s=latency,
            prep_s=prep,
            peak_vram_gb=torch.cuda.max_memory_allocated() / 1024**3,
        )


def _from_pretrained(cls, model_id: str, revision: str, **kw):
    # Revisions are pinned, so the local snapshot is authoritative: load from its folder and only touch
    # the Hub when the snapshot is missing. (Catching OSError from from_pretrained would also swallow
    # real failures such as Windows running out of page file.)
    try:
        path = snapshot_download(model_id, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError:
        path = snapshot_download(model_id, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    return cls.from_pretrained(path, **kw)


def _quant_config(quant: str) -> BitsAndBytesConfig | None:
    # The vision tower, projector and lm_head stay in fp16; only the Qwen2 decoder is quantized.
    skip = ["vision_tower", "multi_modal_projector", "lm_head"]
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=skip,
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    if quant == "fp16":
        return None
    raise ValueError(f"unknown quant '{quant}' (nf4 | int8 | fp16)")
