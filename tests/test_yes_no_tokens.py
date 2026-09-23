"""Tokenizer quirks here silently break p_yes, so pin the answer token IDs down."""

import pytest

import wncf
from transformers import AutoTokenizer

from wncf.scorer_llava import (CHAT, IMAGE, NO_VARIANTS, PROMPT_VIEWS, PROMPTS, YES_VARIANTS, answer_token_ids,
                               hf_template)

CORE = ["Yes", "yes", " Yes", " yes", "No", "no", " No", " no"]


def _tok(key):
    model_id, rev = wncf.MODELS[key]
    try:
        return AutoTokenizer.from_pretrained(model_id, revision=rev, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} not downloaded")


@pytest.fixture(scope="module", params=sorted(wncf.MODELS))
def tok(request):
    return _tok(request.param)


def test_core_answers_are_single_tokens(tok):
    for v in CORE:
        assert len(tok.encode(v, add_special_tokens=False)) == 1, v


def test_yes_and_no_sets_are_disjoint_and_round_trip(tok):
    yes, no = answer_token_ids(tok, YES_VARIANTS), answer_token_ids(tok, NO_VARIANTS)
    assert len(yes) >= 4 and len(no) >= 4
    assert not set(yes) & set(no)
    assert all(tok.decode([i]).strip().lower() == "yes" for i in yes)
    assert all(tok.decode([i]).strip().lower() == "no" for i in no)


def test_both_model_sizes_share_answer_ids():
    a, b = _tok("0.5b"), _tok("7b")
    for variants in (YES_VARIANTS, NO_VARIANTS):
        assert answer_token_ids(a, variants) == answer_token_ids(b, variants)


def test_paper_prompt_puts_each_image_before_its_sentence():
    body = PROMPTS["paper"]
    assert body.count(IMAGE) == 4
    # paper, Sec. IV-A: "<image1> This is a cross-sectional image of a peg. <image2> This is another ..."
    assert body.startswith(f"{IMAGE} This is a cross-sectional image of a peg.")
    assert body.count(f"{IMAGE} This is ") == 4
    assert body.endswith("Please answer with only yes or no.")
    assert CHAT.format(body=body).endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("prompt_id", sorted(PROMPTS))
def test_every_prompt_puts_each_image_before_its_sentence(prompt_id):
    body = PROMPTS[prompt_id]
    n = 2 * len(PROMPT_VIEWS[prompt_id])  # peg views + socket views
    assert body.count(IMAGE) == n
    assert body.startswith(IMAGE) and body.count(f"{IMAGE} This is ") == n
    assert body.endswith("Please answer with only yes or no.")


def test_hf_format_matches_the_checkpoint_chat_template():
    from transformers import AutoProcessor

    model_id, rev = wncf.MODELS["7b"]
    try:
        proc = AutoProcessor.from_pretrained(model_id, revision=rev, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} not downloaded")
    body = PROMPTS["paper"]
    content = [{"type": "image"}] * 4 + [{"type": "text", "text": body.replace(f"{IMAGE} ", "")}]
    rendered = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
    assert rendered == hf_template(body)


def test_nested_grouping_is_one_multi_image_sample():
    from PIL import Image
    from transformers import AutoProcessor

    model_id, rev = wncf.MODELS["0.5b"]
    try:
        proc = AutoProcessor.from_pretrained(model_id, revision=rev, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} not downloaded")
    imgs = [Image.new("RGB", (512, 512), "white") for _ in range(4)]
    text = CHAT.format(body=PROMPTS["paper"])
    nested = proc(images=[imgs], text=text, return_tensors="pt")
    flat = proc(images=imgs, text=text, return_tensors="pt")
    assert nested["batch_num_images"].tolist() == [4]  # no per-image anyres tiling
    assert flat["batch_num_images"].tolist() == [1, 1, 1, 1]
    assert nested["input_ids"].shape[1] < flat["input_ids"].shape[1] / 3
