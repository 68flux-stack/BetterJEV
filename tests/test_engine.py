"""Planned scoring and single-pass encoding against the fresh reference, with no downloads."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers  # noqa: E402

from semif_phase1.core import LETTERS, direct_messages  # noqa: E402
from semif_phase1.direct import encode_prompt, encode_prompts, score  # noqa: E402
from semif_phase1.engine import _batches, plan, score_planned  # noqa: E402

QWEN_SPLIT = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}"""
              r"""| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""")
TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and "
    "enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def build_tokenizer(corpus, template=TEMPLATE):
    """A Qwen-style byte-level BPE fast tokenizer with chat special tokens."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tokenizer.train_from_iterator(corpus, trainer)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<|im_end|>",
                                                pad_token="<|endoftext|>")
    fast.add_tokens(["<think>", "</think>"])
    fast.chat_template = template
    return fast


def make_row(key, state, question, count=2):
    return {"id": key, "state": state, "question": question,
            "options": [{"id": f"o{index}", "description": f"Choice {LETTERS[index]} applies."}
                        for index in range(count)]}


STATES = [
    "PRIMARY RECORD:\nThe courier confirmed delivery at 14:05 and attached a signed receipt.\n" * 3,
    {"ticket": "Refund requested twice", "amounts": [12.5, 40], "status": "open"},
    "The operator paused the deploy because the canary error rate doubled.",
]
QUESTIONS = ["Does the record confirm completion?", "Is a refund involved?", "Does anyone request review?",
             "Is the evidence contradictory?"]
ROWS = [make_row(f"s{s}-q{q}", state, question, count=2 + (s + q) % 3)
        for s, state in enumerate(STATES) for q, question in enumerate(QUESTIONS)]
ROWS += [make_row("lone", "A single unrelated record.", "Is it relevant?", count=16),
         dict(ROWS[0], id="repeat-of-first")]


@pytest.fixture(scope="module")
def tokenizer():
    corpus = [json.dumps(row) for row in ROWS] + [TEMPLATE] * 3
    return build_tokenizer(corpus)


@pytest.fixture(scope="module")
def model(tokenizer):
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=2, tie_word_embeddings=False,
    )
    network = transformers.Qwen3_5ForCausalLM(config).float().eval()
    with torch.no_grad():
        network.lm_head.weight.mul_(20)  # separate option logits beyond float32 noise
    return network


def reference_encoding(tokenizer, row):
    """The original three-pass check: one encode plus a full re-encode per answer letter."""
    prompt = tokenizer.apply_chat_template(direct_messages(row), tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    slots = [tokenizer.encode(letter, add_special_tokens=False)[0] for letter in LETTERS[: len(row["options"])]]
    assert all(tokenizer.encode(prompt + letter, add_special_tokens=False) == ids + [slot]
               for letter, slot in zip(LETTERS, slots))
    return ids, slots


def test_single_pass_encoding_matches_reference(tokenizer):
    encoded = encode_prompts(tokenizer, ROWS, 4096)
    assert [(ids, slots) for ids, slots, _ in encoded] == [reference_encoding(tokenizer, row) for row in ROWS]
    assert [encode_prompt(tokenizer, row, 4096) for row in ROWS] == encoded


def test_boundary_failures_are_never_memoized():
    template = "{% for message in messages %}<|im_start|>{{ message.content }}{% endfor %}<|im_end|>Answer:"
    merging = build_tokenizer(["Answer:A Answer:B Answer:C"] * 50, template)
    for key in ("first", "second"):
        with pytest.raises(ValueError, match="Answer boundary"):
            encode_prompt(merging, make_row(key, "state", "question"), 4096)


def test_plan_shares_only_worthwhile_prefixes():
    encoded = [([1] * 600 + [2, index], [7, 8], "h") for index in range(3)] + [([1] * 600 + [3], [7, 8], "h")]
    units = plan(encoded, ["a", "a", "a", "b"], max_batch_tokens=10_000, max_batch_rows=8, min_shared_tokens=512)
    assert [(unit.kind, unit.members, unit.prefix) for unit in units] == [("shared", [0, 1, 2], 601),
                                                                          ("fresh", [3], 0)]
    assert _batches([10, 10, 10, 9], [0, 1, 2, 3], 1000, 2) == [[0, 1], [2, 3]]  # row budget
    assert _batches([10, 10, 10, 9], [0, 1, 2, 3], 25, 8) == [[0, 1], [2, 3]]  # token budget
    assert _batches([10, 2, 2], [0, 1, 2], 1000, 8) == [[0], [1, 2]]  # padding budget


@pytest.mark.parametrize("budgets", [{}, {"max_batch_rows": 3, "max_batch_tokens": 1024}, {"max_batch_rows": 1}])
def test_planned_scores_match_fresh_scoring(tokenizer, model, budgets):
    fresh = {row["id"]: score(model, tokenizer, row, {}, 4096) for row in ROWS}
    shuffled = ROWS[1::2] + ROWS[::2]
    results, timing = score_planned(model, tokenizer, shuffled, {}, 4096, min_shared_tokens=16, **budgets)
    assert [result["id"] for result in results] == [row["id"] for row in shuffled]
    assert timing["shared_units"] == len(STATES)
    assert timing["prefill_tokens"] + timing["true_tokens"] < timing["fresh_equivalent_tokens"]
    for result in results:
        expected = fresh[result["id"]]
        assert result["prompt_sha256"] == expected["prompt_sha256"]
        assert result["input_tokens"] == expected["input_tokens"]
        assert result["option_logits"] == pytest.approx(expected["option_logits"], abs=1e-4)
        assert 0 < result["allowed_token_mass"] <= 1
    # Whichever copy comes first in the input is computed; the other points at it.
    pair = {result["id"]: result["plan"]["duplicate_of"] for result in results
            if result["id"] in {"repeat-of-first", ROWS[0]["id"]}}
    computed = next(key for key, source in pair.items() if source is None)
    assert set(pair.values()) == {None, computed}


def test_planned_scoring_rejects_duplicate_ids(tokenizer, model):
    with pytest.raises(ValueError, match="unique"):
        score_planned(model, tokenizer, [ROWS[0], ROWS[0]], {}, 4096)
