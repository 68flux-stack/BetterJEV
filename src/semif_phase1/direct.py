"""Direct categorical decision readout from native next-token logits."""

from __future__ import annotations

import inspect
import time
import weakref

from .core import LETTERS, digest, direct_messages, softmax, synchronize

PROMPT_VERSION = "direct-options-v1"

# Per-tokenizer memo of verified answer boundaries; see _boundary_tail for why reuse is exact.
_MEMOS = weakref.WeakKeyDictionary()
# Only short template tails are memoized, so row content never accumulates in the memo.
_MAX_MEMO_TAIL = 256


def _memo(tokenizer) -> dict | None:
    try:
        return _MEMOS.setdefault(tokenizer, {})
    except TypeError:
        return None


def _slot_ids(tokenizer, count: int) -> list[int]:
    memo = _memo(tokenizer)
    if memo is not None and ("slots", count) in memo:
        return list(memo[("slots", count)])
    result = []
    for letter in LETTERS[:count]:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.decode(encoded) != letter:
            raise ValueError(f"Answer slot {letter!r} is not one exact round-trip token")
        result.append(encoded[0])
    if len(result) != len(set(result)):
        raise ValueError("Answer-slot tokens collide")
    if memo is not None:
        memo[("slots", count)] = list(result)
    return result


def _forward(model, inputs):
    parameters = inspect.signature(model.forward).parameters
    kwargs = dict(inputs, use_cache=False, return_dict=True)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = 1
    return model(**kwargs).logits[:, -1, :]


def _render(tokenizer, row: dict) -> str:
    return tokenizer.apply_chat_template(
        direct_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _boundary_tail(tokenizer, prompt: str, ids: list[int], offsets) -> str | None:
    """Return the prompt text from its last added token onward, when that alone decides the boundary.

    Fast tokenizers split out added tokens before normalizing, pre-tokenizing and merging each
    remaining segment independently. Appending a letter that no added token contains therefore
    changes only the final segment, so the boundary check depends only on this tail.
    """
    if offsets is None or not getattr(tokenizer, "is_fast", False):
        return None
    added = {index: text for text, index in tokenizer.get_added_vocab().items()}
    for position in range(len(ids) - 1, -1, -1):
        if ids[position] in added:
            start = offsets[position][0]
            tail = prompt[start:]
            return tail if prompt.startswith(added[ids[position]], start) and len(tail) <= _MAX_MEMO_TAIL else None
    return None


def _verify_encoding(tokenizer, row: dict, prompt: str, ids: list[int], offsets, max_tokens: int):
    if not ids or len(ids) > max_tokens:
        raise ValueError(f"Row {row['id']}: {len(ids)} input tokens exceed limit {max_tokens}; no truncation allowed")
    slots = _slot_ids(tokenizer, len(row["options"]))
    memo = _memo(tokenizer)
    tail = _boundary_tail(tokenizer, prompt, ids, offsets) if memo is not None else None
    added = "".join(tokenizer.get_added_vocab()) if tail is not None else ""
    for letter, token in zip(LETTERS, slots):
        key = ("boundary", tail, letter, token)
        if tail is not None and letter not in added and key in memo:
            continue
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [token]:
            raise ValueError(f"Answer boundary changes tokenization for slot {letter}")
        if tail is not None and letter not in added:
            memo[key] = True
    return ids, slots, digest(prompt)


def encode_prompt(tokenizer, row: dict, max_tokens: int) -> tuple[list[int], list[int], str]:
    """Encode one decision and verify its single-token answer slots."""
    return encode_prompts(tokenizer, [row], max_tokens)[0]


def encode_prompts(tokenizer, rows: list[dict], max_tokens: int) -> list[tuple[list[int], list[int], str]]:
    """Encode decisions with one tokenizer pass each, batched when the tokenizer supports it."""
    prompts = [_render(tokenizer, row) for row in rows]
    # Wrappers (for example MLX-LM's) may forward is_fast without being callable themselves.
    if callable(tokenizer) and getattr(tokenizer, "is_fast", False) is True:
        batch = tokenizer(prompts, add_special_tokens=False, return_offsets_mapping=True)
        encodings = zip(batch["input_ids"], batch["offset_mapping"])
    else:
        encodings = ((tokenizer.encode(prompt, add_special_tokens=False), None) for prompt in prompts)
    return [
        _verify_encoding(tokenizer, row, prompt, list(ids), offsets, max_tokens)
        for row, prompt, (ids, offsets) in zip(rows, prompts, encodings)
    ]


def score(model, tokenizer, row: dict, metadata: dict, max_tokens: int = 4096) -> dict:
    import torch

    started = time.perf_counter()
    ids, slots, prompt_hash = encode_prompt(tokenizer, row, max_tokens)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": torch.tensor([ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(ids)), dtype=torch.long, device=device),
    }
    synchronize(device)
    forward_start = time.perf_counter()
    with torch.inference_mode():
        vocabulary = _forward(model, inputs)[0].float()
    synchronize(device)
    selected = vocabulary[slots].cpu().tolist()
    return {
        "id": row["id"],
        "option_ids": [option["id"] for option in row["options"]],
        "probabilities": softmax(selected),
        "option_logits": selected,
        "input_tokens": len(ids),
        "forward_seconds": time.perf_counter() - forward_start,
        "total_seconds": time.perf_counter() - started,
        "prompt_sha256": prompt_hash,
        "prompt_version": PROMPT_VERSION,
        "model": metadata,
        "readout": "native full-vocabulary last-position logits restricted to declared answer slots",
        "probability_status": "conditional option score; uncalibrated as decision confidence",
    }
