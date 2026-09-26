"""Planned scoring: tokenize once, prefill each shared token prefix once, batch every suffix.

Fresh scoring spends almost all of its work re-reading the same state for every decision.
This planner computes the exact longest common token prefix of all decisions that carry one
state, runs that prefix once, and scores the remaining suffixes as right-padded batches that
branch from its native cache. Decisions with no worthwhile shared prefix are batched fresh.
Every row still sees exactly its own full prompt tokens; only the execution schedule changes.
"""

from __future__ import annotations

import copy
import inspect
import json
import time
from dataclasses import dataclass

from .core import softmax, synchronize
from .direct import PROMPT_VERSION, encode_prompts
from .shared import _suffix_layout

SERVING_CONFIG = "planned-exact-prefix-v1"


@dataclass
class Unit:
    kind: str  # "shared" prefills members[*][:prefix] once; "fresh" runs full prompts
    members: list[int]  # indices into the unique-prompt list
    prefix: int = 0


def _common_prefix(sequences: list[list[int]]) -> int:
    first, length = sequences[0], min(map(len, sequences))
    for sequence in sequences[1:]:
        if sequence[:length] != first[:length]:
            length = next(index for index in range(length) if sequence[index] != first[index])
    return length


def _batches(lengths: list[int], order: list[int], max_tokens: int, max_rows: int,
             max_padding: float = 0.25) -> list[list[int]]:
    """Cut a length-sorted member list into right-padded batches within row, token and padding budgets."""
    batches, current, width, true = [], [], 0, 0
    for member in order:
        grown, count = max(width, lengths[member]), len(current) + 1
        if current and (count > max_rows or count * grown > max_tokens
                        or count * grown > (1 + max_padding) * (true + lengths[member])):
            batches.append(current)
            current, grown, true = [], lengths[member], 0
        current.append(member)
        width, true = grown, true + lengths[member]
    return batches + ([current] if current else [])


def plan(encoded: list[tuple], states: list[str], *, max_batch_tokens: int, max_batch_rows: int,
         min_shared_tokens: int) -> list[Unit]:
    """Group unique prompts by exact serialized state; share prefixes only where they pay off."""
    groups: dict[str, list[int]] = {}
    for index, state in enumerate(states):
        groups.setdefault(state, []).append(index)
    units, fresh = [], []
    for members in groups.values():
        sequences = [encoded[member][0] for member in members]
        # Keep at least one suffix token per decision so every branch produces its own logits.
        prefix = min(_common_prefix(sequences), min(map(len, sequences)) - 1) if len(members) > 1 else 0
        if (len(members) - 1) * prefix >= min_shared_tokens:
            units.append(Unit("shared", members, prefix))
        else:
            fresh.extend(members)
    lengths = [len(ids) for ids, _, _ in encoded]
    longest_first = sorted(fresh, key=lambda member: -lengths[member])
    units.extend(Unit("fresh", batch) for batch in _batches(lengths, longest_first, max_batch_tokens, max_batch_rows))
    return sorted(units, key=lambda unit: min(unit.members))


def _readout(logits, keep: list[int], ends: list[int], slot_lists: list[list[int]]):
    """Gather each row's end-position vocabulary and reduce it on device before one transfer."""
    import torch

    device = logits.device
    rows = torch.arange(len(ends), device=device)
    columns = torch.tensor([keep.index(end) for end in ends], dtype=torch.long, device=device)
    vocabulary = logits[rows, columns].float()
    width = max(map(len, slot_lists))
    slots = torch.tensor([slots + [slots[0]] * (width - len(slots)) for slots in slot_lists], device=device)
    valid = torch.tensor([[index < len(slots) for index in range(width)] for slots in slot_lists], device=device)
    selected = vocabulary.gather(1, slots)
    mass = (selected.masked_fill(~valid, float("-inf")).logsumexp(-1) - vocabulary.logsumexp(-1)).exp()
    packed = torch.cat([selected, mass[:, None], vocabulary.argmax(-1, keepdim=True).float()], 1).cpu().tolist()
    return [
        (values[: len(slot_list)], values[width], int(values[width + 1]))
        for values, slot_list in zip(packed, slot_lists)
    ]


def iter_planned(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096, *,
                 max_batch_tokens: int = 8192, max_batch_rows: int = 32, min_shared_tokens: int = 512):
    """Yield (row index, result) pairs unit by unit; each row's result matches its own full prompt."""
    import torch

    if max_batch_tokens < 1 or max_batch_rows < 1 or min_shared_tokens < 0:
        raise ValueError("Batch budgets must be positive and min_shared_tokens nonnegative")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Decision IDs must be unique")
    started = time.perf_counter()
    encoded = encode_prompts(tokenizer, rows, max_tokens)
    # Identical prompts with identical answer slots are one computation.
    unique, owners = {}, []
    for index, (ids, slots, _) in enumerate(encoded):
        owners.append(unique.setdefault((tuple(ids), tuple(slots)), len(unique)))
    first = {}
    for index, owner in enumerate(owners):
        first.setdefault(owner, index)
    prompts = [encoded[first[owner]] for owner in range(len(unique))]
    states = [json.dumps(rows[first[owner]]["state"], ensure_ascii=False) for owner in range(len(unique))]
    units = plan(prompts, states, max_batch_tokens=max_batch_tokens, max_batch_rows=max_batch_rows,
                 min_shared_tokens=min_shared_tokens)
    encode_seconds = time.perf_counter() - started
    rows_of = {}
    for index, owner in enumerate(owners):
        rows_of.setdefault(owner, []).append(index)

    device = next(model.parameters()).device
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" not in parameters and hasattr(model, "get_base_model"):
        parameters = inspect.signature(model.get_base_model().forward).parameters
    if "logits_to_keep" not in parameters:
        raise RuntimeError("Model lacks selective-position logits needed by planned scoring")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("Tokenizer requires a padding or EOS token")
    tensor = lambda value: torch.tensor(value, dtype=torch.long, device=device)
    model.eval()
    for number, unit in enumerate(units):
        synchronize(device)
        mark = time.perf_counter()
        readouts, prefill_seconds, padded = {}, 0.0, 0
        with torch.inference_mode():
            if unit.kind == "shared":
                prefix = prompts[unit.members[0]][0][: unit.prefix]
                output = model(input_ids=tensor([prefix]), attention_mask=torch.ones((1, len(prefix)), dtype=torch.long,
                               device=device), use_cache=True, return_dict=True, logits_to_keep=1)
                cache = output.past_key_values
                del output
                if cache is None or cache.get_seq_length() != len(prefix):
                    raise RuntimeError("Invalid native prefix cache")
                if not callable(getattr(cache, "reorder_cache", None)):
                    raise RuntimeError("Native cache does not support duplicate branch selection")
                synchronize(device)
                prefill_seconds = time.perf_counter() - mark
                lengths = [len(ids) - unit.prefix for ids, _, _ in prompts]
                order = sorted(unit.members, key=lambda member: lengths[member])
                chunks = _batches(lengths, order, max_batch_tokens, max_batch_rows)
                for position, chunk in enumerate(chunks):
                    # The last chunk may consume the prefix cache; earlier ones branch from a copy.
                    branch = cache if position == len(chunks) - 1 else copy.deepcopy(cache)
                    branch.reorder_cache(torch.zeros(len(chunk), dtype=torch.long, device=device))
                    layout, ends = _suffix_layout([prompts[member][0][unit.prefix :] for member in chunk],
                                                  unit.prefix, pad)
                    keep = sorted(set(ends))
                    output = model(**{key: tensor(value) for key, value in layout.items()}, past_key_values=branch,
                                   use_cache=True, return_dict=True, logits_to_keep=tensor(keep))
                    values = _readout(output.logits, keep, ends, [prompts[member][1] for member in chunk])
                    readouts.update(zip(chunk, values))
                    padded += len(chunk) * len(layout["input_ids"][0])
                    del output, branch
                del cache
            else:
                sequences = [prompts[member][0] for member in unit.members]
                width = max(map(len, sequences))
                # Right padding only: real tokens never see later pads under causal or recurrent mixing.
                # (Left pads would enter hybrid models' convolution and recurrent state.)
                inputs = {
                    "input_ids": tensor([ids + [pad] * (width - len(ids)) for ids in sequences]),
                    "attention_mask": tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids in sequences]),
                }
                ends = [len(ids) - 1 for ids in sequences]
                keep = sorted(set(ends))
                # A lone prompt uses the same last-position request as direct scoring.
                output = model(**inputs, use_cache=False, return_dict=True,
                               logits_to_keep=1 if len(sequences) == 1 else tensor(keep))
                values = _readout(output.logits, keep, ends, [prompts[member][1] for member in unit.members])
                readouts.update(zip(unit.members, values))
                padded = len(sequences) * width
                del output
        synchronize(device)
        unit_timing = {
            "unit": number,
            "kind": unit.kind,
            "decisions": len(unit.members),
            "prefix_tokens": unit.prefix,
            "true_tokens": sum(len(prompts[member][0]) - unit.prefix for member in unit.members),
            "padded_tokens": padded,
            "prefill_seconds": prefill_seconds,
            "seconds": time.perf_counter() - mark,
        }
        for member in unit.members:
            selected, mass, argmax = readouts[member]
            ids, slots, prompt_hash = prompts[member]
            for index in rows_of[member]:
                row = rows[index]
                yield index, {
                    "id": row["id"],
                    "option_ids": [option["id"] for option in row["options"]],
                    "probabilities": softmax(selected),
                    "option_logits": selected,
                    "answer_token_ids": slots,
                    "input_tokens": len(ids),
                    "prompt_sha256": prompt_hash,
                    "prompt_version": PROMPT_VERSION,
                    "model": {**metadata, "serving_config": SERVING_CONFIG},
                    "readout": "native full-vocabulary end-position logits restricted to declared answer slots",
                    "probability_status": "conditional option score; uncalibrated as decision confidence",
                    "allowed_token_mass": mass,
                    "full_vocab_argmax_id": argmax,
                    "plan": {**unit_timing, "duplicate_of": None if index == first[member] else rows[first[member]]["id"],
                             "encode_seconds": encode_seconds},
                }


def score_planned(model, tokenizer, rows: list[dict], metadata: dict, max_tokens: int = 4096, **budgets):
    """Return results in input order plus whole-run timing."""
    started = time.perf_counter()
    results = [None] * len(rows)
    for index, result in iter_planned(model, tokenizer, rows, metadata, max_tokens, **budgets):
        results[index] = result
    units = {result["plan"]["unit"]: result["plan"] for result in results}
    timing = {
        "total_seconds": time.perf_counter() - started,
        "encode_seconds": results[0]["plan"]["encode_seconds"] if results else 0.0,
        "units": len(units),
        "shared_units": sum(unit["kind"] == "shared" for unit in units.values()),
        "prefill_tokens": sum(unit["prefix_tokens"] for unit in units.values()),
        "true_tokens": sum(unit["true_tokens"] for unit in units.values()),
        "padded_tokens": sum(unit["padded_tokens"] for unit in units.values()),
        "fresh_equivalent_tokens": sum(result["input_tokens"] for result in results),
    }
    return results, timing
