"""Measure planned (auto) scoring against fresh and parallel shared scoring on the owned 37x21 fixture.

The fixture rows are shuffled with a fixed seed before auto scoring so the planner, not input
order, must recover each state's shared prefix.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from semif_phase1.core import load_causal_model
from semif_phase1.direct import score
from semif_phase1.engine import score_planned
from semif_phase1.shared import score_shared


def top(row):
    return max(range(len(row["probabilities"])), key=row["probabilities"].__getitem__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-batch-tokens", type=int, default=8192)
    parser.add_argument("--max-batch-rows", type=int, default=32)
    parser.add_argument("--skip-fresh", action="store_true", help="Skip the ~6 minute fresh reference pass")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".predictions.jsonl").exists():
        parser.error("Output must be new")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if len(rows) != 777 or len(groups) != 37 or any(len(group) != 21 for group in groups.values()):
        parser.error("Expected the committed 37-state x 21-question fixture")
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, "cuda")
    import torch

    budgets = {"max_batch_tokens": args.max_batch_tokens, "max_batch_rows": args.max_batch_rows}
    first = next(iter(groups.values()))
    score(model, tokenizer, first[0], metadata, args.max_tokens)
    score_shared(model, tokenizer, first, metadata, args.max_tokens)
    score_planned(model, tokenizer, first, metadata, args.max_tokens, **budgets)
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)
    report = {
        "version": "shape777-auto-v1",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": metadata,
        "hardware": torch.cuda.get_device_name(0),
        "budgets": budgets,
        "timing_scope": "Warm model; includes prompt construction, tokenization, transfers, forward passes and CPU readout.",
        "results": [],
    }
    predictions = {}
    modes = ("parallel_shared", "planned_auto") if args.skip_fresh else ("fresh", "parallel_shared", "planned_auto")
    for mode in modes:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        extra = {}
        if mode == "fresh":
            values = [score(model, tokenizer, row, metadata, args.max_tokens) for row in rows]
        elif mode == "parallel_shared":
            values, state_times = [], []
            for group in groups.values():
                scored, timing = score_shared(model, tokenizer, group, metadata, args.max_tokens)
                values.extend(scored)
                state_times.append(timing["total_seconds"])
            extra["state_p50_seconds"] = statistics.median(state_times)
        else:
            values, timing = score_planned(model, tokenizer, shuffled, metadata, args.max_tokens, **budgets)
            extra["plan"] = timing
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        predictions[mode] = values
        report["results"].append({
            "mode": mode,
            "wall_seconds": elapsed,
            "decisions_per_second": len(values) / elapsed,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            **extra,
        })
    reference_mode = "parallel_shared" if args.skip_fresh else "fresh"
    reference = {row["id"]: row for row in predictions[reference_mode]}
    report["comparisons"] = {"reference": reference_mode}
    for mode in modes:
        if mode == reference_mode:
            continue
        differences = [
            abs(a - b)
            for row in predictions[mode]
            for a, b in zip(row["probabilities"], reference[row["id"]]["probabilities"])
        ]
        report["comparisons"][mode] = {
            "max_probability_difference": max(differences),
            "mean_probability_difference": statistics.fmean(differences),
            "argmax_flips": [row["id"] for row in predictions[mode] if top(row) != top(reference[row["id"]])],
            "prompt_hashes_match": all(
                row["prompt_sha256"] == reference[row["id"]]["prompt_sha256"] for row in predictions[mode]
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with args.output.with_suffix(".predictions.jsonl").open("x") as stream:
        for mode, values in predictions.items():
            for row in values:
                stream.write(json.dumps({"mode": mode, **row}, allow_nan=False) + "\n")
    print(json.dumps(report["results"]))


if __name__ == "__main__":
    main()
