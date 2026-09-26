# Performance: where the time goes and what `--mode auto` changes

This note decomposes the committed RTX 3090 timings, explains the planned scorer, and marks
which numbers are measured and which are projections awaiting a CUDA run. It does not change
any published claim; the headline table in the README remains the committed evidence.

## Time budget of the committed runs

Source: `results/raw/shape777-direct.json` and its row-level predictions (Qwen3.5-4B, BF16,
RTX 3090, 37 states × 21 decisions, ~1,842 prompt tokens per decision, ~1,765 of them state).

| Component | Measured | Share |
|---|---:|---|
| Fresh decision (full prompt forward) | 0.43 s | ~99% forward |
| State prefill, once per state | 0.406 s | 39% of a parallel state |
| Serial suffix forward, ~77 tokens, batch 1 | 0.058 s | launch/latency bound |
| Serial per-row CPU (encode + readout) | ~0.010 s | outside the forward timers |
| Parallel state total (prefill + 21 batched suffixes) | 1.05 s | — |

Three facts follow:

1. **Work is dominated by re-reading the state.** Fresh scoring processes 1.43 M tokens for
   777 decisions; one prefill per state plus suffixes processes ~0.12 M. That 11× token
   reduction is the source of the published 8.6× parallel speedup, and no kernel change can
   rival it for shared-state workloads.
2. **Tokenization was a large fraction of the fastest path.** The previous `encode_prompt`
   tokenized each ~1,800-token prompt three times (once, then once more per answer letter to
   check the answer boundary), and shared mode tokenized the state a fourth time for its
   prefix. Measured with a Qwen-style byte-level BPE on the fixture: 8.2 ms per decision, or
   ~0.18 s of each 1.05 s parallel state.
3. **The GPU is not near its BF16 peak.** At ~7.2 GFLOP per token, a fresh 1,842-token
   forward needs ~13 TFLOP; 0.43 s is ~30 TFLOP/s against the 3090's ~71 TFLOP/s dense BF16
   tensor peak. The pinned environment has no `flash-linear-attention`, so the Gated DeltaNet
   mixer in 24 of 32 layers runs Transformers' reference PyTorch path: FP32, a Python loop
   over 64-token chunks, and a batched triangular solve. Even on CPU, where every matmul runs
   at the same FP32 rate, that path is 22% of a real-width layer stack at 1,842 tokens; on
   the 3090 the BF16 GEMMs move to tensor cores and the reference path's share grows.

## What changed

### Single-pass exact tokenization (all modes)

`encode_prompt` now tokenizes each prompt once. The answer-boundary check (appending each
answer letter must add exactly that letter's token) is decided only by the text after the
last added token — `</think>\n\n` for Qwen — because fast tokenizers split added tokens out
before normalizing, pre-tokenizing and merging each segment independently. The first prompt
with a given tail runs the full check; later prompts with the same short tail reuse it. The
reuse is skipped when a letter occurs inside an added token, when offsets are unavailable, or
when the tail is long. Failures are never memoized. Batches go through the tokenizer's
parallel batch encoder.

Measured on all 1,032 fixture rows: identical token IDs, answer slots and prompt hashes; CPU
time falls from 7.1 to 2.6 ms per row one at a time and to about 1 ms per row batched
(0.85–1.26 ms across runs on 4 CPU cores).

### Planned scoring: `--mode auto`

`semif-score --mode auto` accepts any input (mixed states, any order) and plans it:

1. Encode every row once, batched.
2. Compute identical prompts once.
3. Group rows by exact serialized state and take the **exact longest common token prefix** of
   each group, computed from the rows' own token IDs (so it also covers
   `", "criterion": "`, not only the state). A prefix is shared when it saves at least
   512 tokens.
4. Prefill each shared prefix once and branch its native cache into right-padded suffix
   batches (`--max-batch-rows`, `--max-batch-tokens`, at most 25% padding per batch).
5. Batch the remaining rows fresh with right padding under the same budgets. Left padding
   would feed pad positions into the hybrid model's convolution and recurrent state, so it is
   never used.
6. Reduce each batch's readout on device (selected option logits, allowed-token mass and
   full-vocabulary argmax) and move it to the host in one transfer.

Every row is still scored on exactly its own full prompt token sequence. Only the execution
schedule and batch shapes change. Results are written in input order and carry
`serving_config: planned-exact-prefix-v1` plus a `plan` record (unit, prefix tokens, padded
tokens, timings).

Correctness evidence (CPU, float32, random-weight Qwen3.5 hybrid model, `tests/test_engine.py`):
option logits match fresh scoring within 2.4e-6 with identical prompt hashes and no argmax
changes. This holds for shuffled input, duplicates, JSON states, 16-option rows and every
batching budget. In BF16 on CUDA, the documented kernel-path drift of prefix reuse (5–6 of
777 argmaxes in the committed run) should be expected here too.

Token accounting on the full fixture (local Qwen-style tokenizer): fresh 1,343,229 tokens,
existing parallel mode 119,569, planned 116,609 (the exact LCP adds ~4 shared tokens per row).

### Optional CUDA kernels

`pip install -e '.[cuda-kernels]'` installs `flash-linear-attention` 0.5.2. Transformers then
dispatches Qwen3.5's gated delta rule to its Triton kernels automatically. Every torch-backend
result records `model.optional_kernels` (for example
`{"flash-linear-attention": "0.5.2", "causal-conv1d": null}`), so kernel choice is auditable
row by row. The loader refuses to run those CUDA-only kernels on CPU or MPS. Kernel outputs
differ from the FP32 reference path at BF16 noise level; compare against committed row-level
predictions before publishing any claim.

## Projection for the 37×21 fixture (not yet measured on CUDA)

| Path | Basis | Decisions/s |
|---|---|---:|
| Fresh direct (committed) | measured | 2.33 |
| Parallel shared (committed) | measured | 20.0 |
| `--mode auto`, reference kernels | prefill 0.406 s + suffix ~0.46 s + CPU ~0.02 s per state | ~24 (projected) |
| `--mode auto` + `cuda-kernels` | depends on the delta-rule share of GPU time | unmeasured |

Relative to the default `--mode direct` path, planned scoring is projected at roughly 10×
on shared-state workloads. Against the already-optimized parallel path, the remaining exact
gains come from CPU work (tokenization), the slightly longer exact prefix, and batching rows
whose states differ.

Measure on one GPU with a new output path:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/shape777_auto.py \
  --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --input benchmarks/data/shape777.jsonl \
  --output shape777-auto-run.json
```

The runner shuffles the fixture before auto scoring, so the planner rather than input order
must recover the groups. It reports wall time, peak memory, argmax flips, probability drift and
prompt-hash agreement against fresh scoring. Repeat it in an environment with
`.[cuda-kernels]` to measure the kernel change separately.
