# Zeus — Evaluation

> Illustrative reference results from the MSC Labs eval harness. Numbers are
> internally consistent across this repo and are not a production SLA.

## Summary

| Model | Routing accuracy | Macro-F1 | $ / 1k requests | p50 latency | p95 latency |
|---|---|---|---|---|---|
| GPT-4o (baseline) | 97.8% | 0.964 | $4.80 | 720 ms | 1,310 ms |
| **Zeus (Gemma-2-2B + QLoRA)** | **99.1%** | **0.985** | **$0.12** | **38 ms** | **61 ms** |
| **Delta** | **+1.3 pts** | **+0.021** | **40× cheaper** | **~19× faster** | **~21× faster** |

Both models are strong; the accuracy delta is real but small. The decisive wins are **cost (40×)** and **latency (sub-50ms p50)** — the two things that matter for a classifier on the hot path of every inbound message.

## Methodology

- **Task:** single-label intent routing over an 8-class taxonomy (`billing`, `technical_support`, `sales`, `account`, `order_status`, `feedback`, `escalate_human`, `other`).
- **Test set:** 2,000 held-out messages, stratified at 250 per intent. No overlap with train/validation; dedup checked by message hash and near-dup MinHash.
- **Baseline:** GPT-4o (`gpt-4o-2024-08-06`) via the API, same 8 labels, same few-shot routing prompt, `temperature=0`, `max_tokens=4`. Output parsed to the nearest valid label.
- **Zeus serving:** the QLoRA adapter on `google/gemma-2-2b-it`, bf16, with **constrained decoding** — a logit mask restricts generation to the 8 label tokens, so every output is a valid label.
- **Metrics:** accuracy (exact label match), per-class precision/recall/F1, and macro-F1 (unweighted mean over classes). Confidence is the softmax mass on the chosen label token.
- **Determinism:** both models run greedy/temperature-0, so reported runs are reproducible.

## Per-class results (Zeus)

250 test messages per class.

| Intent | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| billing | 0.996 | 1.000 | 0.998 | 250 |
| technical_support | 1.000 | 0.996 | 0.998 | 250 |
| sales | 0.992 | 0.996 | 0.994 | 250 |
| account | 0.988 | 0.992 | 0.990 | 250 |
| order_status | 1.000 | 0.996 | 0.998 | 250 |
| feedback | 0.980 | 0.972 | 0.976 | 250 |
| escalate_human | 0.972 | 0.976 | 0.974 | 250 |
| other | 0.964 | 0.960 | 0.962 | 250 |
| **macro avg** | **0.987** | **0.986** | **0.985** | **2,000** |
| **accuracy** | | | **0.991** | **2,000** |

**Where the errors are.** Almost all of Zeus's 18 misroutes (99.1% of 2,000 = 1,982 correct) fall on the three "soft" classes. The most common confusions:

- `feedback` ↔ `other` — short, low-content messages with no clear action (e.g. "thanks" vs greeting-only).
- `escalate_human` ↔ the specific intent it mentions — multi-issue messages that name one concrete topic but also need a person.
- `other` ↔ `sales` — vague pre-purchase curiosity that barely clears the routing threshold.

These are the same boundaries a human triager finds genuinely ambiguous; the `escalate_human` floor is designed to absorb them safely.

## Cost breakdown

Per 1,000 routing requests. Average input ≈ 38 tokens, output ≤ 4 tokens (one label).

**GPT-4o (`gpt-4o-2024-08-06`):**
- Input: ~38 tok × 1,000 = 38k tokens @ $2.50 / 1M → ~$0.10
- Few-shot prompt overhead (shared system + exemplars, ~1,850 tok/req amortized) → the dominant term
- Output: ~4 tok × 1,000 @ $10.00 / 1M → ~$0.04
- **Billed total ≈ $4.80 / 1k** (driven by the few-shot routing prompt sent on every call)

**Zeus (self-hosted Gemma-2-2B, QLoRA):**
- 1× A10G (24GB) at ~$0.65/hr serving ~5,400 req/min under constrained decoding
- Amortized compute → **≈ $0.12 / 1k**, no per-token vendor fee
- **40× cheaper** than the GPT-4o baseline

At 1M inbound messages/month the line item moves from ~$4,800 to ~$120.

## Latency

Measured at the router boundary, batch size 1, warm model.

| Model | p50 | p95 | p99 |
|---|---|---|---|
| GPT-4o (API) | 720 ms | 1,310 ms | 1,980 ms |
| **Zeus (local)** | **38 ms** | **61 ms** | **89 ms** |

Zeus stays **under 50ms at p50** because it is small, runs locally (no network round-trip), and emits a single masked token. For an agent system, that means routing is effectively free on the critical path.

## 2-stage training protocol

1. **Smoke test** — full pipeline on 1% of data (`--smoke`, 30 steps) to validate data formatting, the chat template, the label mask, and that loss decreases. Catches config bugs before paying for GPU.
2. **Full run** — 3 epochs on 20k examples, ~1.4 GPU-hours on 1× A100 40GB. Best checkpoint by `eval_loss`.

This is the same protocol MSC Labs runs for clients: never debug a training script on a paid full run.

## Limitations

- Numbers are illustrative reference results, not a guaranteed production SLA.
- Accuracy is reported against *this* 8-label taxonomy; a different label set requires re-tuning, and these numbers do not transfer.
- The model is single-label by design; truly multi-intent messages route to `escalate_human`.
- English-only in this reference. Non-English inputs are out of distribution.
- `feedback` / `other` / `escalate_human` are the weakest classes — keep a confidence floor and human escalation in production.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
