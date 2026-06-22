# Model Card — Zeus (Intent Router)

> Illustrative reference model from the MSC Labs eval harness. Reference weights and
> datasets are not distributed. All metrics are illustrative reference results.

## Model details

- **Developed by:** MSC Labs
- **Model type:** Decoder-only LLM fine-tuned for single-label intent classification (text routing)
- **Base model:** `google/gemma-2-2b-it` (2B parameters)
- **Adapter:** LoRA / QLoRA adapter (`msc-labs/zeus-intent-router-qlora`, illustrative id)
- **Language:** English
- **License:** Apache-2.0 (code). Base model under the Gemma Terms of Use.
- **Finetuned from:** `google/gemma-2-2b-it`

## Intended use

Zeus is a hot-path **intent router** for agent and support systems. Given one inbound user
message, it returns exactly one intent label and a confidence score, so an orchestrator can
dispatch the message to the correct skill, queue, or sub-agent.

Designed for:

- First-touch routing of support chats, emails, and form submissions.
- Triage in front of a multi-agent system (which specialist handles this?).
- Deflection / escalation gating (route to a human when uncertain).

### Intent label set (8 classes)

| Label | When it fires |
|---|---|
| `billing` | Charges, invoices, refunds, payment methods, pricing disputes. |
| `technical_support` | Bugs, errors, outages, "it's not working," how-to troubleshooting. |
| `sales` | Pre-purchase questions, plan comparisons, demos, upgrades, quotes. |
| `account` | Login, password, profile, permissions, account changes, cancellation. |
| `order_status` | Where is my order / shipment, delivery dates, tracking. |
| `feedback` | Praise, complaints, feature requests, general commentary with no action needed. |
| `escalate_human` | Ambiguous, sensitive, legal/safety, or multi-issue messages that need a person. |
| `other` | Out-of-scope, spam, greetings-only, or messages that fit no other label. |

## Out-of-scope use

- **Not a multi-label classifier.** It returns one intent. Messages that genuinely span two
  intents are trained to route to `escalate_human`, not to split.
- **Not a content-safety or moderation model.** It does not detect abuse, PII, or policy
  violations; pair it with a dedicated filter.
- **Taxonomy-bound.** Accuracy is reported against *this* 8-label taxonomy. Re-tuning is
  required for a different label set; the published numbers do not transfer.
- **English-only** in this reference. Non-English inputs are out of distribution.

## Training data

Illustrative dataset of **24,000** labeled routing examples:

- **~70% synthetic**, generated with a frontier model and templated paraphrase expansion
  across the 8 intents, then deduplicated.
- **~30% curated** from anonymized, relabeled public support-style message corpora.
- Human-reviewed label audit on a 1,500-example sample (inter-annotator agreement κ = 0.91).
- Split: 20,000 train / 2,000 validation / 2,000 test (stratified by intent).

Each row is shaped as:

```json
{"message": "I was charged twice this month", "intent": "billing", "confidence": 0.99}
```

See [`data/README.md`](data/README.md) for the class balance and taxonomy.

## Training procedure

- **Method:** QLoRA (4-bit NF4 base, LoRA adapters on attention + MLP projections).
- **Objective:** supervised fine-tuning to emit a single label token given a routing prompt.
- **Protocol:** 2-stage — a smoke-test run on 1% of data to validate the pipeline, then the
  full run. See [`training/config.yaml`](training/config.yaml) and
  [`training/train.py`](training/train.py).
- **Hardware (reference):** 1× A100 40GB, ~1.4 GPU-hours for the full run.

## Evaluation

- **Test set:** 2,000 held-out messages, stratified across the 8 intents.
- **Baseline:** GPT-4o (`gpt-4o-2024-08-06`), same label set, same few-shot prompt, temp 0.
- **Headline:** 99.1% routing accuracy, 0.985 macro-F1 (vs GPT-4o 97.8% / 0.964).
- **Serving:** constrained decoding (logit mask over the 8 label tokens).

Full methodology, per-class precision/recall, cost, and latency in
[`eval/results.md`](eval/results.md) and [`eval/results.json`](eval/results.json).

## Limitations and risks

- Performance degrades on intents under-represented in training; `feedback` and `other` are
  the hardest classes (see per-class table).
- A mis-route is silent: the model is confident on a wrong label only when the input is far
  out of distribution. Use the confidence score and an `escalate_human` floor in production.
- The taxonomy encodes one business's routing decisions; it is not universal.
- Reported numbers are illustrative reference results, not a production SLA.

## How to cite

```
MSC Labs (2026). Zeus: a QLoRA-tuned Gemma-2-2B intent router.
Reference model. https://labs.msccompany.com.br
```

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://labs.msccompany.com.br/assessment
> Numbers are illustrative reference results from our standard eval harness.
