# Zeus — Dataset

Illustrative reference dataset for the Zeus intent router. The full corpus is **not
distributed**; [`sample.jsonl`](sample.jsonl) holds a representative slice so the schema
and label taxonomy are reproducible.

## Schema

One JSON object per line:

```json
{"message": "<one inbound user message>", "intent": "<one label>", "confidence": <float 0-1>}
```

- `message` — a single inbound message: a support chat line, a form submission, or an email subject+body.
- `intent` — exactly one label from the 8-class taxonomy below.
- `confidence` — the teacher/annotator confidence in the label (used for curriculum weighting and to seed the model's own confidence calibration). Not a model output at training time.

## Label taxonomy (8 classes)

| Label | Definition |
|---|---|
| `billing` | Charges, invoices, refunds, payment methods, pricing disputes. |
| `technical_support` | Bugs, errors, outages, "it's not working," troubleshooting. |
| `sales` | Pre-purchase questions, plan comparisons, demos, upgrades, quotes. |
| `account` | Login, password, profile, permissions, account changes, cancellation. |
| `order_status` | Where is my order / shipment, delivery dates, tracking. |
| `feedback` | Praise, complaints, feature requests; no concrete action required. |
| `escalate_human` | Ambiguous, sensitive, legal/safety, or multi-issue messages needing a person. |
| `other` | Out-of-scope, spam, greetings-only, or anything that fits no other label. |

## Size and splits

| Split | Examples | Notes |
|---|---|---|
| train | 20,000 | stratified across 8 intents |
| validation | 2,000 | held out, used for checkpoint selection |
| test | 2,000 | held out, stratified at 250 per class |
| **total** | **24,000** | |

## Class balance

The training split is intentionally close to **uniform** (~2,500 per class) so the router does
not inherit a popularity prior — a misrouted rare intent is as costly as a misrouted common
one. The validation and test splits are **exactly balanced** at 250 per class so macro-F1 and
accuracy are directly comparable and not inflated by a dominant class.

| Intent | train | val | test |
|---|---|---|---|
| billing | 2,520 | 250 | 250 |
| technical_support | 2,540 | 250 | 250 |
| sales | 2,510 | 250 | 250 |
| account | 2,500 | 250 | 250 |
| order_status | 2,480 | 250 | 250 |
| feedback | 2,470 | 250 | 250 |
| escalate_human | 2,490 | 250 | 250 |
| other | 2,490 | 250 | 250 |

## Provenance

- **~70% synthetic** — generated with a frontier model and templated paraphrase expansion across the 8 intents, then deduplicated (exact-hash + MinHash near-dup).
- **~30% curated** — anonymized, relabeled public support-style message corpora.
- **Label audit** — human review of a 1,500-example sample; inter-annotator agreement κ = 0.91.
- **PII** — synthetic identifiers only; curated rows scrubbed of names, emails, and order numbers (replaced with realistic placeholders).

All content is fictional but believable, framed as illustrative reference data.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
