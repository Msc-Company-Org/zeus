# Zeus — Intent Routing / Classification (Fine-Tuned 2B)

**A 2B open model that routes incoming messages to the right intent — 99.1% routing accuracy vs GPT-4o's 97.8%, at 40× lower inference cost and sub-50ms latency.**

Zeus is a reference model from **MSC Labs**. It fine-tunes `google/gemma-2-2b-it` with QLoRA to classify an incoming user message into exactly one of eight agent-routing intents. It is built to sit on the hot path of an agent system: small enough to self-host cheaply, fast enough that routing is not a latency tax. The same pipeline we build for clients, open and reproducible.

## What it does

- **Input:** a single user `message` (a support chat line, a form submission, an inbound email subject+body).
- **Output:** one `intent` label from a fixed taxonomy, plus a calibrated `confidence` score.
- **Behavior:** when a message is genuinely ambiguous or out-of-scope, Zeus is trained to emit `escalate_human` or `other` rather than force a confident wrong route. The router's job is to be right, then to fail safe.

### Intent taxonomy (8 labels)

`billing` · `technical_support` · `sales` · `account` · `order_status` · `feedback` · `escalate_human` · `other`

## Why a tuned 2B beats a frontier API here

Routing is a narrow, high-volume, latency-sensitive classification task — the worst possible fit for a large general-purpose API. A frontier model is overkill per call, costs real money at scale, and adds hundreds of milliseconds before any downstream work begins. Zeus is trained on the exact label set and decision boundaries of one routing contract, so it is **slightly more accurate, 40× cheaper, and roughly 19× faster** than calling GPT-4o for the same decision. On a router that fires on every inbound message, that gap compounds fast.

## Results

Illustrative reference results from the MSC Labs eval harness. Test set: 2,000 held-out messages across the 8 intents (stratified). Baseline: GPT-4o (`gpt-4o-2024-08-06`), same label set, same few-shot routing prompt, temperature 0.

| Model | Routing accuracy | Macro-F1 | $ / 1k requests | p50 latency |
|---|---|---|---|---|
| GPT-4o (baseline) | 97.8% | 0.964 | $4.80 | 720 ms |
| **Zeus (Gemma-2-2B + QLoRA)** | **99.1%** | **0.985** | **$0.12** | **38 ms** |
| **Delta** | **+1.3 pts** | **+0.021** | **40× cheaper** | **~19× faster** |

The accuracy gap is small — both models are strong on clear messages. The real wins are **cost** and **latency**: at 38ms p50, Zeus routes inside the budget you'd normally reserve for a network round-trip, and at $0.12/1k it is effectively free to run on every inbound message.

See [`eval/results.md`](eval/results.md) for per-class precision/recall, methodology, and the full cost breakdown.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "google/gemma-2-2b-it"
ADAPTER = "msc-labs/zeus-intent-router-qlora"  # illustrative adapter id

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, ADAPTER)

LABELS = [
    "billing", "technical_support", "sales", "account",
    "order_status", "feedback", "escalate_human", "other",
]
SYSTEM = (
    "You are an intent router. Read the user message and respond with exactly one "
    "label from this set: " + ", ".join(LABELS) + ". "
    "If the message is ambiguous or needs a person, use escalate_human. "
    "Respond with only the label."
)

message = "I was charged twice for my subscription this month, can you refund one?"

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": message},
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=4, temperature=0.0, do_sample=False)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
# -> "billing"
```

For production, the recommended deployment is a constrained-decoding head (logit mask over the 8 label tokens) so the model can only emit a valid label. That is how the latency and accuracy numbers above were measured.

## Files

- [`MODEL_CARD.md`](MODEL_CARD.md) — formal model card: intended use, limits, training data.
- [`training/config.yaml`](training/config.yaml) — QLoRA hyperparameters.
- [`training/train.py`](training/train.py) — QLoRA training script (TRL / PEFT).
- [`eval/results.md`](eval/results.md) — full evaluation, per-class table, and cost breakdown.
- [`eval/results.json`](eval/results.json) — machine-readable metrics.
- [`data/sample.jsonl`](data/sample.jsonl) — sample routing rows.
- [`data/README.md`](data/README.md) — dataset description, taxonomy, class balance.

## License

Apache-2.0 for code. Reference weights and datasets are illustrative and not distributed.

---

> Reference model by **MSC Labs** — done-for-you custom model training.
> Want this for your task? → Book a free model audit: https://msc-labs-ai.vercel.app/assessment
> Numbers are illustrative reference results from our standard eval harness.
