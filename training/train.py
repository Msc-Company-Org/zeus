"""
Zeus — QLoRA fine-tuning for an intent router (Gemma-2-2B-it).

Illustrative reference training script from the MSC Labs harness. It fine-tunes a
2B base model to emit exactly one intent label per message. Single-GPU (A100 40GB),
~1.4 GPU-hours for the full run.

Usage:
    python training/train.py --config training/config.yaml             # full run
    python training/train.py --config training/config.yaml --smoke     # 1% smoke test
"""

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(tokenizer, message: str, label: str | None, labels: list[str]) -> str:
    """Render one routing example into the Gemma chat format.

    During training we append the gold label so the model learns to emit it;
    at inference `label` is None and we stop after `add_generation_prompt`.
    """
    system = (
        "You are an intent router. Read the user message and respond with exactly "
        "one label from this set: " + ", ".join(labels) + ". "
        "If the message is ambiguous or needs a person, use escalate_human. "
        "Respond with only the label."
    )
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    if label is not None:
        text += f"{label}{tokenizer.eos_token}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true", help="run the 1%% smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    m, q, lz, d, t = (
        cfg["model"], cfg["quantization"], cfg["lora"], cfg["data"], cfg["training"],
    )
    labels = d["labels"]

    # --- tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(m["base_model"])
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit base model (QLoRA) ---
    bnb = BitsAndBytesConfig(
        load_in_4bit=q["load_in_4bit"],
        bnb_4bit_quant_type=q["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, q["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        m["base_model"],
        quantization_config=bnb,
        attn_implementation=m["attn_implementation"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # --- LoRA adapters ---
    lora = LoraConfig(
        r=lz["r"],
        lora_alpha=lz["lora_alpha"],
        lora_dropout=lz["lora_dropout"],
        bias=lz["bias"],
        task_type=lz["task_type"],
        target_modules=lz["target_modules"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # --- data ---
    ds = load_dataset(
        "json",
        data_files={"train": d["train_file"], "eval": d["eval_file"]},
    )
    if args.smoke and cfg.get("smoke_test", {}).get("enabled"):
        frac = cfg["smoke_test"]["sample_fraction"]
        n = max(1, int(len(ds["train"]) * frac))
        ds["train"] = ds["train"].shuffle(seed=t["seed"]).select(range(n))
        print(f"[smoke] training on {n} examples")

    def fmt(batch):
        return {
            "text": [
                build_prompt(tokenizer, msg, lab, labels)
                for msg, lab in zip(batch[d["message_field"]], batch[d["label_field"]])
            ]
        }

    ds = ds.map(fmt, batched=True, remove_columns=ds["train"].column_names)

    # --- training args ---
    targs = TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=1 if args.smoke else t["num_train_epochs"],
        max_steps=cfg["smoke_test"]["max_steps"] if args.smoke else -1,
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        optim=t["optim"],
        bf16=t["bf16"],
        max_grad_norm=t["max_grad_norm"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        seed=t["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        dataset_text_field="text",
        max_seq_length=d["max_seq_length"],
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(t["output_dir"])
    tokenizer.save_pretrained(t["output_dir"])
    print(f"Saved adapter to {t['output_dir']}")


if __name__ == "__main__":
    main()
