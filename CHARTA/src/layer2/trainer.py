"""
src/layer2/trainer.py
Fine-tunes ClinicalBERT for Chemical-Induced Disease (CID) relation extraction
using LoRA on BC5CDR. Designed to run on Google Colab T4 (Step 34).
"""
import json
from pathlib import Path
from typing import Optional
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, TaskType
import torch
from shared.utils import get_logger
from layer2.config import (CLINICALBERT_MODEL, BC5CDR_RE_DATASET,
                           BC5CDR_RE_CONFIG, LORA_RANK, LORA_ALPHA,
                           LORA_TARGET_MODULES)

logger = get_logger(__name__)

def _build_entity_pair_examples(dataset_split) -> list[dict]:
    """Convert BC5CDR KB split to (sentence, e1, e2, label) classification rows."""
    examples = []
    for item in dataset_split:
        for passage in item.get("passages", []):
            text = passage.get("text", "")
            entities = passage.get("entities", [])
            diseases  = [e for e in entities if e["type"] == "Disease"]
            chemicals = [e for e in entities if e["type"] == "Chemical"]
            rels = {(r["arg1_id"], r["arg2_id"]) for r in item.get("relations", [])}
            for d in diseases:
                for c in chemicals:
                    label = 1 if (c["id"], d["id"]) in rels or \
                                (d["id"], c["id"]) in rels else 0
                    inp = (f"[E1] {c['text'][0] if c['text'] else ''} [/E1] "
                           f"{text} "
                           f"[E2] {d['text'][0] if d['text'] else ''} [/E2]")
                    examples.append({"text": inp, "label": label})
    return examples

def finetune_relation_model(
    dataset_name: str = BC5CDR_RE_DATASET,
    config_name: str = BC5CDR_RE_CONFIG,
    output_dir: str = "models/lora_weights/clinicalbert_rel/",
    num_epochs: int = 10,
) -> None:
    """Fine-tune ClinicalBERT with LoRA for CID relation extraction."""
    logger.info("Loading BC5CDR dataset ...")
    ds = load_dataset(dataset_name, config_name)

    tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        CLINICALBERT_MODEL, num_labels=2)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True,
                         padding="max_length", max_length=256)

    train_rows = _build_entity_pair_examples(ds["train"])
    val_rows   = _build_entity_pair_examples(ds["validation"])
    from datasets import Dataset
    train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True)
    val_ds   = Dataset.from_list(val_rows).map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
    val_ds.set_format("torch",   columns=["input_ids","attention_mask","label"])

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=2e-4,
        weight_decay=0.01,
        fp16=True,                  # requires GPU — will error on CPU; OK on Colab T4
        logging_steps=50,
        report_to="none",
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=train_ds, eval_dataset=val_ds)
    logger.info("Starting LoRA fine-tuning of ClinicalBERT ...")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"LoRA weights saved to {output_dir}")