"""QLoRA training utilities for EXP-003."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.raft import RaftExample

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class QloraTrainingConfig:
    """Fixed training configuration derived from the experiment spec."""

    model_name: str
    output_dir: Path
    seed: int
    rank: int
    alpha: int
    dropout: float
    target_modules: tuple[str, ...]
    learning_rate: float
    max_seq_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    warmup_ratio: float
    weight_decay: float
    max_steps: int = -1
    attn_implementation: str = "eager"


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Minimal training artifact summary."""

    adapter_dir: Path
    train_time_seconds: float
    peak_vram_mb: float | None


class SupervisedChatDataset(Dataset):
    """Tokenized user/assistant chat examples with masked prompt labels."""

    def __init__(
        self,
        examples: list[RaftExample],
        tokenizer: AutoTokenizer,
        max_seq_length: int,
    ) -> None:
        self.records = [
            _tokenize_example(example, tokenizer, max_seq_length)
            for example in examples
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.records[index]


class CompletionOnlyTrainer(Trainer):
    """Trainer that limits Gemma2 logits to the supervised answer suffix."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        logits_to_keep, shift_labels = _build_suffix_loss_inputs(labels)
        outputs = model(
            **inputs,
            labels=labels,
            logits_to_keep=logits_to_keep,
            shift_labels=shift_labels,
            num_items_in_batch=num_items_in_batch,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def train_qlora_adapter(
    examples: list[RaftExample],
    config: QloraTrainingConfig,
) -> TrainingRunResult:
    """Fine-tune a quantized backbone with LoRA adapters."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA-visible runtime with bitsandbytes 4-bit support."
        )

    set_seed(config.seed)
    random.seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = SupervisedChatDataset(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    model = _load_trainable_model(config)

    training_args = TrainingArguments(
        output_dir=str(config.output_dir / "trainer_state"),
        overwrite_output_dir=True,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        torch_empty_cache_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=config.max_steps,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )
    trainer = CompletionOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    trainer.train()
    train_time_seconds = time.perf_counter() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else None
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Finished QLoRA training for seed %d in %.1fs",
        config.seed,
        train_time_seconds,
    )
    return TrainingRunResult(
        adapter_dir=config.output_dir,
        train_time_seconds=train_time_seconds,
        peak_vram_mb=peak_vram_mb,
    )


def _load_trainable_model(config: QloraTrainingConfig) -> AutoModelForCausalLM:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation=config.attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def _build_suffix_loss_inputs(labels: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Build a compact shifted-label tensor for answer-only suffix loss."""

    completion_lengths = (labels != -100).sum(dim=1)
    if torch.any(completion_lengths <= 0):
        raise ValueError("Each QLoRA training example must contain at least one supervised answer token")

    logits_to_keep = int(completion_lengths.max().item()) + 1
    shift_labels = torch.full(
        (labels.size(0), logits_to_keep),
        fill_value=-100,
        dtype=labels.dtype,
        device=labels.device,
    )
    for row_index, completion_length in enumerate(completion_lengths.tolist()):
        answer_tokens = labels[row_index][labels[row_index] != -100]
        suffix_start = logits_to_keep - (completion_length + 1)
        shift_labels[row_index, suffix_start : suffix_start + completion_length] = answer_tokens

    return logits_to_keep, shift_labels


def _tokenize_example(
    example: RaftExample,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
) -> dict[str, list[int]]:
    prompt_messages = [{"role": "user", "content": example.prompt}]
    full_messages = [
        {"role": "user", "content": example.prompt},
        {"role": "assistant", "content": example.answer},
    ]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    if len(full_ids) > max_seq_length:
        overflow = len(full_ids) - max_seq_length
        logger.warning(
            "RAFT example %s exceeds max_seq_length: %d > %d (overflow %d tokens, left-truncating context)",
            example.question_id,
            len(full_ids),
            max_seq_length,
            overflow,
        )
        full_ids = full_ids[-max_seq_length:]
        prompt_len = max(len(prompt_ids) - overflow, 0)
    else:
        prompt_len = len(prompt_ids)
    prompt_len = min(prompt_len, len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
