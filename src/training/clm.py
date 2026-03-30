"""CLM (Causal Language Modeling) continued pretraining for EXP-004."""

from __future__ import annotations

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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.training.qlora import QloraTrainingConfig, TrainingRunResult

logger = logging.getLogger(__name__)

MIN_CHUNK_TOKENS = 64


class CorpusChunkDataset(Dataset):
    """Tokenized plain-text chunks for causal LM pretraining."""

    def __init__(
        self,
        corpus_text: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int,
    ) -> None:
        token_ids = tokenizer(
            corpus_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        self.chunks: list[dict[str, list[int]]] = []
        for start in range(0, len(token_ids), max_seq_length):
            chunk_ids = token_ids[start : start + max_seq_length]
            if len(chunk_ids) < MIN_CHUNK_TOKENS:
                break
            self.chunks.append(
                {
                    "input_ids": chunk_ids,
                    "labels": list(chunk_ids),
                    "attention_mask": [1] * len(chunk_ids),
                }
            )

        logger.info(
            "CLM dataset: %d tokens → %d chunks of ≤%d tokens",
            len(token_ids),
            len(self.chunks),
            max_seq_length,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.chunks[index]


def train_clm_adapter(
    corpus_texts: list[str],
    config: QloraTrainingConfig,
) -> TrainingRunResult:
    """Train a QLoRA adapter on corpus text with causal LM loss."""

    if not torch.cuda.is_available():
        raise RuntimeError("CLM training requires CUDA with bitsandbytes 4-bit support.")

    logger.info(
        "=== CLM training start === seed=%d, rank=%d, alpha=%d, lr=%s, epochs=%d, "
        "max_seq_length=%d, batch=%d, grad_accum=%d, max_steps=%d",
        config.seed, config.rank, config.alpha, config.learning_rate,
        config.epochs, config.max_seq_length, config.per_device_batch_size,
        config.gradient_accumulation_steps, config.max_steps,
    )

    set_seed(config.seed)
    random.seed(config.seed)

    logger.info("Loading tokenizer from %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    logger.info("Concatenating %d document texts", len(corpus_texts))
    corpus_text = "\n\n".join(corpus_texts)
    logger.info("Corpus: %d chars, %d words", len(corpus_text), len(corpus_text.split()))

    dataset = CorpusChunkDataset(
        corpus_text=corpus_text,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
    )
    if len(dataset) == 0:
        raise ValueError("CLM dataset is empty — corpus may be too short or max_seq_length too large")

    logger.info("Loading quantized model + LoRA for CLM training")
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
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_before = torch.cuda.memory_allocated() / 1024 / 1024
        logger.info("VRAM before training: %.1f MB", vram_before)

    total_steps = len(dataset) * config.epochs // (
        config.per_device_batch_size * config.gradient_accumulation_steps
    )
    if config.max_steps > 0:
        total_steps = min(total_steps, config.max_steps)
    logger.info(
        "Starting Trainer.train() — %d chunks × %d epochs = ~%d optimizer steps",
        len(dataset), config.epochs, total_steps,
    )

    start_time = time.perf_counter()
    trainer.train()
    train_time_seconds = time.perf_counter() - start_time
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024
        if torch.cuda.is_available()
        else None
    )
    logger.info("Training done in %.1fs, peak VRAM: %.1f MB", train_time_seconds, peak_vram_mb or 0)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving adapter to %s", config.output_dir)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Finished CLM training for seed %d in %.1fs (%d chunks, %d epochs)",
        config.seed,
        train_time_seconds,
        len(dataset),
        config.epochs,
    )
    return TrainingRunResult(
        adapter_dir=config.output_dir,
        train_time_seconds=train_time_seconds,
        peak_vram_mb=peak_vram_mb,
    )


def _load_trainable_model(config: QloraTrainingConfig) -> AutoModelForCausalLM:
    logger.info("Loading %s with 4-bit NF4 quantization (double_quant=True)", config.model_name)
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
    if torch.cuda.is_available():
        logger.info("VRAM after model load: %.1f MB", torch.cuda.memory_allocated() / 1024 / 1024)

    logger.info(
        "Attaching LoRA: rank=%d, alpha=%d, dropout=%.2f, targets=%s",
        config.rank, config.alpha, config.dropout, config.target_modules,
    )
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
