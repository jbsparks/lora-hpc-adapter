#!/usr/bin/env python3
import argparse
import json
import logging
import os
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from transformers.utils import logging as hf_logging

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n{output}"
CHATHPC_TEMPLATE = (
    "You are a knowledgeable HPC user advocate with practical, operations-ready experience in Slurm, MPI, and Spack. "
    "Provide concise, correct guidance with safe placeholders for site-specific flags (e.g., -p <partition>, -A <account>). "
    "Avoid destructive commands and keep answers deterministic and actionable."
)


def format_example(example: dict) -> dict:
    text = PROMPT_TEMPLATE.format(
        instruction=example["instruction"].strip(),
        output=example["output"].strip(),
    )
    return {"text": text}


def load_context(path: str) -> str:
    if path.endswith(".jsonl"):
        for line in open(path, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "context" in obj:
                return str(obj["context"])
            break
    elif path.endswith(".json"):
        obj = json.loads(open(path, "r", encoding="utf-8").read())
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and "context" in item:
                    return str(item["context"])
        elif isinstance(obj, dict) and "context" in obj:
            return str(obj["context"])
    else:
        raise SystemExit("--context-path must be .json or .jsonl")
    raise SystemExit("Context not found in file")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT fine-tune.")
    parser.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--data-path", default="data_train.jsonl")
    parser.add_argument("--output-dir", default="codellama-hpc-lora")
    parser.add_argument("--use-4bit", action="store_true", help="Enable QLoRA 4-bit loading")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use-chat-template", action="store_true", help="Format training data as chat")
    parser.add_argument("--context-path", default=None, help="Optional JSON/JSONL context for chat system prompt")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        hf_logging.set_verbosity_info()
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        logging.info("Debug enabled")

    if not (args.data_path.endswith(".jsonl") or args.data_path.endswith(".json")):
        raise SystemExit("--data-path must be .jsonl or .json")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = None
    if args.use_4bit:
        if not torch.cuda.is_available():
            raise SystemExit("--use-4bit requires CUDA (bitsandbytes).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"
    if args.debug:
        logging.info("Device=%s dtype=%s", device, dtype)
        logging.info("Model=%s", args.model_id)
        logging.info("Train examples=%d", len(dataset))
        logging.info("Output dir=%s", args.output_dir)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        quantization_config=bnb_config,
    )
    if device != "cuda":
        model.to(device)
    if args.debug:
        logging.info("Model loaded in %.2fs", time.time() - t0)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        fp16=torch.backends.mps.is_available(),
        report_to="none",
        seed=42,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        gradient_checkpointing=not torch.backends.mps.is_available(),
    )

    context = None
    if args.context_path:
        context = load_context(args.context_path)

    def chat_format(example: dict) -> str:
        system = context or CHATHPC_TEMPLATE
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": example["instruction"].strip()},
            {"role": "assistant", "content": example["output"].strip()},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    if args.use_chat_template:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=chat_format,
            args=training_args,
        )
    else:
        dataset = dataset.map(format_example, remove_columns=dataset.column_names)
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
