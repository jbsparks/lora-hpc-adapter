#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import logging as hf_logging


RUBRIC = """You are a strict HPC ops evaluator. Score Answer A vs Answer B on:
1) Technical correctness
2) Completeness (requested examples present)
3) Clarity/conciseness
4) Practicality (mentions placeholders, avoids unsafe guesses)

Return JSON only in this exact schema:
{
  "score_a": 0,
  "score_b": 0,
  "winner": "A|B|Tie",
  "rationale": "...",
  "issues_a": ["..."],
  "issues_b": ["..."]
}
"""


def build_prompt(question: str, answer_a: str, answer_b: str) -> str:
    return (
        f"{RUBRIC}\n"
        f"Question:\n{question}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}\n\n"
        "Respond with JSON only."
    )


def build_chat_prompt(tokenizer, question: str, answer_a: str, answer_b: str) -> str:
    system = "You are a strict evaluator. Return JSON only."
    user = build_prompt(question, answer_a, answer_b)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return text.strip()
    return match.group(0).strip()


def load_rules(path: str | None) -> dict:
    if not path:
        return {"rules": []}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_rule(rules: dict, question: str) -> dict | None:
    q = question.lower()
    for rule in rules.get("rules", []):
        match = rule.get("question_contains")
        if match and match.lower() in q:
            return rule
    return None


def rule_check(rule: dict, answer: str) -> tuple[list[str], list[str]]:
    hard = []
    soft = []
    text = answer.lower()
    for token in rule.get("tokens", []):
        if not token:
            continue
        kind = token[0]
        value = token[1:] if kind in ("@", "~") else token
        missing = value.lower() not in text
        if missing:
            if kind == "@":
                hard.append(f"Missing required token: {value}")
            else:
                soft.append(f"Missing preferred token: {value}")
    return hard, soft


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge baseline vs tuned answers.")
    parser.add_argument("judge_model", help="Judge model id or path")
    parser.add_argument("question", help="Original question")
    parser.add_argument("answer_a", help="Baseline answer")
    parser.add_argument("answer_b", help="Tuned answer")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--local-files-only", action="store_true", help="Do not attempt to download models")
    parser.add_argument("--use-chat-template", action="store_true", help="Use tokenizer chat template for prompt")
    parser.add_argument("--no-eos", action="store_true", help="Do not use eos_token_id to stop generation")
    parser.add_argument("--stream", action="store_true", help="Stream generated tokens to stdout")
    parser.add_argument("--rule-check", action="store_true", help="Apply deterministic rule checks")
    parser.add_argument("--rules-path", default=None, help="JSON file with required tokens for rule check")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        hf_logging.set_verbosity_info()
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        logging.info("Debug enabled")

    tokenizer = AutoTokenizer.from_pretrained(
        args.judge_model, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        logging.info("Judge model=%s", args.judge_model)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        local_files_only=args.local_files_only,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    if args.debug:
        logging.info("Model loaded in %.2fs", time.time() - t0)

    if args.use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt = build_chat_prompt(tokenizer, args.question, args.answer_a, args.answer_b)
    else:
        prompt = build_prompt(args.question, args.answer_a, args.answer_b)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None if args.no_eos else tokenizer.eos_token_id,
    )
    if args.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = args.min_new_tokens
        gen_kwargs["min_length"] = input_len + args.min_new_tokens

    if args.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer
        result_holder = {}

        def _run_generate():
            result_holder["ids"] = model.generate(**gen_kwargs)

        with torch.no_grad():
            gen_thread = threading.Thread(target=_run_generate)
            gen_thread.start()
            for chunk in streamer:
                print(chunk, end="", flush=True)
            gen_thread.join()
        output_ids = result_holder.get("ids")
    else:
        with torch.no_grad():
            output_ids = model.generate(**gen_kwargs)

    generated_ids = output_ids[0][input_len:] if output_ids is not None else torch.tensor([])
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if args.debug:
        logging.info("Generated tokens=%d", generated_ids.shape[0])
        logging.info("Decoded chars=%d", len(response))
    response = extract_json(response)

    try:
        parsed = json.loads(response)
        if args.rule_check:
            rules = load_rules(args.rules_path)
            selected = select_rule(rules, args.question)
            if selected:
                hard_a, soft_a = rule_check(selected, args.answer_a)
                hard_b, soft_b = rule_check(selected, args.answer_b)
                parsed.setdefault("issues_a", [])
                parsed.setdefault("issues_b", [])
                parsed["issues_a"] = list(dict.fromkeys(parsed["issues_a"] + hard_a + soft_a))
                parsed["issues_b"] = list(dict.fromkeys(parsed["issues_b"] + hard_b + soft_b))
                if isinstance(parsed.get("score_a"), int):
                    parsed["score_a"] = max(0, parsed["score_a"] - len(hard_a))
                if isinstance(parsed.get("score_b"), int):
                    parsed["score_b"] = max(0, parsed["score_b"] - len(hard_b))
                if parsed.get("score_a") is not None and parsed.get("score_b") is not None:
                    if parsed["score_a"] > parsed["score_b"]:
                        parsed["winner"] = "A"
                    elif parsed["score_b"] > parsed["score_a"]:
                        parsed["winner"] = "B"
                    else:
                        parsed["winner"] = "Tie"
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print(response)


if __name__ == "__main__":
    main()
