#!/usr/bin/env python3
import argparse
import logging
import os
import json
import sys
import time
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from transformers.utils import logging as hf_logging

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
CHATHPC_TEMPLATE = (
    "You are a powerful LLM model for HPC operations. Your job is to answer questions about HPC.\n"
    "You are given a question and context regarding HPC operations.\n\n"
    "You must output the HPC response that answers the question.\n\n"
    "### Input:\n{instruction}\n"
    "### Context:\n{context}\n\n"
    "### Response:\n"
)


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
        raise SystemExit("--use-context must be .json or .jsonl")
    raise SystemExit("Context not found in file")


def build_prompt(instruction: str, context: str | None) -> str:
    if context:
        return CHATHPC_TEMPLATE.format(
            instruction=instruction.strip(),
            context=context.strip(),
        )
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def build_chat_prompt(tokenizer, instruction: str, context: str | None) -> str:
    system = context or "You are a helpful HPC assistant."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline inference.")
    parser.add_argument("model_id", help="Base model id or path, e.g. codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("question", help="User question/prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Force at least N tokens before stopping")
    parser.add_argument("--use-4bit", action="store_true", help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument("--stream", action="store_true", help="Stream generated tokens to stdout")
    parser.add_argument("--stream-to", default=None, help="Write streamed tokens to a file")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-time", type=float, default=None, help="Max generation time in seconds")
    parser.add_argument("--heartbeat", type=int, default=0, help="Print progress every N seconds during generate")
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache during generation")
    parser.add_argument("--use-context", default=None, help="Path to JSON/JSONL context file")
    parser.add_argument("--show-context", action="store_true", help="Print resolved context and exit")
    parser.add_argument("--use-chat-template", action="store_true", help="Use tokenizer chat template for prompt")
    parser.add_argument("--no-eos", action="store_true", help="Do not use eos_token_id to stop generation")
    parser.add_argument("--suppress-special", action="store_true", help="Suppress eos/pad tokens during generation")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.debug:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        hf_logging.set_verbosity_info()
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"
        logging.info("Debug enabled")

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

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        dtype = torch.bfloat16
        device = "cuda"
    elif args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available()):
        dtype = torch.float16
        device = "mps"
    else:
        dtype = torch.float32
        device = "cpu"
    if args.debug:
        logging.info("Device=%s dtype=%s", device, dtype)
        logging.info("Model=%s", args.model_id)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        quantization_config=bnb_config,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    if args.debug:
        logging.info("Model loaded in %.2fs", time.time() - t0)

    context = load_context(args.use_context) if args.use_context else None
    if args.show_context:
        if context is None:
            print("")
        else:
            print(context)
        return
    if args.use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt = build_chat_prompt(tokenizer, args.question, context)
    else:
        prompt = build_prompt(args.question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if args.debug:
        logging.info("Prompt length=%d chars", len(prompt))
        logging.info("Input shape=%s", tuple(inputs["input_ids"].shape))

    if args.stream:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = None
    if args.debug:
        logging.info("Starting generation (max_new_tokens=%d)", args.max_new_tokens)
        if args.max_time:
            logging.info("Max generation time=%.1fs", args.max_time)

    stop_event = threading.Event()

    def heartbeat():
        start = time.time()
        while not stop_event.wait(args.heartbeat):
            logging.info("Generating... %.1fs elapsed", time.time() - start)

    hb_thread = None
    if args.heartbeat > 0:
        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None if args.no_eos else tokenizer.eos_token_id,
        streamer=streamer,
        max_time=args.max_time,
        use_cache=not args.no_cache,
    )
    if args.min_new_tokens > 0:
        gen_kwargs["min_new_tokens"] = args.min_new_tokens
        gen_kwargs["min_length"] = input_len + args.min_new_tokens

    if args.suppress_special:
        bad_ids = []
        if tokenizer.eos_token_id is not None:
            bad_ids.append([tokenizer.eos_token_id])
        if tokenizer.pad_token_id is not None:
            bad_ids.append([tokenizer.pad_token_id])
        if bad_ids:
            gen_kwargs["bad_words_ids"] = bad_ids
    input_len = inputs["input_ids"].shape[1]

    if args.stream:
        stream_file = open(args.stream_to, "w", encoding="utf-8") if args.stream_to else None
        result_holder = {}

        def _run_generate():
            result_holder["ids"] = model.generate(**gen_kwargs)

        gen_thread = threading.Thread(target=_run_generate)
        gen_thread.start()
        collected = []
        for chunk in streamer:
            sys.stdout.write(chunk)
            sys.stdout.flush()
            if stream_file:
                stream_file.write(chunk)
                stream_file.flush()
            collected.append(chunk)
        gen_thread.join()

        # Fallback: if streamer yielded nothing, decode the output ids.
        if "ids" in result_holder and args.debug:
            generated_ids = result_holder["ids"][0][input_len:]
            logging.info("Generated tokens=%d (stream)", generated_ids.shape[0])

        if not collected and "ids" in result_holder:
            generated_ids = result_holder["ids"][0][input_len:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if answer:
                sys.stdout.write(answer)
                sys.stdout.flush()
                if stream_file:
                    stream_file.write(answer)
                    stream_file.flush()
            elif args.debug:
                logging.info("Decoded chars=0 (stream fallback).")
                logging.info("First 20 token ids=%s", generated_ids[:20].tolist())

        if stream_file:
            stream_file.close()
        stop_event.set()
        if hb_thread:
            hb_thread.join(timeout=1)
    else:
        output_ids = model.generate(**gen_kwargs)
        stop_event.set()
        if hb_thread:
            hb_thread.join(timeout=1)
        generated_ids = output_ids[0][input_len:]
        if args.suppress_special:
            bad_ids = set()
            if tokenizer.eos_token_id is not None:
                bad_ids.add(tokenizer.eos_token_id)
            if tokenizer.pad_token_id is not None:
                bad_ids.add(tokenizer.pad_token_id)
            token_list = [t for t in generated_ids.tolist() if t not in bad_ids]
            generated_ids = torch.tensor(token_list, device=generated_ids.device)
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        if args.debug:
            logging.info("Generated tokens=%d", generated_ids.shape[0])
            logging.info("Decoded chars=%d", len(answer))
        print(answer, end="")


if __name__ == "__main__":
    main()
