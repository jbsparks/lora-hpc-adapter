#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

DEFAULT_QUESTION = (
    "What is the Slurm srun command to start an interactive session? "
    "Give a minimal example and one requesting 4 CPUs and 16GB for 1 hour."
)


def _stream_reader(stream, sink, buffer_list):
    while True:
        chunk = stream.read(1)
        if chunk == "":
            break
        sink.write(chunk)
        sink.flush()
        buffer_list.append(chunk)


class _Tee:
    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, data):
        for sink in self._sinks:
            sink.write(data)

    def flush(self):
        for sink in self._sinks:
            sink.flush()


def run_script(args, capture=False, capture_stdout=False, stdout_path=None, progress=False):
    cmd = [sys.executable, *args]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if progress:
        stdout_file = open(stdout_path, "w", encoding="utf-8") if stdout_path else None
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout_buf = []
        stderr_buf = []
        if stdout_file:
            stdout_sink = _Tee(sys.stdout, stdout_file)
        else:
            stdout_sink = sys.stdout
        t_out = threading.Thread(target=_stream_reader, args=(process.stdout, stdout_sink, stdout_buf))
        t_err = threading.Thread(target=_stream_reader, args=(process.stderr, sys.stderr, stderr_buf))
        t_out.start()
        t_err.start()
        return_code = process.wait()
        t_out.join()
        t_err.join()
        if stdout_file:
            stdout_file.close()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        return "".join(stdout_buf).strip()
    if stdout_path:
        with open(stdout_path, "w", encoding="utf-8") as handle:
            subprocess.run(cmd, check=True, stdout=handle, env=env)
        return ""
    if capture_stdout:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True, env=env)
        return result.stdout.strip()
    if capture:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        return result.stdout.strip()
    subprocess.run(cmd, check=True, env=env)
    return ""


def write_or_print(text: str, out_path: str | None) -> None:
    if out_path:
        Path(out_path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-level CLI for the HPC LoRA demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser("baseline", help="Run baseline inference.")
    baseline.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    baseline.add_argument("--question", default=DEFAULT_QUESTION)
    baseline.add_argument("--max-new-tokens", type=int, default=512)
    baseline.add_argument("--min-new-tokens", type=int, default=0)
    baseline.add_argument("--use-4bit", action="store_true")
    baseline.add_argument("--debug", action="store_true")
    baseline.add_argument("--progress", action="store_true", help="Stream logs/progress to console")
    baseline.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    baseline.add_argument("--max-time", type=float, default=None)
    baseline.add_argument("--heartbeat", type=int, default=0)
    baseline.add_argument("--no-cache", action="store_true")
    baseline.add_argument("--use-context", default=None)
    baseline.add_argument("--show-context", action="store_true")
    baseline.add_argument("--use-chat-template", action="store_true")
    baseline.add_argument("--no-eos", action="store_true")
    baseline.add_argument("--suppress-special", action="store_true")
    baseline.add_argument("--out", help="Write answer to a file")

    finetune = subparsers.add_parser("finetune", help="Run LoRA/QLoRA fine-tune.")
    finetune.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    finetune.add_argument("--data-path", default="data_train.jsonl")
    finetune.add_argument("--output-dir", default="codellama-hpc-lora")
    finetune.add_argument("--use-4bit", action="store_true")
    finetune.add_argument("--max-seq-length", type=int, default=512)
    finetune.add_argument("--num-epochs", type=int, default=3)
    finetune.add_argument("--batch-size", type=int, default=1)
    finetune.add_argument("--grad-accum", type=int, default=4)
    finetune.add_argument("--lr", type=float, default=2e-4)
    finetune.add_argument("--use-chat-template", action="store_true")
    finetune.add_argument("--context-path", default=None)
    finetune.add_argument("--debug", action="store_true")
    finetune.add_argument("--progress", action="store_true", help="Stream logs/progress to console")

    tuned = subparsers.add_parser("tuned", help="Run inference with LoRA adapter.")
    tuned.add_argument("--adapter-path", default="codellama-hpc-lora")
    tuned.add_argument("--question", default=DEFAULT_QUESTION)
    tuned.add_argument("--max-new-tokens", type=int, default=512)
    tuned.add_argument("--min-new-tokens", type=int, default=64)
    tuned.add_argument("--use-4bit", action="store_true")
    tuned.add_argument("--debug", action="store_true")
    tuned.add_argument("--progress", action="store_true", help="Stream logs/progress to console")
    tuned.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    tuned.add_argument("--max-time", type=float, default=None)
    tuned.add_argument("--heartbeat", type=int, default=0)
    tuned.add_argument("--no-cache", action="store_true")
    tuned.add_argument("--use-context", default=None)
    tuned.add_argument("--show-context", action="store_true")
    tuned.add_argument("--use-chat-template", action="store_true")
    tuned.add_argument("--no-eos", action="store_true")
    tuned.add_argument("--suppress-special", action="store_true")
    tuned.add_argument("--out", help="Write answer to a file")

    judge = subparsers.add_parser("judge", help="Judge baseline vs tuned answers.")
    judge.add_argument("--judge-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    judge.add_argument("--question", default=DEFAULT_QUESTION)
    judge.add_argument("--answer-a", required=True)
    judge.add_argument("--answer-b", required=True)
    judge.add_argument("--max-new-tokens", type=int, default=512)
    judge.add_argument("--min-new-tokens", type=int, default=0)
    judge.add_argument("--debug", action="store_true")
    judge.add_argument("--progress", action="store_true", help="Stream logs/progress to console")
    judge.add_argument("--local-files-only", action="store_true")
    judge.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    judge.add_argument("--no-eos", action="store_true")
    judge.add_argument("--rule-check", action="store_true")
    judge.add_argument("--rules-path", default=None)
    judge.add_argument("--out", help="Write JSON to a file")

    demo = subparsers.add_parser("demo", help="Run baseline -> finetune -> tuned -> judge.")
    demo.add_argument("--model-id", default="codellama/CodeLlama-7b-Instruct-hf")
    demo.add_argument("--adapter-path", default="codellama-hpc-lora")
    demo.add_argument("--data-path", default="data_train.jsonl")
    demo.add_argument("--judge-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    demo.add_argument("--question", default=DEFAULT_QUESTION)
    demo.add_argument("--max-new-tokens", type=int, default=512)
    demo.add_argument("--min-new-tokens", type=int, default=64)
    demo.add_argument("--use-4bit", action="store_true")
    demo.add_argument("--debug", action="store_true")
    demo.add_argument("--progress", action="store_true", help="Stream logs/progress to console")
    demo.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    demo.add_argument("--max-time", type=float, default=None)
    demo.add_argument("--heartbeat", type=int, default=0)
    demo.add_argument("--no-cache", action="store_true")
    demo.add_argument("--use-context", default=None)
    demo.add_argument("--show-context", action="store_true")
    demo.add_argument("--use-chat-template", action="store_true")
    demo.add_argument("--no-eos", action="store_true")
    demo.add_argument("--suppress-special", action="store_true")
    demo.add_argument("--baseline-out", default="baseline.txt")
    demo.add_argument("--tuned-out", default="tuned.txt")
    demo.add_argument("--judge-out", default="judge.json")
    demo.add_argument("--skip-finetune", action="store_true")

    pretty = subparsers.add_parser("print-data", help="Pretty print training data.")
    pretty.add_argument("--data-path", default="data_train.jsonl")
    pretty.add_argument("--width", type=int, default=100)

    args = parser.parse_args()

    if args.command == "baseline":
        capture_stdout = args.debug and not args.progress
        stdout_path = None if args.progress else (args.out if args.out else None)
        answer = run_script(
            [
                "ask.py",
                args.model_id,
                args.question,
                "--max-new-tokens",
                str(args.max_new_tokens),
                *(["--min-new-tokens", str(args.min_new_tokens)] if args.min_new_tokens else []),
                *(["--use-4bit"] if args.use_4bit else []),
                *(["--debug"] if args.debug else []),
                *(["--stream"] if args.progress else []),
                *(["--stream-to", args.out] if args.progress and args.out else []),
                "--device",
                args.device,
                *(["--max-time", str(args.max_time)] if args.max_time else []),
                *(["--heartbeat", str(args.heartbeat)] if args.heartbeat else []),
                *(["--no-cache"] if args.no_cache else []),
                *(["--use-context", args.use_context] if args.use_context else []),
                *(["--show-context"] if args.show_context else []),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--no-eos"] if args.no_eos else []),
                *(["--suppress-special"] if args.suppress_special else []),
            ],
            capture=not capture_stdout and stdout_path is None,
            capture_stdout=capture_stdout and stdout_path is None,
            stdout_path=stdout_path,
            progress=args.progress,
        )
        if args.out and args.progress:
            answer = Path(args.out).read_text(encoding="utf-8").strip()
            write_or_print(answer, args.out)
        elif stdout_path:
            answer = Path(stdout_path).read_text(encoding="utf-8").strip()
            write_or_print(answer, args.out)
        else:
            write_or_print(answer, args.out)
        return

    if args.command == "finetune":
        run_script(
            [
                "finetune_lora.py",
                "--model-id",
                args.model_id,
                "--data-path",
                args.data_path,
                "--output-dir",
                args.output_dir,
                "--max-seq-length",
                str(args.max_seq_length),
                "--num-epochs",
                str(args.num_epochs),
                "--batch-size",
                str(args.batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--lr",
                str(args.lr),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--context-path", args.context_path] if args.context_path else []),
                *(["--use-4bit"] if args.use_4bit else []),
                *(["--debug"] if args.debug else []),
            ],
            capture=False,
            progress=args.progress,
        )
        return

    if args.command == "tuned":
        capture_stdout = args.debug and not args.progress
        stdout_path = None if args.progress else (args.out if args.out else None)
        answer = run_script(
            [
                "ask_lora.py",
                args.adapter_path,
                args.question,
                "--max-new-tokens",
                str(args.max_new_tokens),
                *(["--min-new-tokens", str(args.min_new_tokens)] if args.min_new_tokens else []),
                *(["--use-4bit"] if args.use_4bit else []),
                *(["--debug"] if args.debug else []),
                *(["--stream"] if args.progress else []),
                *(["--stream-to", args.out] if args.progress and args.out else []),
                "--device",
                args.device,
                *(["--max-time", str(args.max_time)] if args.max_time else []),
                *(["--heartbeat", str(args.heartbeat)] if args.heartbeat else []),
                *(["--no-cache"] if args.no_cache else []),
                *(["--use-context", args.use_context] if args.use_context else []),
                *(["--show-context"] if args.show_context else []),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--no-eos"] if args.no_eos else []),
                *(["--suppress-special"] if args.suppress_special else []),
            ],
            capture=not capture_stdout and stdout_path is None,
            capture_stdout=capture_stdout and stdout_path is None,
            stdout_path=stdout_path,
            progress=args.progress,
        )
        if args.out and args.progress:
            answer = Path(args.out).read_text(encoding="utf-8").strip()
            write_or_print(answer, args.out)
        elif stdout_path:
            answer = Path(stdout_path).read_text(encoding="utf-8").strip()
            write_or_print(answer, args.out)
        else:
            write_or_print(answer, args.out)
        return

    if args.command == "judge":
        capture_stdout = args.debug and not args.progress
        stdout_path = args.out if args.out else None
        result = run_script(
            [
                "judge.py",
                args.judge_model,
                args.question,
                args.answer_a,
                args.answer_b,
                "--max-new-tokens",
                str(args.max_new_tokens),
                *(["--min-new-tokens", str(args.min_new_tokens)] if args.min_new_tokens else []),
                *(["--debug"] if args.debug else []),
                *(["--local-files-only"] if args.local_files_only else []),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--no-eos"] if args.no_eos else []),
                *(["--stream"] if args.progress else []),
                *(["--rule-check"] if args.rule_check else []),
                *(["--rules-path", args.rules_path] if args.rules_path else []),
            ],
            capture=not capture_stdout and stdout_path is None,
            capture_stdout=capture_stdout and stdout_path is None,
            stdout_path=stdout_path,
            progress=args.progress,
        )
        if not stdout_path:
            write_or_print(result, None)
        return

    if args.command == "demo":
        capture_stdout = args.debug and not args.progress
        baseline_stdout_path = None if args.progress else (args.baseline_out if args.baseline_out else None)
        baseline_answer = run_script(
            [
                "ask.py",
                args.model_id,
                args.question,
                "--max-new-tokens",
                str(args.max_new_tokens),
                *(["--min-new-tokens", str(args.min_new_tokens)] if args.min_new_tokens else []),
                *(["--use-4bit"] if args.use_4bit else []),
                *(["--debug"] if args.debug else []),
                *(["--stream"] if args.progress else []),
                *(["--stream-to", args.baseline_out] if args.progress and args.baseline_out else []),
                "--device",
                args.device,
                *(["--max-time", str(args.max_time)] if args.max_time else []),
                *(["--heartbeat", str(args.heartbeat)] if args.heartbeat else []),
                *(["--no-cache"] if args.no_cache else []),
                *(["--use-context", args.use_context] if args.use_context else []),
                *(["--show-context"] if args.show_context else []),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--no-eos"] if args.no_eos else []),
                *(["--suppress-special"] if args.suppress_special else []),
            ],
            capture=not capture_stdout and baseline_stdout_path is None,
            capture_stdout=capture_stdout and baseline_stdout_path is None,
            stdout_path=baseline_stdout_path,
            progress=args.progress,
        )
        if args.baseline_out and args.progress:
            baseline_answer = Path(args.baseline_out).read_text(encoding="utf-8").strip()
        elif baseline_stdout_path:
            baseline_answer = Path(baseline_stdout_path).read_text(encoding="utf-8").strip()
        write_or_print(baseline_answer, args.baseline_out)

        if not args.skip_finetune:
            run_script(
                [
                    "finetune_lora.py",
                    "--model-id",
                    args.model_id,
                    "--data-path",
                    args.data_path,
                    "--output-dir",
                    args.adapter_path,
                    *(["--use-4bit"] if args.use_4bit else []),
                    *(["--debug"] if args.debug else []),
                ],
                capture=False,
                progress=args.progress,
            )

        tuned_stdout_path = None if args.progress else (args.tuned_out if args.tuned_out else None)
        tuned_answer = run_script(
            [
                "ask_lora.py",
                args.adapter_path,
                args.question,
                "--max-new-tokens",
                str(args.max_new_tokens),
                *(["--min-new-tokens", str(args.min_new_tokens)] if args.min_new_tokens else []),
                *(["--use-4bit"] if args.use_4bit else []),
                *(["--debug"] if args.debug else []),
                *(["--stream"] if args.progress else []),
                *(["--stream-to", args.tuned_out] if args.progress and args.tuned_out else []),
                "--device",
                args.device,
                *(["--max-time", str(args.max_time)] if args.max_time else []),
                *(["--heartbeat", str(args.heartbeat)] if args.heartbeat else []),
                *(["--no-cache"] if args.no_cache else []),
                *(["--use-context", args.use_context] if args.use_context else []),
                *(["--show-context"] if args.show_context else []),
                *(["--use-chat-template"] if args.use_chat_template else []),
                *(["--no-eos"] if args.no_eos else []),
                *(["--suppress-special"] if args.suppress_special else []),
            ],
            capture=not capture_stdout and tuned_stdout_path is None,
            capture_stdout=capture_stdout and tuned_stdout_path is None,
            stdout_path=tuned_stdout_path,
            progress=args.progress,
        )
        if args.tuned_out and args.progress:
            tuned_answer = Path(args.tuned_out).read_text(encoding="utf-8").strip()
        elif tuned_stdout_path:
            tuned_answer = Path(tuned_stdout_path).read_text(encoding="utf-8").strip()
        write_or_print(tuned_answer, args.tuned_out)

        judge_stdout_path = args.judge_out if args.judge_out else None
        judge_result = run_script(
            [
                "judge.py",
                args.judge_model,
                args.question,
                baseline_answer,
                tuned_answer,
                *(["--debug"] if args.debug else []),
            ],
            capture=not capture_stdout and judge_stdout_path is None,
            capture_stdout=capture_stdout and judge_stdout_path is None,
            stdout_path=judge_stdout_path,
            progress=args.progress,
        )
        if not judge_stdout_path:
            write_or_print(judge_result, None)
        return

    if args.command == "print-data":
        if not (args.data_path.endswith(".jsonl") or args.data_path.endswith(".json")):
            raise SystemExit("--data-path must be .jsonl or .json")
        path = Path(args.data_path)
        if path.suffix == ".jsonl":
            items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            items = json.loads(path.read_text(encoding="utf-8"))

        col_width = max(40, (args.width - 5) // 2)
        header = f"{'INSTRUCTION'.ljust(col_width)} | {'OUTPUT'.ljust(col_width)}"
        print(header)
        print("-" * len(header))

        for item in items:
            instruction = str(item.get("instruction", "")).strip()
            output = str(item.get("output", "")).strip()
            inst_lines = textwrap.wrap(instruction, width=col_width) or [""]
            out_lines = textwrap.wrap(output, width=col_width) or [""]
            rows = max(len(inst_lines), len(out_lines))
            inst_lines += [""] * (rows - len(inst_lines))
            out_lines += [""] * (rows - len(out_lines))
            for i in range(rows):
                print(f"{inst_lines[i].ljust(col_width)} | {out_lines[i].ljust(col_width)}")
            print("-" * len(header))
        return


if __name__ == "__main__":
    main()
