import argparse

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None
from typing import List

from .pipeline import (
    init_config,
    run_build_index,
    run_curate,
    run_full,
    run_generate,
    run_ingest,
)


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="datasetgen")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create datasetgen.yaml config")
    p_init.add_argument("--out", default="datasetgen.yaml")

    p_ingest = sub.add_parser("ingest", help="ingest documents to corpus.jsonl")
    p_ingest.add_argument("--input", nargs="+", required=True)
    p_ingest.add_argument("--out", required=True)
    p_ingest.add_argument("--license", default="unknown")

    p_index = sub.add_parser("build-index", help="build chunk store + BM25 index")
    p_index.add_argument("--corpus", required=True)
    p_index.add_argument("--out", required=True)
    p_index.add_argument("--max-chars", type=int, default=800)
    p_index.add_argument("--overlap", type=int, default=120)

    p_gen = sub.add_parser("generate", help="generate candidates.jsonl")
    p_gen.add_argument("--index", required=True)
    p_gen.add_argument("--hypothesis", default="datasetgen.yaml")
    p_gen.add_argument("--out", required=True)
    p_gen.add_argument("--limit", type=int, default=0)

    p_curate = sub.add_parser("curate", help="curate train/eval JSONL")
    p_curate.add_argument("--in", dest="in_path", required=True)
    p_curate.add_argument("--out", required=True)
    p_curate.add_argument("--eval-out", required=True)
    p_curate.add_argument("--manifest", default="manifest.jsonl")
    p_curate.add_argument("--judge", choices=["stub", "off", "human"], default="stub")
    p_curate.add_argument("--human-review-out", default="")
    p_curate.add_argument("--human-review-in", default="")
    p_curate.add_argument("--index", default="")

    p_pp = sub.add_parser("pretty-print", help="pretty print a JSONL file")
    p_pp.add_argument("--in", dest="in_path", required=True)
    p_pp.add_argument("--out", default="")
    p_pp.add_argument("--mode", choices=["json", "text"], default="json")
    p_pp.add_argument("--limit", type=int, default=0)

    p_run = sub.add_parser("run", help="run full pipeline")
    p_run.add_argument("--input", nargs="+", required=True)
    p_run.add_argument("--hypothesis", default="datasetgen.yaml")
    p_run.add_argument("--outdir", required=True)

    return parser


def main(argv: List[str] = None) -> int:
    if load_dotenv is not None:
        load_dotenv()
    parser = _common_parser()
    args = parser.parse_args(argv)

    if args.cmd == "init":
        init_config(args.out)
        return 0
    if args.cmd == "ingest":
        run_ingest(args.input, args.out, args.license)
        return 0
    if args.cmd == "build-index":
        run_build_index(args.corpus, args.out, max_chars=args.max_chars, overlap=args.overlap)
        return 0
    if args.cmd == "generate":
        run_generate(args.index, args.hypothesis, args.out, question_limit=args.limit)
        return 0
    if args.cmd == "curate":
        run_curate(
            args.in_path,
            args.out,
            args.eval_out,
            args.manifest,
            judge_mode=args.judge,
            human_review_out=args.human_review_out,
            human_review_in=args.human_review_in,
            index_dir=args.index,
        )
        return 0
    if args.cmd == "pretty-print":
        from .pretty import pretty_print_jsonl

        pretty_print_jsonl(args.in_path, args.out, mode=args.mode, limit=args.limit)
        return 0
    if args.cmd == "run":
        run_full(args.input, args.hypothesis, args.outdir)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
