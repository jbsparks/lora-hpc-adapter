import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from .chunking import chunk_corpus
from .dedupe import exact_dedupe, token_jaccard_dedupe
from .export import split_train_eval, write_jsonl, write_manifest
from .index import build_bm25_index, load_index, retrieve, save_index
from .ingest import ingest_paths, load_corpus, write_corpus
from .judge import apply_repair, deterministic_validators, judge_stub
from .llm import LLMClient, OpenAIClient, StubLLMClient
from .prompts import answer_prompt, qgen_prompt


DEFAULT_CONFIG = {
    "hypothesis": "Use the following documents as sources, generate user-based use-cases and questions, then generate a concise validated single answer.",
    "topics": {
        "slurm.interactive": {"commands": ["srun", "salloc"], "quota": 12},
        "slurm.discovery": {"commands": ["sinfo", "squeue"], "quota": 10},
        "slurm.batch": {"commands": ["sbatch"], "quota": 8},
    },
    "rules": {
        "must_include": ["use placeholders for partition/account"],
        "must_avoid": ["hard-coded -p/-A values"],
    },
    "quotas": {"L1": 18, "L2": 10, "L3": 4},
    "model": {"generator": "stub", "judge": "stub", "openai_model": "", "openai_api_key": ""},
    "question_index": "",
    "answer_index": "",
    "question_limit": 0,
    "seed": 0,
}


DIFFICULTY_ORDER = ["L1", "L2", "L3"]


def load_hypothesis_config(path: str) -> Dict:
    if not path:
        return DEFAULT_CONFIG
    p = Path(path)
    if not p.exists():
        return DEFAULT_CONFIG
    import yaml

    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data


def init_config(path: str) -> None:
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(DEFAULT_CONFIG, f, sort_keys=False)


def run_ingest(inputs: List[str], out_path: str, license_str: str = "unknown") -> None:
    docs = ingest_paths(inputs, license_str=license_str)
    write_corpus(docs, out_path)


def run_build_index(corpus_path: str, out_dir: str, max_chars: int = 800, overlap: int = 120) -> None:
    corpus = load_corpus(corpus_path)
    chunks = [c.__dict__ for c in chunk_corpus(corpus, max_chars=max_chars, overlap=overlap)]
    index = build_bm25_index(chunks)
    save_index(chunks, index, out_dir)


def _assign_topic(topics: Dict[str, Dict], seed: int) -> List[str]:
    rng = random.Random(seed)
    topic_list = []
    for topic, meta in topics.items():
        topic_list.extend([topic] * int(meta.get("quota", 0)))
    if not topic_list:
        topic_list = list(topics.keys())
    rng.shuffle(topic_list)
    return topic_list


def _difficulty_cycle(quotas: Dict[str, int]) -> List[str]:
    order = []
    for d in DIFFICULTY_ORDER:
        order.extend([d] * int(quotas.get(d, 0)))
    return order or ["L1"]


def _default_questions_for_chunk(chunk: Dict[str, str], topic: str) -> List[str]:
    base = chunk["text"].split("\n")[0].strip()
    q1 = f"How do I use {chunk['keywords'][0] if chunk['keywords'] else 'Slurm'} for {topic.replace('slurm.', '').replace('_', ' ')}?"
    q2 = f"What is a safe example command for {topic.replace('slurm.', '')} based on: {base[:80]}?"
    return [q1, q2]


def _parse_questions(text: str) -> List[str]:
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [str(q).strip() for q in data if str(q).strip()]
    except Exception:
        return []
    return []


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def _grounded_in_chunks(item: Dict[str, str], chunk_map: Dict[str, str]) -> bool:
    answer = item.get("output", "")
    citations = item.get("citations", [])
    if not citations:
        return False
    ans_tokens = set(_tokenize(answer))
    if not ans_tokens:
        return False
    for cid in citations:
        chunk_text = chunk_map.get(cid, "")
        if not chunk_text:
            continue
        chunk_tokens = set(_tokenize(chunk_text))
        if not chunk_tokens:
            continue
        overlap = ans_tokens & chunk_tokens
        if len(overlap) >= 3:
            return True
    return False


def run_generate(
    index_dir: str,
    hypothesis_path: str,
    out_path: str,
    llm: LLMClient = None,
    question_limit: int = 0,
) -> None:
    cfg = load_hypothesis_config(hypothesis_path)
    rng = random.Random(cfg.get("seed", 0))
    question_index_dir = cfg.get("question_index") or index_dir
    answer_index_dir = cfg.get("answer_index") or index_dir
    q_chunks, q_index = load_index(question_index_dir)
    a_chunks, a_index = load_index(answer_index_dir)
    if llm is None:
        model_cfg = cfg.get("model", {})
        if model_cfg.get("generator") == "openai":
            llm = OpenAIClient(
                model=model_cfg.get("openai_model") or "gpt-4.1-mini",
                api_key=model_cfg.get("openai_api_key") or None,
            )
        else:
            llm = StubLLMClient(seed=cfg.get("seed", 0))

    topics = _assign_topic(cfg.get("topics", {}), cfg.get("seed", 0))
    difficulty_cycle = _difficulty_cycle(cfg.get("quotas", {}))

    items: List[Dict[str, str]] = []
    question_limit = int(question_limit or cfg.get("question_limit") or 0)
    for i, topic in enumerate(topics):
        query = " ".join(cfg.get("topics", {}).get(topic, {}).get("commands", [])) or topic
        q_retrieved = retrieve(query, q_chunks, q_index, top_k=2)
        if not q_retrieved:
            continue
        q_chunk = q_retrieved[0]

        prompt = qgen_prompt(q_chunk["text"], topic, cfg.get("rules", {}))
        qgen_text = llm.generate(prompt).text.strip()
        questions = _parse_questions(qgen_text)
        if not questions:
            questions = _default_questions_for_chunk(q_chunk, topic)
        for q in questions:
            if question_limit and len(items) >= question_limit:
                break
            a_retrieved = retrieve(q, a_chunks, a_index, top_k=2)
            if not a_retrieved:
                continue
            a_chunk = a_retrieved[0]
            ans_prompt = answer_prompt(q, a_chunk["text"], cfg.get("rules", {}))
            llm_answer = llm.generate(ans_prompt).text.strip()
            example_cmd = a_chunk["keywords"][0] if a_chunk["keywords"] else "srun"
            if topic == "slurm.interactive":
                example_line = "Example: srun --pty -p <partition> -A <account> <command>"
            else:
                example_line = f"Example: {example_cmd} -p <partition> -A <account> <command>"
            if not llm_answer:
                answer = f"Use placeholders for site-specific flags.\n{example_line}"
            else:
                answer = llm_answer
                needs_placeholders = "<partition>" not in answer or "<account>" not in answer
                needs_interactive = topic == "slurm.interactive" and "srun --pty" not in answer
                if (needs_placeholders or needs_interactive) and "Example:" not in answer:
                    answer = f"{answer}\n{example_line}"
            items.append({
                "instruction": q,
                "output": answer,
                "topic": topic,
                "difficulty": difficulty_cycle[i % len(difficulty_cycle)],
                "citations": [a_chunk["chunk_id"]],
            })
        if question_limit and len(items) >= question_limit:
            break

    write_jsonl(items, out_path)


def run_curate(
    in_path: str,
    out_train: str,
    out_eval: str,
    manifest_path: str,
    judge_mode: str = "stub",
    human_review_out: str = "",
    human_review_in: str = "",
    index_dir: str = "",
) -> None:
    items: List[Dict[str, str]] = []
    with Path(in_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    curated: List[Dict[str, str]] = []
    review_queue: List[Dict[str, str]] = []
    chunk_map: Dict[str, str] = {}
    if index_dir:
        chunks, _ = load_index(index_dir)
        chunk_map = {c["chunk_id"]: c["text"] for c in chunks}
    for item in items:
        notes = deterministic_validators(item)
        if chunk_map and not _grounded_in_chunks(item, chunk_map):
            notes.append("Answer not grounded in cited chunks.")
        verdict = {"keep": True, "fix": ""}
        if judge_mode == "stub":
            verdict = judge_stub(item["instruction"], item["output"])
        if notes:
            verdict["keep"] = False

        if judge_mode == "human":
            review_queue.append({
                "instruction": item["instruction"],
                "output": item["output"],
                "citations": item.get("citations", []),
                "topic": item.get("topic", ""),
                "difficulty": item.get("difficulty", ""),
                "validator_notes": notes,
                "decision": "",
                "revised_output": "",
            })
            continue

        if judge_mode != "off" and verdict.get("keep"):
            item["output"] = apply_repair(item["output"], verdict.get("fix", ""))
            curated.append(item)
        elif judge_mode == "off":
            curated.append(item)

    if review_queue and human_review_out:
        write_jsonl(review_queue, human_review_out)

    if human_review_in:
        reviewed: List[Dict[str, str]] = []
        with Path(human_review_in).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                reviewed.append(json.loads(line))
        curated = []
        for item in reviewed:
            if item.get("decision") == "keep":
                curated.append({
                    "instruction": item.get("instruction", ""),
                    "output": item.get("revised_output") or item.get("output", ""),
                    "topic": item.get("topic", ""),
                    "difficulty": item.get("difficulty", ""),
                    "citations": item.get("citations", []),
                })

    curated = exact_dedupe(curated)
    curated = token_jaccard_dedupe(curated)

    train_items, eval_items = split_train_eval(curated)
    write_jsonl(train_items, out_train)
    write_jsonl(eval_items, out_eval)
    write_manifest(curated, manifest_path)


def run_full(inputs: List[str], hypothesis_path: str, out_dir: str) -> None:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    corpus_path = str(out_dir_path / "corpus.jsonl")
    index_dir = str(out_dir_path / "index")
    candidates_path = str(out_dir_path / "candidates.jsonl")
    train_path = str(out_dir_path / "train.jsonl")
    eval_path = str(out_dir_path / "eval.jsonl")
    manifest_path = str(out_dir_path / "manifest.jsonl")

    run_ingest(inputs, corpus_path)
    run_build_index(corpus_path, index_dir)
    run_generate(index_dir, hypothesis_path, candidates_path)
    run_curate(candidates_path, train_path, eval_path, manifest_path)
