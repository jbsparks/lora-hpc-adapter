from typing import Dict, List


def curriculum_prompt(hypothesis: str, topics: Dict[str, Dict[str, List[str]]], rules: Dict[str, List[str]]) -> str:
    return (
        "You are an expert curriculum designer.\n"
        f"Hypothesis: {hypothesis}\n"
        f"Topics: {list(topics.keys())}\n"
        f"Rules: {rules}\n"
        "Return a concise coverage plan with quotas per topic and difficulty."
    )


def qgen_prompt(chunk_text: str, topic: str, rules: Dict[str, List[str]]) -> str:
    return (
        "Generate 2-4 questions grounded ONLY in the provided chunk.\n"
        f"Topic: {topic}\n"
        f"Rules: {rules}\n"
        f"Chunk:\n{chunk_text}\n"
        "Return JSON list of question strings."
    )


def answer_prompt(question: str, chunk_text: str, rules: Dict[str, List[str]]) -> str:
    return (
        "Answer the question using ONLY the provided chunk.\n"
        f"Rules: {rules}\n"
        f"Question: {question}\n"
        f"Chunk:\n{chunk_text}\n"
        "Return a concise answer with command examples when relevant."
    )


def judge_prompt(question: str, answer: str, chunk_text: str, rules: Dict[str, List[str]]) -> str:
    return (
        "Judge if the answer is grounded in the chunk and follows rules.\n"
        f"Rules: {rules}\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Chunk:\n{chunk_text}\n"
        "Return JSON with fields: keep(boolean), grounded(boolean), notes(string), fix(string or empty)."
    )
