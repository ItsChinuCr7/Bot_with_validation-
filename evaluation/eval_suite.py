import argparse
import json
import logging
import math
from pathlib import Path

from config.settings import EMBEDDING_MODEL, LLM_MODEL, VECTOR_INDEX_PATH
from observability.experiment_tracking import log_experiment

logger = logging.getLogger(__name__)
REQUIRED_FIELDS = {"question", "ground_truth"}


def load_dataset(path: str) -> list[dict]:
    dataset_path = Path(path)
    with dataset_path.open(encoding="utf-8") as stream:
        examples = json.load(stream)
    if not isinstance(examples, list) or not examples:
        raise ValueError("Evaluation dataset must be a non-empty JSON array.")
    for index, example in enumerate(examples):
        missing = REQUIRED_FIELDS - example.keys()
        if missing:
            raise ValueError(f"Dataset row {index} is missing: {', '.join(sorted(missing))}")
        if "answer" in example and "contexts" in example:
            if not isinstance(example["contexts"], list) or not example["contexts"]:
                raise ValueError(f"Dataset row {index} must have at least one context.")
    return examples


def _collect_local_rag_outputs(examples: list[dict], model_name: str) -> list[dict]:
    if not Path(VECTOR_INDEX_PATH).exists():
        raise FileNotFoundError(
            f"No FAISS index at {VECTOR_INDEX_PATH}; set VECTOR_INDEX_PATH to an indexed corpus."
        )
    from langchain_huggingface import HuggingFaceEmbeddings
    from models.model_factory import create_llm
    from reasoning.rag_pipeline import run_rag_pipeline
    from retrieval.retriever import get_mmr_retriever
    from retrieval.vector_store import load_vector_store

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vector_db = load_vector_store(embeddings)
    retriever = get_mmr_retriever(vector_db)
    llm = create_llm(model_name=model_name, mode="light")
    generated = []
    for example in examples:
        answer, _, _, _, context, _ = run_rag_pipeline(
            llm, retriever, example["question"], return_metadata=True
        )
        generated.append({
            **example,
            "answer": answer,
            "contexts": [
                section.strip()
                for section in context.split("\n\n---\n\n")
                if section.strip()
            ],
        })
    return generated


def evaluate_dataset(
    dataset_path: str,
    model_name: str,
    output_path: str,
    query_index: bool = False,
) -> dict:
    examples = load_dataset(dataset_path)
    if query_index:
        examples = _collect_local_rag_outputs(examples, model_name)
    elif any("answer" not in example or "contexts" not in example for example in examples):
        raise ValueError(
            "Precomputed evaluation rows need answer and contexts; use --query-index "
            "to generate them from the local FAISS index."
        )
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RAG evaluation needs the optional dependencies. "
            "Install them with: pip install -r requirements-ml.txt"
        ) from exc

    llm = LangchainLLMWrapper(ChatOllama(model=model_name))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    )
    result = evaluate(
        Dataset.from_list(examples),
        metrics=[
            faithfulness,
            context_precision,
            answer_relevancy,
            answer_correctness,
        ],
        llm=llm,
        embeddings=embeddings,
    )
    score_frame = result.to_pandas()
    rows = score_frame.to_dict(orient="records")
    averages = {
        metric: float(score_frame[metric].dropna().mean())
        for metric in (
            "faithfulness",
            "context_precision",
            "answer_relevancy",
            "answer_correctness",
        )
    }
    if any(not math.isfinite(score) for score in averages.values()):
        raise RuntimeError("RAGAS returned a missing or non-finite metric score.")
    report = {
        "model": model_name,
        "dataset": str(Path(dataset_path)),
        "examples": len(examples),
        "scores": averages,
        "results": rows,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log_experiment(
        run_name=f"ragas-eval-{model_name}",
        params={"model": model_name, "dataset": dataset_path, "query_index": query_index},
        metrics=averages,
        artifacts=[str(destination)],
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers with RAGAS.")
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/sample_rag_eval.json",
        help="JSON dataset with question, answer, contexts, and ground_truth fields.",
    )
    parser.add_argument("--model", default=LLM_MODEL, help="Locally installed Ollama model.")
    parser.add_argument("--output", default="evaluation/results/latest.json")
    parser.add_argument("--min-faithfulness", type=float, default=0.8)
    parser.add_argument(
        "--query-index",
        action="store_true",
        help="Generate answers and retrieved contexts from the configured FAISS index.",
    )
    args = parser.parse_args()

    report = evaluate_dataset(args.dataset, args.model, args.output, args.query_index)
    for name, score in report["scores"].items():
        print(f"{name}: {score:.3f}")
    faithfulness = report["scores"]["faithfulness"]
    if faithfulness < args.min_faithfulness:
        logger.error(
            "Faithfulness %.3f is below required threshold %.3f",
            faithfulness,
            args.min_faithfulness,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
