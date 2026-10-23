from collections.abc import Callable, Iterable

from observability.experiment_tracking import log_experiment


def run_ab_test(
    queries: Iterable[str],
    model_names: tuple[str, str],
    answer_fn: Callable[[str, str], str],
    score_fn: Callable[[str, str], float],
) -> dict:
    """Run paired queries against two models and return aggregate scores."""
    if len(model_names) != 2 or model_names[0] == model_names[1]:
        raise ValueError("A/B testing requires two distinct model names.")

    results = []
    totals = {name: 0.0 for name in model_names}
    counts = {name: 0 for name in model_names}
    for query in queries:
        candidates = {}
        for model_name in model_names:
            answer = answer_fn(model_name, query)
            score = float(score_fn(query, answer))
            if not 0.0 <= score <= 1.0:
                raise ValueError("A/B score function must return a value from 0 to 1.")
            candidates[model_name] = {"answer": answer, "score": score}
            totals[model_name] += score
            counts[model_name] += 1
        winner = max(model_names, key=lambda name: candidates[name]["score"])
        results.append({"query": query, "candidates": candidates, "winner": winner})

    averages = {
        name: totals[name] / counts[name] if counts[name] else 0.0
        for name in model_names
    }
    report = {
        "models": list(model_names),
        "average_scores": averages,
        "winner": max(model_names, key=averages.get) if results else None,
        "results": results,
    }
    log_experiment(
        run_name="paired-model-ab-test",
        params={"models": model_names, "query_count": len(results)},
        metrics={f"average_score_{name}": score for name, score in averages.items()},
    )
    return report
