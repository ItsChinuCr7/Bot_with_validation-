import os
from pathlib import Path

_MISSING_MLFLOW_LOGGED = False


def log_experiment(
    run_name: str,
    params: dict,
    metrics: dict,
    artifacts: list[str] | None = None,
) -> None:
    global _MISSING_MLFLOW_LOGGED
    if os.getenv("AI_ENABLE_MLFLOW", "false").lower() != "true":
        return
    try:
        import mlflow
    except ModuleNotFoundError as exc:
        if exc.name != "mlflow":
            raise
        if not _MISSING_MLFLOW_LOGGED:
            import logging
            logging.getLogger(__name__).warning(
                "MLflow tracking is enabled but MLflow is not installed. "
                "Install requirements-ml.txt."
            )
            _MISSING_MLFLOW_LOGGED = True
        return

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        str(Path(__file__).resolve().parents[1] / "mlruns"),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-inference"))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({key: str(value) for key, value in params.items() if value is not None})
        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and value is not None
        })
        for artifact in artifacts or []:
            mlflow.log_artifact(artifact)
