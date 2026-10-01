from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from signova.config import (
    DEFAULT_MODEL_PATH,
    FEATURE_SCHEMA,
    MODELS_DIR,
    ensure_project_dirs,
    load_vocabulary,
)
from signova.training.dataset import DatasetManager


def build_model(model_type: str, random_state: int) -> Pipeline:
    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=260,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        classifier = MLPClassifier(
            hidden_layer_sizes=(192, 96),
            activation="relu",
            alpha=0.0008,
            batch_size="auto",
            learning_rate_init=0.001,
            max_iter=450,
            early_stopping=False,
            n_iter_no_change=25,
            random_state=random_state,
        )
    else:
        raise ValueError("model_type must be 'mlp' or 'random_forest'.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def train_model(
    dataset_dir: Path | None = None,
    output_path: Path = DEFAULT_MODEL_PATH,
    model_type: str = "mlp",
    min_samples_per_class: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    ensure_project_dirs()
    manager = DatasetManager(dataset_dir) if dataset_dir else DatasetManager()
    available_counts = {
        label: count
        for label, count in manager.sample_counts().items()
        if count > 0
    }
    insufficient = sorted(
        label for label, count in available_counts.items() if count < min_samples_per_class
    )
    if insufficient:
        raise RuntimeError(
            f"Each dataset class needs at least {min_samples_per_class} valid samples: "
            + ", ".join(insufficient)
        )
    x, y, sample_paths = manager.load_samples(min_samples_per_class=min_samples_per_class)

    if x.shape[0] == 0:
        raise RuntimeError(
            "No training samples found. Collect landmark samples before training."
        )

    class_counts = Counter(y)
    unsupported_labels = sorted(set(str(label) for label in class_counts) - set(load_vocabulary()))
    if unsupported_labels:
        raise RuntimeError(
            "Training data contains labels outside the approved vocabulary: "
            + ", ".join(unsupported_labels)
        )
    if len(class_counts) < 2:
        raise RuntimeError("At least two sign labels are required for model training.")

    if min(class_counts.values()) < min_samples_per_class:
        raise RuntimeError(
            f"Each trained class needs at least {min_samples_per_class} samples."
        )

    x_train, x_temp, y_train, y_temp = _safe_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
    )
    x_val, x_test, y_val, y_test = _safe_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
    )

    model = build_model(model_type=model_type, random_state=random_state)
    model.fit(x_train, y_train)

    metrics = _evaluate_model(model, x_train, y_train, x_val, y_val, x_test, y_test)
    metrics.update(
        {
            "model_type": model_type,
            "feature_schema": FEATURE_SCHEMA["name"],
            "feature_length": FEATURE_SCHEMA["length"],
            "classes": [str(label) for label in sorted(class_counts)],
            "class_counts": {
                str(label): int(count)
                for label, count in sorted(class_counts.items())
            },
            "sample_count": int(x.shape[0]),
            "train_count": int(x_train.shape[0]),
            "validation_count": int(x_val.shape[0]),
            "test_count": int(x_test.shape[0]),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_samples": [
                path.resolve().relative_to(manager.dataset_dir.resolve()).as_posix()
                for path in sample_paths
            ],
        }
    )

    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "loss_curve_"):
        metrics["loss_curve"] = [float(value) for value in classifier.loss_curve_]
    if hasattr(classifier, "validation_scores_"):
        metrics["validation_scores"] = [
            float(value) for value in classifier.validation_scores_
        ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "labels": [str(label) for label in sorted(class_counts)],
        "feature_schema": FEATURE_SCHEMA,
        "metrics": metrics,
    }
    joblib.dump(payload, output_path)

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    labels_path = output_path.with_name("labels.json")
    labels_path.write_text(
        json.dumps({"labels": sorted(str(label) for label in class_counts)}, indent=2),
        encoding="utf-8",
    )
    metadata_path = output_path.with_name("model_metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "feature_schema": FEATURE_SCHEMA,
                "labels": sorted(str(label) for label in class_counts),
                "metrics": metrics,
                "model_file": output_path.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_confusion_matrix_csv(
        output_path.with_suffix(".confusion_matrix.csv"),
        metrics["confusion_matrix_labels"],
        metrics["confusion_matrix"],
    )
    _write_confusion_matrix_png(
        output_path.with_suffix(".confusion_matrix.png"),
        metrics["confusion_matrix_labels"],
        metrics["confusion_matrix"],
    )

    return metrics


def _safe_split(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = Counter(y)
    stratify = y if min(counts.values()) >= 2 and len(y) >= len(counts) * 2 else None
    try:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def _evaluate_model(
    model: Pipeline,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    labels = sorted(set(y_train) | set(y_val) | set(y_test))

    metrics: dict[str, Any] = {
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "validation_accuracy": float(accuracy_score(y_val, val_pred)),
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_test, test_pred, labels=labels).tolist(),
    }

    if hasattr(model, "predict_proba"):
        try:
            classes = [str(label) for label in model.named_steps["classifier"].classes_]
            metrics["validation_loss"] = float(
                log_loss(y_val, model.predict_proba(x_val), labels=classes)
            )
            metrics["test_loss"] = float(
                log_loss(y_test, model.predict_proba(x_test), labels=classes)
            )
        except ValueError:
            metrics["validation_loss"] = None
            metrics["test_loss"] = None

    return metrics


def _write_confusion_matrix_csv(
    path: Path,
    labels: list[str],
    matrix: list[list[int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *row])


def _write_confusion_matrix_png(
    path: Path,
    labels: list[str],
    matrix: list[list[int]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    array = np.asarray(matrix)
    fig, axis = plt.subplots(figsize=(max(8, len(labels) * 0.45), 7))
    image = axis.imshow(array, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("SIGNOVA Confusion Matrix")
    for row_index in range(array.shape[0]):
        for column_index in range(array.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(array[row_index, column_index]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SIGNOVA sign classifier.")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset directory.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH, help="Model output.")
    parser.add_argument(
        "--model-type",
        choices=["mlp", "random_forest"],
        default="mlp",
        help="Classifier architecture.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=8,
        help="Minimum samples per class to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(
        dataset_dir=args.dataset,
        output_path=args.output,
        model_type=args.model_type,
        min_samples_per_class=args.min_samples,
    )
    print(f"Model saved to {args.output}")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Metrics directory: {MODELS_DIR}")


if __name__ == "__main__":
    main()
