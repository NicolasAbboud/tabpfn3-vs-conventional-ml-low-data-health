"""
Central experimental protocol.

This module is intentionally separate from ``data_pipeline.py``:

- data_pipeline.py      -> data ingestion, cleaning, fixed train/test split,
                           balanced training subsets, provenance
- experiment_protocol.py -> experiment design, seeds, metrics, model-selection
                            rules, and run-count expectations

All downstream experiment notebooks (03, 04, 05) should import this module
instead of redefining these constants independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final
from data_pipeline import EXPERIMENT, DATASETS

# ---------------------------------------------------------------------------
# Protocol identity
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: Final[str] = "low_data_tabpfn_v1"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

EXPECTED_DATASETS: Final[tuple[str, ...]] = tuple(DATASETS.keys())

TEST_SIZE: Final[float] = EXPERIMENT.test_size
TEST_SEED: Final[int] = EXPERIMENT.test_seed

SUBSET_SEEDS: Final[tuple[int, ...]] = EXPERIMENT.training_seeds

TRAINING_SIZES: Final[tuple[int, ...]] = EXPERIMENT.training_sizes


# ---------------------------------------------------------------------------
# Randomness control
# ---------------------------------------------------------------------------
#
# IMPORTANT:
#   subset_seed -> controls which rows are selected for a training subset
#   model_seed  -> controls model-side randomness
#   cv_seed     -> controls shuffled CV folds
#   hpo_seed    -> controls RandomizedSearchCV parameter sampling

MODEL_SEED: Final[int] = 42
CV_SEED: Final[int] = 42
HPO_SEED: Final[int] = 42


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

CLASSIFICATION_THRESHOLD: Final[float] = 0.50

# Primary ranking/evaluation metrics for the thesis.
PRIMARY_METRICS: Final[tuple[str, ...]] = (
    "roc_auc",
    "pr_auc",
)

# Secondary threshold-dependent metrics.
SECONDARY_METRICS: Final[tuple[str, ...]] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
)


# ---------------------------------------------------------------------------
# Hyperparameter optimisation
# ---------------------------------------------------------------------------

# HPO is intentionally not run for the extremely small 8- and 16-example
HPO_MIN_TRAINING_SIZE: Final[int] = 32

HPO_TRAINING_SIZES: Final[tuple[int, ...]] = tuple(
    size for size in TRAINING_SIZES
    if size >= HPO_MIN_TRAINING_SIZE
)

HPO_N_ITER: Final[int] = 20
HPO_CV_SPLITS: Final[int] = 5
HPO_SCORING: Final[str] = "roc_auc"


# ---------------------------------------------------------------------------
# Model identity / execution settings
# ---------------------------------------------------------------------------

TRADITIONAL_MODELS: Final[tuple[str, ...]] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)

# Baseline experiment uses non-optimised model configurations.
LOGISTIC_REGRESSION_MAX_ITER: Final[int] = 1000
XGBOOST_OBJECTIVE: Final[str] = "binary:logistic"
XGBOOST_EVAL_METRIC: Final[str] = "logloss"

TABPFN_MODEL_VERSION: Final[str] = "V3"


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------

# Core identity + metric columns shared by ALL three result tables.
CORE_RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "protocol_version",
    "dataset",
    "model",
    "training_size",
    "subset_seed",
    "model_seed",
    "classification_threshold",
    *PRIMARY_METRICS,
    *SECONDARY_METRICS,
)

# Only the HPO table carries these.
HPO_EXTRA_COLUMNS: Final[tuple[str, ...]] = ("cv_seed", "hpo_seed", "best_cv_score", "best_params")

BASELINE_RESULT_COLUMNS: Final[tuple[str, ...]] = CORE_RESULT_COLUMNS
TABPFN_RESULT_COLUMNS: Final[tuple[str, ...]] = CORE_RESULT_COLUMNS
HPO_RESULT_COLUMNS: Final[tuple[str, ...]] = CORE_RESULT_COLUMNS + HPO_EXTRA_COLUMNS


# ---------------------------------------------------------------------------
# Derived experiment counts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExperimentProtocol:
    """Immutable view of the central thesis experiment design."""

    datasets: tuple[str, ...] = EXPECTED_DATASETS
    training_sizes: tuple[int, ...] = TRAINING_SIZES
    subset_seeds: tuple[int, ...] = SUBSET_SEEDS

    test_size: float = TEST_SIZE
    test_seed: int = TEST_SEED

    model_seed: int = MODEL_SEED
    cv_seed: int = CV_SEED
    hpo_seed: int = HPO_SEED

    classification_threshold: float = CLASSIFICATION_THRESHOLD

    hpo_training_sizes: tuple[int, ...] = HPO_TRAINING_SIZES
    hpo_n_iter: int = HPO_N_ITER
    hpo_cv_splits: int = HPO_CV_SPLITS
    hpo_scoring: str = HPO_SCORING

    protocol_version: str = PROTOCOL_VERSION


PROTOCOL = ExperimentProtocol()


def expected_baseline_runs() -> int:
    """Expected number of runs for Notebook 03."""
    return (
        len(EXPECTED_DATASETS)
        * len(TRAINING_SIZES)
        * len(SUBSET_SEEDS)
        * len(TRADITIONAL_MODELS)
    )


def expected_tabpfn_runs() -> int:
    """Expected number of TabPFN runs for Notebook 04."""
    return (
        len(EXPECTED_DATASETS)
        * len(TRAINING_SIZES)
        * len(SUBSET_SEEDS)
    )


def expected_hpo_runs() -> int:
    """Expected number of HPO runs for Notebook 05."""
    return (
        len(EXPECTED_DATASETS)
        * len(HPO_TRAINING_SIZES)
        * len(SUBSET_SEEDS)
        * len(TRADITIONAL_MODELS)
    )


def validate_protocol() -> None:
    """Fail fast if the central protocol contains an invalid configuration."""

    if TEST_SIZE <= 0 or TEST_SIZE >= 1:
        raise ValueError("TEST_SIZE must be strictly between 0 and 1.")

    if not TRAINING_SIZES:
        raise ValueError("TRAINING_SIZES must not be empty.")

    if any(size <= 0 for size in TRAINING_SIZES):
        raise ValueError("All training sizes must be positive.")

    if any(size % 2 != 0 for size in TRAINING_SIZES):
        raise ValueError(
            "All training sizes must be divisible by 2 because subsets are binary-class balanced."
        )

    if not SUBSET_SEEDS:
        raise ValueError("SUBSET_SEEDS must not be empty.")

    if len(set(SUBSET_SEEDS)) != len(SUBSET_SEEDS):
        raise ValueError("SUBSET_SEEDS must be unique.")

    if HPO_MIN_TRAINING_SIZE not in TRAINING_SIZES:
        raise ValueError(
            "HPO_MIN_TRAINING_SIZE must correspond to one of TRAINING_SIZES."
        )

    if HPO_CV_SPLITS < 2:
        raise ValueError("HPO_CV_SPLITS must be at least 2.")

    if HPO_N_ITER < 1:
        raise ValueError("HPO_N_ITER must be at least 1.")

    if not 0 < CLASSIFICATION_THRESHOLD < 1:
        raise ValueError("CLASSIFICATION_THRESHOLD must be strictly between 0 and 1.")


validate_protocol()


if __name__ == "__main__":
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Datasets: {EXPECTED_DATASETS}")
    print(f"Training sizes: {TRAINING_SIZES}")
    print(f"Subset seeds: {SUBSET_SEEDS}")
    print(f"Test split: {TEST_SIZE} (seed={TEST_SEED})")
    print(f"Model seed: {MODEL_SEED}")
    print(f"CV seed: {CV_SEED}")
    print(f"HPO seed: {HPO_SEED}")
    print(f"HPO sizes: {HPO_TRAINING_SIZES}")
    print(f"TabPFN model version: {TABPFN_MODEL_VERSION}")
    print(f"Expected baseline ML runs: {expected_baseline_runs()}")
    print(f"Expected TabPFN runs: {expected_tabpfn_runs()}")
    print(f"Expected HPO runs: {expected_hpo_runs()}")
