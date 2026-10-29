"""Shared configuration, cleaning and data-preparation helpers.

Used by ``01_data_exploration.ipynb`` and ``02_data_preparation.ipynb`` so both
notebooks read exactly the same configuration and apply exactly the same
cleaning rules.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_EDA_DIR = PROJECT_ROOT / "results" / "eda"

MANIFEST_NAME = "experiment_manifest.json"
CLEANED_NAME = "cleaned.csv.gz"
INDEX_NAME = "raw_row_id"

DAYS_PER_YEAR = 365.25 

FEATURE_KINDS = ("binary", "ordinal", "nominal", "numeric")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetConfig:
    """Everything the pipeline needs to know about one dataset.

    Column names refer to the *raw* file. ``cleaned_feature_types`` returns the
    names after renaming (e.g. ``age`` -> ``age_years``).
    """

    name: str
    file: str  # relative to PROJECT_ROOT
    target: str
    class_labels: tuple[str, str]  # (label for 0, label for 1)
    sep: str = ","
    id_columns: tuple[str, ...] = ()
    binary: tuple[str, ...] = ()
    ordinal: tuple[str, ...] = ()
    nominal: tuple[str, ...] = ()
    numeric: tuple[str, ...] = ()
    # Inclusive validity bounds per column; rows outside are removed.
    range_rules: dict[str, tuple[float, float]] = field(default_factory=dict)
    # (higher, lower): rows where ``higher <= lower`` are removed.
    ordered_pair: tuple[str, str] | None = None
    # {raw_column: new_name}: values are converted from days to years.
    days_to_years: dict[str, str] = field(default_factory=dict)
    # {column: {old_value: new_value}} applied after cleaning.
    binary_recode: dict[str, dict[int, int]] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return PROJECT_ROOT / self.file

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return (*self.binary, *self.ordinal, *self.nominal, *self.numeric)

    @property
    def modeling_columns(self) -> tuple[str, ...]:
        """Features + target (i.e. everything except identifiers)."""
        return (*self.feature_columns, self.target)

    @property
    def expected_columns(self) -> set[str]:
        return {*self.id_columns, *self.modeling_columns}

    def cleaned_feature_types(self) -> dict[str, list[str]]:
        rename = self.days_to_years
        return {
            kind: [rename.get(col, col) for col in getattr(self, kind)]
            for kind in FEATURE_KINDS
        }


DATASETS: dict[str, DatasetConfig] = {
    "diabetes": DatasetConfig(
        name="diabetes",
        file="data/raw/diabetes/diabetes.csv",
        target="Diabetes_binary",
        class_labels=("No diabetes", "Diabetes"),
        binary=(
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk",
            "Sex",
        ),
        ordinal=("GenHlth", "Education", "Income", "Age"),
        numeric=("BMI", "MentHlth", "PhysHlth"),
        
    ),
    "cardiovascular": DatasetConfig(
        name="cardiovascular",
        file="data/raw/cardiovascular/cardiovascular.csv",
        sep=";",
        target="cardio",
        class_labels=("No CVD", "CVD"),
        id_columns=("id",),
        binary=("gender", "smoke", "alco", "active"),
        ordinal=("cholesterol", "gluc"),  # 1 = normal, 2 = above, 3 = well above
        numeric=("age", "height", "weight", "ap_hi", "ap_lo"),
        # EXHEART ranges
        range_rules={
            "ap_hi": (60, 250),    # systolic BP, mmHg
            "ap_lo": (40, 160),    # diastolic BP, mmHg
            "height": (100, 220),  # cm
            "weight": (30, 200),   # kg
        },
        ordered_pair=("ap_hi", "ap_lo"),  # systolic must exceed diastolic
        days_to_years={"age": "age_years"},
        binary_recode={"gender": {1: 0, 2: 1}},  # 1/2 -> 0/1, no relabelling
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    """Split / subsampling design (mirrors the thesis exposé)."""

    test_size: float = 0.20
    test_seed: int = 42
    training_seeds: tuple[int, ...] = (42, 123, 456, 789, 1011)
    training_sizes: tuple[int, ...] = tuple(2**k for k in range(3, 14))  # 8..8192


EXPERIMENT = ExperimentConfig()


# --------------------------------------------------------------------------- #
# Loading and auditing
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, for raw-data provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """Read a raw CSV and validate its columns against the configuration.

    The row position in the file becomes the index (``raw_row_id``).
    Pandas' default NA tokens are used.
    """
    if not cfg.path.exists():
        raise FileNotFoundError(f"{cfg.name}: raw file not found at {cfg.path}")

    df = pd.read_csv(cfg.path, sep=cfg.sep)
    df.index.name = INDEX_NAME

    missing = cfg.expected_columns - set(df.columns)
    unexpected = set(df.columns) - cfg.expected_columns
    if missing or unexpected:
        raise ValueError(
            f"{cfg.name}: column mismatch between file and config. "
            f"Missing from file: {sorted(missing)}; not in config: {sorted(unexpected)}."
        )
    return df


def _is_integer_valued(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True
    if pd.api.types.is_float_dtype(series):
        values = series.to_numpy()
        return bool(np.isfinite(values).all() and (values == np.floor(values)).all())
    return False


def column_audit(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """One row per column: declared type vs what the data actually looks like."""
    declared = {col: kind for kind in FEATURE_KINDS for col in getattr(cfg, kind)}
    declared.update({col: "id" for col in cfg.id_columns})
    declared[cfg.target] = "target"

    rows = {}
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique())
        note = ""
        if declared.get(col) == "binary" and n_unique > 2:
            note = "declared binary but has >2 distinct values"
        rows[col] = {
            "declared": declared.get(col, "?"),
            "dtype": str(series.dtype),
            "n_unique": n_unique,
            "n_missing": int(series.isna().sum()),
            "min": series.min(),
            "max": series.max(),
            "integer_valued": _is_integer_valued(series),
            "note": note,
        }
    return pd.DataFrame(rows).T


def iqr_outlier_summary(df: pd.DataFrame, columns, k: float = 1.5) -> pd.DataFrame:
    """Tukey-fence outlier counts. A screening tool only: skewed or
    zero-inflated columns will flag many legitimate values."""
    rows = {}
    for col in columns:
        series = df[col]
        q1, q3 = series.quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - k * spread, q3 + k * spread
        below, above = int((series < lower).sum()), int((series > upper).sum())
        rows[col] = {
            "min": series.min(), "q1": q1, "q3": q3, "max": series.max(),
            "lower_fence": lower, "upper_fence": upper,
            "n_below": below, "n_above": above,
            "% outside fences": round(100 * (below + above) / len(series), 2),
        }
    return pd.DataFrame(rows).T


def top_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Unique feature pairs with |correlation| >= threshold, strongest first."""
    upper = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = (
        corr.where(upper)
        .stack()
        .rename_axis(["feature_a", "feature_b"])
        .rename("rho")
        .reset_index()
    )
    pairs = pairs[pairs["rho"].abs() >= threshold]
    return pairs.sort_values("rho", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def validity_violations(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Boolean frame, one column per validity rule (True = row violates it)."""
    checks = {}
    for col, (low, high) in cfg.range_rules.items():
        checks[f"{col} outside [{low:g}, {high:g}]"] = ~df[col].between(low, high)
    if cfg.ordered_pair is not None:
        higher, lower = cfg.ordered_pair
        checks[f"{higher} <= {lower}"] = df[higher] <= df[lower]
    return pd.DataFrame(checks, index=df.index)


def downcast_integers(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    """Downcast integer-valued columns to reduce memory usage.

    Casts binary, ordinal, nominal, and target columns to int8.
    Casts integer-valued numeric columns to int16 if they fit within range.
    Raises ValueError if required integer columns contain non-integers or overflow.
    """
    types = cfg.cleaned_feature_types()
    small = [*types["binary"], *types["ordinal"], *types["nominal"], cfg.target]
    plan = {col: (np.int8, True) for col in small}
    plan.update({col: (np.int16, False) for col in types["numeric"]})

    out = df.copy()
    for col, (dtype, must_be_integer) in plan.items():
        series = out[col]
        if not _is_integer_valued(series):
            if must_be_integer:
                raise ValueError(f"{cfg.name}: column '{col}' must be integer valued.")
            continue
        info = np.iinfo(dtype)
        if series.min() < info.min or series.max() > info.max:
            if must_be_integer:
                raise ValueError(f"{cfg.name}: column '{col}' does not fit {dtype.__name__}.")
            continue
        out[col] = series.astype(dtype)
    return out


@dataclass
class CleaningResult:
    data: pd.DataFrame
    log: pd.DataFrame          # one row per step: rows removed, target prevalence
    rule_violations: pd.Series  # rows violating each rule (measured after dedup)


def _log_row(step: str, rows_before: int, frame: pd.DataFrame, target: str) -> dict:
    removed = rows_before - len(frame)
    return {
        "step": step,
        "rows_before": rows_before,
        "rows_removed": removed,
        "rows_after": len(frame),
        "pct_removed": round(100 * removed / rows_before, 2),
        "target_prevalence": float(frame[target].mean()),
    }


def apply_cleaning(df: pd.DataFrame, cfg: DatasetConfig) -> CleaningResult:
    """Deduplicate, apply validity rules, transform and downcast.

    Order: (1) exact duplicates, (2) validity rules, (3) column transforms,
    (4) dtype downcast. The raw row index is preserved throughout.

    Duplicates are detected on the *modelling columns* (features + target),
    i.e. identifier columns are ignored. 
    """
    target = cfg.target
    n_raw = len(df)
    log = [{
        "step": "raw", "rows_before": n_raw, "rows_removed": 0, "rows_after": n_raw,
        "pct_removed": 0.0, "target_prevalence": float(df[target].mean()),
    }]

    # 1. Exact duplicates (keep first occurrence).
    duplicate_mask = df.duplicated(subset=list(cfg.modeling_columns), keep="first")
    deduped = df.loc[~duplicate_mask]
    log.append(_log_row("1. drop exact duplicates", n_raw, deduped, target))

    # 2. Validity rules.
    violations = validity_violations(deduped, cfg)
    if violations.shape[1]:
        valid = deduped.loc[~violations.any(axis=1)]
        rule_counts = violations.sum().astype(int)
    else:
        valid = deduped
        rule_counts = pd.Series(dtype=int)
    log.append(_log_row("2. apply validity rules", len(deduped), valid, target))

    # 3. Column transforms.
    cleaned = valid.rename(columns=cfg.days_to_years)
    for new_name in cfg.days_to_years.values():
        cleaned[new_name] = cleaned[new_name] / DAYS_PER_YEAR
    for col, mapping in cfg.binary_recode.items():
        recoded = cleaned[col].map(mapping)
        if recoded.isna().any():
            raise ValueError(f"{cfg.name}: '{col}' has values not covered by {mapping}.")
        cleaned[col] = recoded

    # 4. Compact dtypes.
    cleaned = downcast_integers(cleaned, cfg)
    return CleaningResult(cleaned, pd.DataFrame(log), rule_counts)


# --------------------------------------------------------------------------- #
# Features / target / split / subsampling
# --------------------------------------------------------------------------- #
def assert_binary_target(y: pd.Series, name: str = "") -> None:
    """Fail loudly before any cast could silently truncate the target."""
    if not y.isin([0, 1]).all():
        raise ValueError(f"{name}: target must contain only 0/1 (no NaN, no fractions).")


def split_features_target(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, pd.Series]:
    assert_binary_target(df[cfg.target], cfg.name)
    y = df[cfg.target].astype("int8")
    X = df.drop(columns=[cfg.target, *cfg.id_columns])
    return X, y


def stratified_split_indices(y: pd.Series, exp: ExperimentConfig = EXPERIMENT):
    """Stratified train/test split on index labels (row traceability)."""
    return train_test_split(
        y.index, test_size=exp.test_size, random_state=exp.test_seed, stratify=y
    )


def _subset_random_state(size: int, seed: int) -> int:
    """Deterministic, independent random state for one (size, seed) cell.

    Combines ``size`` and ``seed`` through a fixed hash so every cell draws
    from an unrelated stream: the draw for size 512 shares no relationship
    with the draw for size 1024 at the same seed.
    """
    digest = hashlib.sha256(f"{size}-{seed}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def make_balanced_subsets(
    y_train: pd.Series, sizes, seeds
) -> dict[int, dict[int, list[int]]]:
    """Independent class-balanced random subsets of the training pool.

    Returns ``{size: {seed: [index labels]}}``. Every ``(size, seed)`` cell is
    drawn independently.
    """
    classes = sorted(y_train.unique())
    n_classes = len(classes)
    subsets: dict[int, dict[int, list[int]]] = {}
    for size in sizes:
        if size % n_classes:
            raise ValueError(f"Training size {size} is not divisible by {n_classes} classes.")
        per_class = size // n_classes
        subsets[size] = {}
        for seed in seeds:
            state = _subset_random_state(size, seed)
            drawn: list[int] = []
            for cls in classes:
                pool = y_train[y_train == cls]
                if len(pool) < per_class:
                    raise ValueError(
                        f"Class {cls} has {len(pool)} training rows; {per_class} needed for size {size}."
                    )
                drawn.extend(pool.sample(n=per_class, random_state=state).index.tolist())
            # Shuffle so classes are not contiguous.
            subsets[size][seed] = pd.Series(drawn).sample(frac=1, random_state=state).tolist()
    return subsets


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_subsets(subsets, X_train, y_train, test_index) -> int:
    """Structural integrity checks; returns the number of subsets verified."""
    train_set, test_set = set(X_train.index), set(test_index)
    n_classes = y_train.nunique()
    checked = 0
    for size, by_seed in subsets.items():
        for seed, idx in by_seed.items():
            tag = f"size={size}, seed={seed}"
            idx_set = set(idx)
            _require(len(idx) == size, f"wrong length ({tag})")
            _require(len(idx_set) == size, f"repeated rows inside subset ({tag})")
            _require(idx_set <= train_set, f"rows outside the training pool ({tag})")
            _require(idx_set.isdisjoint(test_set), f"overlap with test set ({tag})")
            X_sub, y_sub = X_train.loc[idx], y_train.loc[idx]
            counts = y_sub.value_counts()
            _require(len(counts) == n_classes, f"missing class ({tag})")
            _require((counts == size // n_classes).all(), f"not balanced ({tag})")
            _require(not X_sub.isna().any().any(), f"NaN in features ({tag})")
            _require(not y_sub.isna().any(), f"NaN in target ({tag})")
            _require(X_sub.shape[1] == X_train.shape[1], f"feature count mismatch ({tag})")
            checked += 1
    return checked


def feature_overlap_report(X_train, y_train, X_test, y_test) -> dict[str, float]:
    """How often does a test row share its exact feature vector with the train pool?
    """
    hash_train = pd.util.hash_pandas_object(X_train, index=False).to_numpy()
    hash_test = pd.util.hash_pandas_object(X_test, index=False).to_numpy()

    train_label_mean = (
        pd.DataFrame({"h": hash_train, "y": y_train.to_numpy()}).groupby("h")["y"].mean()
    )
    test_frame = pd.DataFrame({"h": hash_test, "y": y_test.to_numpy()})
    matched = test_frame[test_frame["h"].isin(train_label_mean.index)]
    majority = (matched["h"].map(train_label_mean) >= 0.5).astype(int)
    return {
        "test_rows": float(len(test_frame)),
        "test_rows_with_identical_train_profile": float(len(matched)),
        "share_with_identical_train_profile": len(matched) / len(test_frame),
        "label_agreement": float((majority == matched["y"]).mean()) if len(matched) else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def environment_info() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_experiment(
    cleaned: dict[str, pd.DataFrame],
    splits: dict[str, tuple],
    subsets: dict[str, dict],
    cleaning_logs: dict[str, pd.DataFrame],
    raw_hashes: dict[str, str],
    exp: ExperimentConfig = EXPERIMENT,
    out_dir: Path = PROCESSED_DIR,
) -> Path:
    """Write cleaned data (gzipped CSV) and a JSON manifest; return its path.

    ``splits[name] = (train_index, test_index)``. Subsamples are stored as
    index lists only, never as copies of the data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_classes = {int(df[DATASETS[name].target].nunique()) for name, df in cleaned.items()}
    if len(n_classes) != 1:
        raise ValueError(f"Datasets disagree on number of classes: {n_classes}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment_info(),
        "config": {
            "test_size": exp.test_size,
            "test_seed": exp.test_seed,
            "training_seeds": list(exp.training_seeds),
            "training_sizes": list(exp.training_sizes),
            "n_classes": n_classes.pop(),
        },
        "datasets": {},
    }

    for name, df in cleaned.items():
        cfg = DATASETS[name]
        train_idx, test_idx = splits[name]
        dataset_dir = out_dir / name
        dataset_dir.mkdir(exist_ok=True)
        df.to_csv(dataset_dir / CLEANED_NAME, index_label=INDEX_NAME, compression="gzip")

        manifest["datasets"][name] = {
            "raw_file": cfg.file,
            "raw_sha256": raw_hashes[name],
            "target": cfg.target,
            "id_columns": list(cfg.id_columns),
            "feature_types": cfg.cleaned_feature_types(),
            "cleaned_file": f"{name}/{CLEANED_NAME}",
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "n_rows_cleaned": len(df),
            "cleaning_log": cleaning_logs[name].to_dict("records"),
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "subsets": {
                str(size): {str(seed): [int(i) for i in idx] for seed, idx in by_seed.items()}
                for size, by_seed in subsets[name].items()
            },
        }

    manifest_path = out_dir / MANIFEST_NAME
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, default=_json_default)
    return manifest_path


def load_prepared_experiment(in_dir: Path = PROCESSED_DIR) -> dict:
    """Rebuild the experiment from disk.

    Supports both:
    - the original compressed ``cleaned.csv.gz`` files
    - uncompressed ``cleaned.csv`` files created when Kaggle unpacks .gz files
    """
    in_dir = Path(in_dir)

    manifest_path = in_dir / MANIFEST_NAME

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Experiment manifest not found:\n{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    prepared = {}
    split_data = {}
    training_subsets = {}

    for name, meta in manifest["datasets"].items():

        # ------------------------------------------------------------
        # Locate cleaned dataset
        # ------------------------------------------------------------

        cleaned_path = in_dir / meta["cleaned_file"]

        
        # cleaned.csv.gz -> cleaned.csv
        if not cleaned_path.exists() and cleaned_path.suffix == ".gz":

            uncompressed_path = cleaned_path.with_suffix("")

            if uncompressed_path.exists():
                cleaned_path = uncompressed_path

        if not cleaned_path.exists():
            raise FileNotFoundError(
                f"{name}: cleaned data file not found.\n"
                f"Tried:\n"
                f"  {in_dir / meta['cleaned_file']}\n"
                f"  {(in_dir / meta['cleaned_file']).with_suffix('')}"
            )

        # ------------------------------------------------------------
        # Load cleaned data
        # ------------------------------------------------------------

        df = pd.read_csv(
            cleaned_path,
            index_col=INDEX_NAME,
            dtype=meta["dtypes"],
            float_precision="round_trip",
        )

        # ------------------------------------------------------------
        # Features and target
        # ------------------------------------------------------------

        y = df[meta["target"]]

        X = df.drop(
            columns=[
                meta["target"],
                *meta["id_columns"],
            ]
        )

        # ------------------------------------------------------------
        # Rebuild fixed train/test split
        # ------------------------------------------------------------

        train_idx = pd.Index(
            meta["train_indices"],
            name=INDEX_NAME,
        )

        test_idx = pd.Index(
            meta["test_indices"],
            name=INDEX_NAME,
        )

        # ------------------------------------------------------------
        # Store prepared data
        # ------------------------------------------------------------

        prepared[name] = {
            "X": X,
            "y": y,
        }

        split_data[name] = {
            "X_train": X.loc[train_idx],
            "X_test": X.loc[test_idx],
            "y_train": y.loc[train_idx],
            "y_test": y.loc[test_idx],
            "train_indices": train_idx,
            "test_indices": test_idx,
        }

        # ------------------------------------------------------------
        # Rebuild predefined training subsets
        # ------------------------------------------------------------

        training_subsets[name] = {
            int(size): {
                int(seed): {
                    "X": X.loc[idx],
                    "y": y.loc[idx],
                    "indices": idx,
                }
                for seed, idx in by_seed.items()
            }
            for size, by_seed in meta["subsets"].items()
        }

    # ------------------------------------------------------------
    # Final experiment object
    # ------------------------------------------------------------

    return {
        "prepared_data": prepared,
        "split_data": split_data,
        "training_subsets": training_subsets,
        "config": manifest["config"],
        "manifest": manifest,
}
