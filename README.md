# Learning from Less: Evaluating TabPFN-3 Against Conventional Machine Learning for Low-Data Classification on Real-World Health Data

Code and experiment framework accompanying the Bachelor thesis **"Learning from Less: Evaluating TabPFN-3 Against Conventional Machine Learning for Low-Data Classification on Real-World Health Data"**.

- **Author:** Nicolas Abboud
- **Supervisor:** Dr. Aditya Mushyam
- **Institution:** IU International University of Applied Sciences
- **Study programme:** B.Sc. Applied Artificial Intelligence

---

## Overview

Labelled health data are often scarce, and conventional classifiers depend on task-specific training data. **TabPFN-3** is a tabular foundation model pretrained on synthetic tasks. It predicts through in-context learning, with no task-specific parameter updates.

This project compares TabPFN-3 with **Logistic Regression**, **Random Forest** and **XGBoost** on two real-world tabular health datasets (diabetes and cardiovascular disease) under controlled low-data conditions.

- Class-balanced training subsets of **8 to 8,192 observations** (eleven sizes, powers of two)
- **Five sampling seeds** per size
- A **single fixed external test set** per dataset
- Conventional models evaluated with **default settings** and after **hyperparameter optimization (HPO)**
- **Stability** measured as the standard deviation of ROC-AUC across seeds

## Research Questions

| # | Question |
|---|----------|
| **RQ1** | How does TabPFN-3 compare with conventional models using default hyperparameters across different low-data training sizes? |
| **RQ2** | To what extent does hyperparameter optimization narrow or reverse the performance gap between conventional models and TabPFN-3? |
| **RQ3** | How does training-set size affect the performance and seed-level stability of TabPFN-3 relative to conventional models? |

## Datasets

Both datasets are public and must be downloaded separately (they are not redistributed here).

| Dataset | Source | Records (raw → cleaned) | Features | Target |
|---------|--------|-------------------------|----------|--------|
| Diabetes Health Indicators (BRFSS 2015) | [Kaggle – Teboul (2021)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) | 253,680 → 229,474 | 21 | `Diabetes_binary` |
| Cardiovascular Disease | [Kaggle – Ulianova (2018)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) | 70,000 → 68,612 | 11 | `cardio` |

**Cleaning steps**

1. Exact-duplicate removal on the modelling columns (24,206 diabetes rows, 24 cardiovascular rows).
2. Physiological validity rules for the cardiovascular data only, following the EXHEART pipeline (Biswas & Laila, 2026): systolic BP 60–250 mmHg, diastolic BP 40–160 mmHg, height 100–220 cm, weight 30–200 kg, systolic > diastolic.
3. Age converted from days to years, and gender recoded to 0/1.
4. Integer columns downcast to compact types.

Scaling and encoding are deliberately left out of preparation. Where a model needs scaling, it is fitted inside the model pipeline on the training subset only, to avoid leakage.

## Experimental Design

1. **Data validation and cleaning:** the schema is audited and the SHA-256 hash of each raw file is stored in the manifest.
2. **Fixed stratified split:** 80/20, seed 42. The test set is never resampled and keeps its natural prevalence.
3. **Class-balanced subsampling:** sizes 8, 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096 and 8,192, with subset seeds 42, 123, 456, 789 and 1011. This gives 110 subsets in total. Subsets are independent draws, not nested, and are stored as index lists in a JSON manifest.
4. **Model training and evaluation** on the full external test set.
5. **Paired statistical analysis** (descriptive; no significance tests are used as primary evidence).

**Seed separation.** Four independent random sources are kept apart, so that variation across subset seeds reflects which training rows were drawn:

| Source | Role | Values |
|--------|------|--------|
| `subset_seed` | Selects training rows | 42, 123, 456, 789, 1011 |
| `model_seed` | Estimator randomness | 42 (fixed) |
| `cv_seed` | CV fold assignment during HPO | 42 (fixed) |
| `hpo_seed` | Hyperparameter sampling | 42 (fixed) |

All design constants (datasets, sizes, seeds, metrics, threshold, HPO settings, TabPFN version) are defined once in a central protocol module, `experiment_protocol.py` (version `low_data_tabpfn_v1`). Every experimental notebook imports it, and each run asserts that the stored data manifest matches the protocol.

## Models

| Model | Configuration |
|-------|---------------|
| **TabPFN-3** | `tabpfn` package v8.3.0, created with `TabPFNClassifier.create_default_for_version(ModelVersion.V3, …)`, `n_estimators="auto"`, `random_state=42`, package defaults otherwise. No HPO, no categorical feature indices. Checkpoint: `tabpfn-v3-classifier-v3_default.ckpt`. Inference on GPU (Tesla T4), test predictions batched in chunks of 1,000 rows. |
| **Logistic Regression** | `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)` (L2, C=1, L-BFGS) |
| **Random Forest** | `RandomForestClassifier(random_state=42)` (library defaults) |
| **XGBoost** | `XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)` (library defaults) |

### Hyperparameter optimization

Each conventional model is tuned on every training subset with **at least 32 observations** (nine sizes) using `RandomizedSearchCV`: 20 sampled configurations, stratified 5-fold CV with shuffling, ROC-AUC as selection criterion, `random_state=42`. The best configuration is refitted on the full subset and evaluated once on the external test set. The test set is never used for tuning. TabPFN-3 is not tuned.

| Model | Search space | Configurations |
|-------|--------------|----------------|
| Logistic Regression | `C` ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30}, `class_weight` ∈ {None, balanced} | 20 |
| Random Forest | `n_estimators` ∈ {100, 300}, `max_depth` ∈ {5, 10, None}, `min_samples_leaf` ∈ {1, 5}, `max_features` ∈ {sqrt, log2, None} | 36 |
| XGBoost | `max_depth` ∈ {3, 6, 9}, `learning_rate` ∈ {0.01, 0.1, 0.3}, `n_estimators` ∈ {100, 300}, `subsample` ∈ {0.8, 1.0}, `colsample_bytree` ∈ {0.8, 1.0} | 72 |

Search-space definitions are hashed and stored in the checkpoint metadata, so a run cannot be resumed under a modified search space.

In total there are **710 final test-evaluation runs**: 330 default conventional runs, 110 TabPFN-3 runs and 270 HPO runs.

## Evaluation

- **Primary metrics:** ROC-AUC and PR-AUC (average precision; chance level equals test prevalence).
- **Secondary metrics** (threshold 0.5, descriptive only): accuracy, balanced accuracy, precision, recall, F1. Specificity and predicted-positive rate are derived from stored metrics.
- **Paired differences:** TabPFN-3 minus each conventional model per dataset, size and seed, with seed-win counts (k/5) and aggregation into tiny (8–16), small (32–128), medium (256–1,024) and large (2,048–8,192) regimes.
- **Data efficiency:** the equivalent training size, i.e. how many labelled observations a conventional model needs to match TabPFN-3 at a reference size, using interpolated learning curves on a log₂ size scale.
- **Stability:** sample SD of ROC-AUC across the five subset seeds, SD ratios (conventional / TabPFN-3), and the log-SD vs. log-n slope.
- **Learning-curve fits:** descriptive saturating-curve fits on seed-mean ROC-AUC for n = 32 to 8,192.

The analysis step (Notebook 06) never refits models. It loads the three final result tables and validates them (row counts, key uniqueness, seeds, protocol version, threshold, metric ranges) before building the analysis tables and figures.

## Environment

| Component | Version |
|-----------|---------|
| Python | 3.12 |
| tabpfn | 8.3.0 |
| scikit-learn | 1.6.1 |
| XGBoost | 3.2.0 |
| NumPy | 2.0.2 |
| pandas | 2.3.3 |
| Hardware (TabPFN-3 inference) | GPU, Tesla T4 |

A GPU is recommended for TabPFN-3 inference. The conventional models run on CPU.

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/NicolasAbboud/tabpfn3-vs-conventional-ml-low-data-health.git
cd tabpfn3-vs-conventional-ml-low-data-health

# 2. Create an environment (Python 3.12) and install the dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install tabpfn==8.3.0 scikit-learn==1.6.1 xgboost==3.2.0 numpy==2.0.2 pandas==2.3.3
```

3. Download the two Kaggle datasets (links above) and place the raw files where the notebooks expect them. The loader rejects any file whose columns differ from the declared configuration.
4. Run the notebooks in order: data preparation and subset generation, then the default runs, TabPFN-3 runs, HPO runs, and finally the analysis notebook (Notebook 06).

## Reproducibility

- All constants live in `experiment_protocol.py`, so experiments are comparable by construction.
- Subset random states are derived deterministically from the training size and subset seed via SHA-256, and regenerating the subsets reproduces them exactly.
- Subsets pass automated integrity checks: correct size, no repeated rows, membership in the training pool, disjointness from the test set, exact class balance, no missing values and consistent feature dimensionality.
- A leakage audit confirmed that no row appears in both the training pool and the test set. Feature-vector overlap (1.09% of diabetes test rows, 0.06% of cardiovascular test rows) is reported, not removed.
- Raw-file SHA-256 hashes, subset indices and HPO search-space hashes are stored for provenance.
