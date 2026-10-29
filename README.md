# Learning from Less: Evaluating TabPFN-3 Against Conventional Machine Learning for Low-Data Classification on Real-World Health Data

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![TabPFN](https://img.shields.io/badge/TabPFN-3-orange.svg)](https://github.com/PriorLabs/TabPFN)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-f7931e.svg)](https://scikit-learn.org/)
[![Research](https://img.shields.io/badge/Project-Bachelor's%20Thesis-lightgrey.svg)]()

## 1. Overview

The availability of labelled observations is an important constraint in supervised health classification. Although healthcare systems generate substantial quantities of data, the amount of reliable, accessible, and task-specific labelled data available for a particular prediction problem may be limited.

Conventional machine-learning models estimate task-specific predictive relationships from the observations provided during training. Tabular foundation models approach the problem differently by acquiring predictive knowledge during pretraining and applying it to previously unseen datasets through in-context learning.

This project investigates whether **TabPFN-3 can use limited labelled data more effectively than conventional machine-learning models** on two real-world health-classification datasets.

The evaluation compares TabPFN-3 with three established classifiers:

- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost

The models are evaluated across eleven training-set sizes, from 8 to 8,192 observations, under controlled experimental conditions. Conventional models are evaluated both with default configurations and after hyperparameter optimization (HPO). The study also examines how predictive performance and its variability across training subsets change as training-set size increases.

The repository contains the notebooks, shared Python modules, experimental configuration, and analysis code used in the study.

## 2. Research Questions

The experiments address three research questions.

**RQ1 — Comparative performance**

How does TabPFN-3 compare with conventional machine-learning models using default hyperparameters across different low-data training sizes?

**RQ2 — Effect of hyperparameter optimization**

To what extent does hyperparameter optimization narrow or reverse the performance gap between conventional machine-learning models and TabPFN-3?

**RQ3 — Training-set size and stability**

How does training-set size affect the predictive performance and training-subset stability of TabPFN-3 relative to conventional machine-learning models?

The study does not assume that either paradigm is inherently superior. Instead, it evaluates their relative performance empirically under a shared experimental protocol.

## 3. Experimental Design

### 3.1 Datasets

Two publicly available binary health-classification datasets are used.

| Dataset | Original records | Records after cleaning | Predictors | Target |
|---|---:|---:|---:|---|
| Diabetes Health Indicators (BRFSS 2015) | 253,680 | 229,474 | 21 | `Diabetes_binary` |
| Cardiovascular Disease | 70,000 | 68,612 | 11 | `cardio` |

**Dataset sources**

- **Diabetes Health Indicators:** [Kaggle — Diabetes Health Indicators Dataset, by Teboul (2021)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Cardiovascular Disease:** [Kaggle — Cardiovascular Disease Dataset, by Ulianova (2018)](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

### 3.2 Data preprocessing

The preprocessing pipeline includes:

1. Removal of exact duplicate records based on the modelling features and target, excluding the identifier column.
2. Application of predefined physiological validity rules to the cardiovascular dataset.
3. Conversion of age from days to years in the cardiovascular dataset and recoding of its gender variable.
4. Conversion of integer-valued columns to compact integer types where appropriate.
5. Creation of a fixed, stratified 80/20 training/test split using random seed 42.

The test partition retains the natural class prevalence and remains fixed across all experimental conditions. Model-dependent transformations, such as feature scaling, are fitted within the training pipeline to avoid leakage from test observations.

### 3.3 Low-data training subsets

Training subsets are sampled exclusively from each dataset's training partition.

The evaluated training sizes are:

`8, 16, 32, 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192`

Five subset seeds are used:

`42, 123, 456, 789, 1011`

Each subset contains equal numbers of positive and negative observations. Subsets are sampled independently across training sizes rather than constructed as nested extensions of smaller subsets.

This design supports a controlled comparison of performance across data-availability conditions. Because the test set retains the original class distribution, threshold-dependent metrics are interpreted with particular care.

### 3.4 Model evaluation

TabPFN-3 is evaluated using the V3 model version and its default configuration. Logistic Regression, Random Forest, and XGBoost are evaluated using their default configurations for RQ1 and optimized configurations for RQ2.

Hyperparameter optimization uses randomized search with 20 sampled configurations, stratified five-fold cross-validation, and ROC-AUC as the selection criterion. HPO is performed for training sizes of at least 32 observations; the two smallest training sizes remain baseline-only conditions.

All models are evaluated on the same held-out test partition for each dataset.

### 3.5 Evaluation metrics

The primary evaluation metrics are:

- **ROC-AUC:** Area under the receiver operating characteristic curve.
- **PR-AUC:** Precision–recall performance, computed as average precision in this implementation.

Secondary metrics include accuracy, balanced accuracy, precision, recall, and F1-score. Specificity and predicted-positive rate are also examined to characterize behaviour at the fixed classification threshold of 0.5.

Training-subset stability is assessed through the standard deviation of ROC-AUC across the five subset seeds at each training size and experimental condition.

## 4. Repository Structure

The repository is organized as follows:

```text
tabpfn3-vs-conventional-ml-low-data-health/
│
├── data/
│   └── raw/                     # Source datasets supplied by the user
│
├── results/
│   ├── eda/                     # Exploratory analysis outputs
│   └── ...                      # Experimental results and analysis outputs
│
├── 01_data_exploration.ipynb
├── 02_data_preparation.ipynb
├── 03_traditional_ml.ipynb
├── 04_tabpfn.ipynb
├── 05_traditional_ml_hpo.ipynb
├── 06_results_analysis.ipynb
│
├── data_pipeline.py             # Shared loading and preprocessing
├── experiment_protocol.py       # Central experimental configuration
├── eda_plots.py                 # Exploratory visualizations
├── requirements.txt             # Python dependencies
└── README.md
```

### Notebook responsibilities

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Data inspection and exploratory analysis |
| `02_data_preparation.ipynb` | Data cleaning, partitioning, and subset generation |
| `03_traditional_ml.ipynb` | Evaluation of default conventional models |
| `04_tabpfn.ipynb` | Evaluation of TabPFN-3 |
| `05_traditional_ml_hpo.ipynb` | Hyperparameter optimization and evaluation |
| `06_results_analysis.ipynb` | Result validation, comparative analysis, and figures |

The shared modules centralize functionality and configuration to reduce inconsistencies between experimental notebooks.

## 5. Installation and Setup

### 5.1 Requirements

The documented experimental environment uses Python 3.12 with the pinned dependencies in `requirements.txt`.

### 5.2 Clone the repository

```bash
git clone https://github.com/NicolasAbboud/tabpfn3-vs-conventional-ml-low-data-health.git

cd tabpfn3-vs-conventional-ml-low-data-health
```

### 5.3 Create a virtual environment

**Linux/macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 5.4 Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The dependency file specifies the principal packages and versions used for the experiments, including TabPFN, scikit-learn, XGBoost, NumPy, pandas, SciPy, and the notebook and visualization tools.

## 6. Dataset Setup

Download both datasets from the sources listed in Section 3.1.

Place the original CSV files at the paths expected by `data_pipeline.py`:

```text
data/
└── raw/
    ├── diabetes/
    │   └── diabetes.csv
    └── cardiovascular/
        └── cardiovascular.csv
```

The diabetes file is read as a comma-separated CSV. The cardiovascular file is read as a semicolon-separated CSV.

**Important:** Preserve the expected column names and schema. The data loader checks the supplied files against the declared dataset configuration and raises an error if columns are missing or unexpected.

The data-preparation notebook generates the cleaned data, training/test partitions, subset indices, and provenance metadata required by the subsequent experiments.

## 7. Running the Experiments

Execute the notebooks in the following order:

1. `01_data_exploration.ipynb`
2. `02_data_preparation.ipynb`
3. `03_traditional_ml.ipynb`
4. `04_tabpfn.ipynb`
5. `05_traditional_ml_hpo.ipynb`
6. `06_results_analysis.ipynb`

The preparation stage must complete before running the model-evaluation notebooks because the downstream experiments rely on the shared prepared data and experimental manifest.

The full experimental workflow involves 710 final test-evaluation runs:

| Experimental condition | Final evaluation runs |
|---|---:|
| Conventional models with default configurations | 330 |
| TabPFN-3 | 110 |
| Conventional models after HPO | 270 |
| **Total** | **710** |

## 8. Reproducibility and Experimental Controls

Reproducibility is supported through a centralized experimental protocol and automated data-integrity checks.

### Centralized configuration

`experiment_protocol.py` defines the training sizes, subset seeds, model-side seed, cross-validation seed, HPO seed, classification threshold, primary metrics, and HPO settings.

The protocol identifier is:

`low_data_tabpfn_v1`

### Deterministic subset generation

Subset random states are derived deterministically from the training size and subset seed using SHA-256. The resulting subset indices are recorded so that the same observations can be reused across experimental notebooks.

### Data provenance and validation

The pipeline records SHA-256 hashes of the raw input files and stores the experiment manifest and subset indices. Automated checks verify subset sizes, class balance, duplicate-row absence, membership in the training partition, separation from the test set, and feature dimensionality.

A leakage audit confirmed that no raw row identifier occurs in both the training and test partitions. Identical feature vectors can nevertheless occur across partitions, reflecting the structure of the source datasets; this overlap is documented rather than removed.

### Separation of random sources

The experimental design distinguishes four random sources:

- `subset_seed`: determines which observations are sampled.
- `model_seed`: controls stochasticity within estimators.
- `cv_seed`: controls cross-validation fold assignment.
- `hpo_seed`: controls hyperparameter sampling.

The model, cross-validation, and HPO seeds are held fixed across the relevant comparisons. Consequently, the stability analysis primarily measures the effect of changing training-subset composition.

### Result validation

The analysis notebook validates result-table schemas, run counts, keys, seeds, protocol identifiers, classification thresholds, and metric ranges before constructing the analysis tables and figures.

For strict reproducibility, users should retain the raw dataset versions, environment, source-code revision, experiment manifest, and resulting output files together.
