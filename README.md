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
