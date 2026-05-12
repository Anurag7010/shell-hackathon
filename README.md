# Fuel Blend Property Prediction

### Shell.ai Hackathon — Multi-target regression on blended fuel properties

---

Fuel blending is essentially a mixing problem: take five components in varying proportions, combine them, and figure out what the resulting mixture looks like across ten different physical or chemical properties. The tricky part is that blending isn't always linear — properties interact in non-trivial ways, some follow logarithmic rules, others harmonic ones. This project builds a stacked ensemble that learns those patterns from data.

---

## The Problem

> Given the fraction of each of five components and ten properties per component, predict the ten blend properties of the resulting mixture.

- **Inputs:** 5 component fractions + 10 properties × 5 components = 55 raw features per sample
- **Outputs:** `BlendProperty1` through `BlendProperty10` (10 continuous targets)
- **Metric:** MAPE — Mean Absolute Percentage Error, averaged across all 10 targets
- **Competition:** [Shell.ai Hackathon — Fuel Blend Property Prediction](https://www.shell.ai/hackathon)

---

## Approach

### Feature Engineering

The raw 55 features are expanded significantly using physical intuition about how blending actually works:

**Mixing rules** — three different weighted aggregation schemes per property:

- _Linear mixing_ — weighted average by fraction (ideal mixing assumption)
- _Geometric mixing_ — log-space weighted average (used for viscosity-type properties)
- _Harmonic mixing_ — reciprocal-space aggregation (used for resistivity-type properties)

**Interaction features** — pairwise fraction products and property differences between every component pair. Captures non-linear synergy or antagonism between components.

**Statistical summaries** — mean, std, range, skew, kurtosis across components for each property, and across properties for each component. Adds context about spread and shape.

**Entropy features** — Shannon entropy of the fraction distribution captures how "mixed" a blend is. A blend dominated by one component behaves very differently from one with equal parts of all five.

**PCA** — 15 principal components extracted from the full 50-dimensional property matrix, fitted on training data only to prevent leakage onto test.

Total engineered features: ~400+

---

### Model Stack

A two-level stacking ensemble is used:

**Level 1 — Base learners (per target):**
| Model | Role |
|---|---|
| LightGBM | Fast gradient boosting, handles large feature sets well |
| XGBoost | Complementary boosting with different regularization |
| CatBoost | Robust to outliers, MAPE-native loss |
| Extra Trees | High variance, decorrelated from boosting methods |
| Random Forest | Stable ensemble baseline |

All base learners use early stopping during cross-validation folds (LGB/XGB/CatBoost) to find optimal tree counts rather than using a fixed number.

**Level 2 — Meta-learner:**
Ridge regression trained on out-of-fold predictions from all five base learners. Simple, fast, and resistant to overfitting at the meta level.

---

### Why OOF Stacking?

The old approach trained the meta-model on the same data the base models saw — this inflates meta-model performance because the base models have already "seen" those targets. Proper OOF stacking works like this:

```
For each fold k of 5:
    Train base models on folds 1..4
    Predict on fold k → store as OOF predictions

Train Ridge meta-model on the full OOF predictions
Retrain base models on all data
```

Every training row gets exactly one prediction from a model that never saw it. The meta-model then learns to combine base predictions with no information leakage.

---

### Post-processing

Predictions are clipped to `[p01, p99]` of the training label distribution per target. This prevents extreme extrapolation on test samples that fall outside the training support, which MAPE heavily penalizes.

---

## Quickstart

```bash
git clone <repo-url>
cd shell_hackathon

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Place train.csv and test.csv in data/
python shell_optimized.py
```

This will:

1. Split training data (85% train / 15% validation)
2. Train and evaluate, printing an accuracy report
3. Retrain on full data
4. Write `outputs/submission.csv` and `outputs/pipeline.pkl`

Optional flags:

```bash
python shell_optimized.py --val-split 0.2        # larger hold-out
python shell_optimized.py --tune                 # Optuna HPO (slow)
python shell_optimized.py --no-submission        # eval only, no file output
python shell_optimized.py --load-pipeline outputs/pipeline.pkl  # skip retraining
```

---

## Validation Results

> Fill these in after running the pipeline. OOF MAPE is computed during training via cross-validation. Val MAPE is computed on the 15% hold-out. Accuracy % = (1 − MAPE) × 100.

| Target          | OOF MAPE | Val MAPE | Val Accuracy |
| --------------- | -------- | -------- | ------------ |
| BlendProperty1  | TBD      | TBD      | TBD%         |
| BlendProperty2  | TBD      | TBD      | TBD%         |
| BlendProperty3  | TBD      | TBD      | TBD%         |
| BlendProperty4  | TBD      | TBD      | TBD%         |
| BlendProperty5  | TBD      | TBD      | TBD%         |
| BlendProperty6  | TBD      | TBD      | TBD%         |
| BlendProperty7  | TBD      | TBD      | TBD%         |
| BlendProperty8  | TBD      | TBD      | TBD%         |
| BlendProperty9  | TBD      | TBD      | TBD%         |
| BlendProperty10 | TBD      | TBD      | TBD%         |
| **Mean**        | **TBD**  | **TBD**  | **TBD%**     |

---

## Project Structure

```
shell_hackathon/
│
├── data/
│   ├── train.csv              ← provided by competition
│   ├── test.csv               ← provided by competition
│   └── sample_solution.csv    ← reference format
│
├── src/
│   ├── __init__.py
│   ├── features.py            ← FeatureEngineer class
│   ├── models.py              ← base models + TargetTrainer
│   ├── pipeline.py            ← ShellPipeline (load → fit → predict)
│   └── utils.py               ← logging, report printing
│
├── notebooks/
│   └── eda.ipynb              ← data exploration and mixing-rule analysis
│
├── tests/
│   └── test_features.py       ← pytest unit tests
│
├── outputs/                   ← gitignored; submissions and pickles land here
│
├── shell_optimized.py         ← entry point (imports from src/)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Design Decisions

**RobustScaler over StandardScaler**
Chemical property data often has outliers — components with extreme values in one property. RobustScaler uses median and IQR instead of mean and std, making it much more stable when a handful of rows have very large or very small property values.

**OOF stacking over naive stacking**
Naive stacking (training meta-model on training set predictions) is a data leakage issue — base models have already seen the target when generating those predictions, so the meta-model learns to trust them too much. OOF stacking ensures every prediction fed to the meta-model is genuinely out-of-sample.

**Ridge as meta-learner**
A simple linear combination of base model predictions works well here because the base models are already diverse and well-calibrated. Ridge adds L2 regularization to prevent the meta-model from over-weighting any single base learner when they happen to agree on the training folds.

**Per-target clipping**
MAPE is asymmetric — a large positive error on a small true value (e.g. predicting 2.0 when truth is 0.1) can be catastrophic. Clipping at training distribution percentiles prevents the model from making these explosive extrapolation errors on test samples outside the training distribution.

**Early stopping**
Rather than fixing tree counts at 1000 or 2000, early stopping monitors validation loss during each fold and stops when performance plateaus. This makes the ensemble less sensitive to the `n_estimators` hyperparameter and avoids overfitting in later boosting rounds.

---

## Dependencies

Core: `pandas`, `numpy`, `scipy`, `scikit-learn`
Boosting: `lightgbm`, `xgboost`, `catboost`
Optional: `optuna` (for hyperparameter tuning)
Utilities: `joblib`

See `requirements.txt` for pinned versions.

---

## Notes

- The final submission is always trained on **all** available training data — the 85/15 split is used only to estimate generalization before submitting.
- Running with `--tune` enables Optuna HPO per target — expect ~2–4× longer runtime but potentially better scores on difficult targets.
- Pickle files are large (~several hundred MB) due to storing all base model weights. They are gitignored.
