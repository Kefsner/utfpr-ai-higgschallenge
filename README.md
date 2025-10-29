# Higgs Boson Discovery with Machine Learning

A scientific Jupyter notebook reproducing a **Higgs boson discovery analysis** using the official **ATLAS Higgs Boson Machine Learning Challenge (HiggsML)** dataset.
Unlike typical ML benchmarks (e.g. MNIST or Iris), the objective here is to maximize **discovery significance** — quantified by the **Approximate Median Significance (AMS)** — rather than accuracy or F1.

This notebook bridges **particle-physics intuition** and **statistical inference**, demonstrating how modern gradient-boosting models can isolate a rare signal (H→τ⁺τ⁻) from an overwhelming background.

> **Format:** Stand-alone Jupyter Notebook (`notebook.ipynb`) with code, derivations, methodology, and full results.

---

## 🔬 Key Features

* **Dual metrics:**

  * Training monitored by **weighted ROC AUC** (ranking quality).
  * Evaluation by **AMS** (discovery significance).
* **Physics-aware preprocessing:**

  * Replace detector sentinel values `-999.0` with `NaN`.
  * One-hot encode `PRI_jet_num` to expose event topology.
  * Keep NaNs for **LightGBM/XGBoost** (native handling).
  * Use **masked imputation** for **RandomForest** (sentinel + missing indicator).
* **Weighted validation:**
  Stratified K-Fold cross-validation with proper per-event weights (`Weight` column).
* **Unified model interface:**
  Common wrapper for **LightGBM**, **XGBoost**, and **RandomForest**.
* **Hyperparameter optimization with Optuna:**
  End-to-end tuning that maximizes **mean CV-AMS**, resumable via `optuna_runs.db`.
* **Exact AMS optimization:**
  Finds the threshold that maximizes AMS through a **vectorized sweep** over all unique model scores.
* **Kaggle-ready submission:**
  Produces `submission.csv` with `RankOrder` and `Class` fields.

---

## ⚙️ Stack & Dependencies

| Category                    | Library / Tool                            |
| --------------------------- | ----------------------------------------- |
| Language & Notebook         | Python 3.12+, Jupyter                     |
| Data Science                | `numpy`, `pandas`, `matplotlib`, `plotly` |
| Machine Learning            | `scikit-learn`                            |
| Gradient Boosting           | `lightgbm`, `xgboost`                     |
| Hyperparameter Optimization | `optuna`                                  |

> Full list of versions in **`requirements.txt`**.

---

## 📁 Repository Structure

```
.
├── Avaliação-IA.pdf                            # Assignment brief / evaluation criteria
├── Higgs challenge.pdf                         # Official challenge documentation (AMS definition, weights)
├── The Higgs boson machine learning challenge.pdf  # Reference paper (physics + statistical context)
├── notebook.ipynb                              # Main scientific notebook
├── optuna_runs.db                              # Optuna study database (resumable)
├── requirements.txt                            # Python dependencies
├── submission.csv                              # Kaggle-style output (generated)
└── README.md                                   # You are here
```

> **Dataset:** place `training.csv` and `test.csv` inside a `./data/` directory:
> `data/training.csv`, `data/test.csv`

---

## 🚀 Setup & Execution

### 1) Create and activate a virtual environment

```bash
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2) Launch Jupyter

```bash
jupyter lab     # or: jupyter notebook
```

Open **`notebook.ipynb`** and execute all cells in order.

The notebook will:

1. Load and preprocess the dataset (`data/training.csv`, `data/test.csv`)
2. Train baseline models (LightGBM, XGBoost, RandomForest) using **Stratified K-Fold**
3. Compute **weighted ROC AUC** and **AMS** per fold
4. Run **Optuna** for end-to-end hyperparameter tuning (stored in `optuna_runs.db`)
5. Retrain the best model on the full dataset
6. Generate **`submission.csv`** for Kaggle-style scoring

---

## 🧠 Methodology Summary

* **Training metric:** Weighted ROC AUC — invariant to thresholds, robust to class imbalance.
* **Discovery metric:** AMS — the approximate Gaussian discovery significance:

  [
  \mathrm{AMS} = \sqrt{2\left[(s+b+b_\text{reg})\ln!\left(1+\frac{s}{b+b_\text{reg}}\right) - s\right]}, \quad b_\text{reg}=10
  ]

  where ( s ) and ( b ) are the weighted signal and background yields after applying a threshold on model scores.
* **Threshold optimization:** exact, monotonic sweep across all unique scores.
* **Cross-validation:** Stratified K-Fold with weights propagated to both training and metrics.

---

## 📊 Results (Spoiler)

After full cross-validation and hyperparameter tuning:

| Model        | Mean ROC AUC | Mean AMS | Best Threshold |
| ------------ | ------------ | -------- | -------------- |
| LightGBM     | ~0.88        | ~3.2     | ~0.53          |
| XGBoost      | **~0.89**    | **~3.5** | ~0.51          |
| RandomForest | ~0.83        | ~2.8     | ~0.47          |

* The **best model (XGBoost)** achieved an **AMS ≈ 3.5**, corresponding to about **3.5 σ** statistical significance.
* In real HEP terms, **5 σ** is the canonical discovery threshold — thus, our result constitutes **strong evidence** of a Higgs-like signal.
* The AMS curve shows a clear plateau, confirming **stable generalization** across folds.

---

## 🧩 Practical Notes

* **NaN handling:** LightGBM/XGBoost natively split on missing values; RandomForest requires explicit imputation.
* **Reproducibility:** The `optuna_runs.db` file enables resuming and comparing studies.
* **Runtime:** Optuna tuning (~20 trials, 5-fold CV) runs in under 1 hour on a modern CPU.
* **Customization:** Increase `n_trials` in `study.optimize()` for deeper searches.

---

## 📚 References

1. ATLAS Collaboration, *Higgs Boson Machine Learning Challenge (HiggsML)* — official documentation.
2. Adam-Bourdarios, C. *et al.*, “The Higgs Boson Machine Learning Challenge,” *J. Phys.: Conf. Ser.* 664 (2015).
3. Cowan, G. *et al.*, “Asymptotic formulae for likelihood-based tests of new physics,” *Eur. Phys. J. C* 71 (2011).
4. Bhat, P., *Statistical Methods in Particle Physics* — overview of likelihood-ratio tests and significance.
5. LightGBM, XGBoost, and Optuna official documentation.
