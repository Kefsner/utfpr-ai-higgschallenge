# Higgs Boson Discovery with Machine Learning

A scientific Jupyter notebook reproducing a **Higgs boson discovery analysis** using the official **ATLAS Higgs Boson Machine Learning Challenge (HiggsML)** dataset.
Unlike standard ML benchmarks (e.g. MNIST or Iris), the goal is to maximize **discovery significance** — quantified by the **Approximate Median Significance (AMS)** — rather than accuracy or F1.

This notebook bridges **particle-physics intuition** and **statistical inference**, showing how modern gradient-boosting models can isolate a rare signal (H→τ⁺τ⁻) from an overwhelming background.

> **Format:** Stand-alone Jupyter Notebook (`notebook.ipynb`) with full code, derivations, methodology, and results.

---

## 🔬 Key Features

* **Dual metrics**

  * Training monitored by **weighted ROC AUC**.
  * Evaluation by **AMS** (discovery significance).
* **Physics-aware preprocessing**

  * Replace sentinel `-999.0` with `NaN`.
  * One-hot encode `PRI_jet_num` to represent event topology.
  * Keep NaNs for LightGBM/XGBoost (native handling).
* **Weighted validation**

  * Stratified K-Fold cross-validation with event weights.
* **Unified model interface**

  * Shared wrapper for LightGBM, XGBoost, and RandomForest.
* **Hyperparameter optimization**

  * Bayesian tuning with Optuna, maximizing mean CV-AMS.
* **Exact AMS optimization**

  * Vectorized threshold sweep for stable AMS maximization.
* **Kaggle-ready submission**

  * Generates `submission.csv` with final predictions.

---

## ⚙️ Environment Setup

```bash
python3 --version
python3 -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🧠 Methodology Summary

* **Training metric:** Weighted ROC AUC (threshold-independent ranking).
* **Discovery metric:**
  [
  \mathrm{AMS} = \sqrt{2\left[(s+b+b_\text{reg})\ln!\left(1+\frac{s}{b+b_\text{reg}}\right) - s\right]}, \quad b_\text{reg}=10
  ]
  where ( s ) and ( b ) are weighted signal and background yields above the optimal threshold.
* **Threshold optimization:** Exact sweep over all unique scores.
* **Cross-validation:** Weighted Stratified K-Fold (5 splits).
* **Tuning:** Optuna study maximizing mean AMS across folds.

---

## 📊 Results

| Model         | Mean ROC AUC | Mean AMS   |
| ------------- | ------------ | ---------- |
| LightGBM      | 0.9229       | 1.3950     |
| XGBoost       | **0.9321**   | **1.6022** |
| Random Forest | 0.9197       | 1.3751     |

**Best Model:**
→ XGBoost (Mean ROC AUC = 0.9321, Mean AMS = 1.6022)

**Optuna best trial:**
→ `Trial 28` with AMS = **1.69638**

**Kaggle Public Score:**
→ **3.59276**

## 📚 References

**[1]** Aad, Georges, et al. "Observation of a new particle in the search for the Standard Model Higgs boson with the ATLAS detector at the LHC." Physics Letters B 716.1 (2012): 1-29.

**[2]** Adam-Bourdarios, Claire, et al. "Learning to discover: the higgs boson machine learning challenge (2014)." URL http://higgsml.lal.in2p3.fr/documentation.

**[3]** Adam-Bourdarios, Claire, et al. "The Higgs boson machine learning challenge." NIPS 2014 workshop on high-energy physics and machine learning. PMLR, 2015.