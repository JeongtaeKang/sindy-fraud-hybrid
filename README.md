# SINDy-Augmented Hybrid Fraud Detection

This repository contains code for a hybrid fraud detection framework that augments static features with SINDy-derived dynamic indicators and trains LightGBM/XGBoost baselines and hybrids. It also includes SHAP/PDP/LIME explainability artifacts (code).

## How to run
- Python 3.11 (recommended)
- `pip install -r requirements.txt`
- Put the PaySim raw CSV **outside** the repo or download from the original source.
- Run `python new_hybrid_model_11.py`

## Data Availability
The PaySim dataset is publicly available from the original repository. Raw data are **not** included in this repo to avoid size/licensing issues. See the paper’s Data Availability Statement.

## Citation
If you use this code, please cite the associated manuscript (Electronics, MDPI).

