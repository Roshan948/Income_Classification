# Adult Census Income Classifier

## Overview

Binary classification project on the Adult Census Income dataset: given a
person's demographic, educational, and employment attributes, predict whether
their annual income is above $50,000 or at/below it. The dataset is
imbalanced (roughly 3-to-1 toward the lower bracket), so F1 is used as the
primary comparison metric rather than raw accuracy throughout the notebook.

## Approach

- Missing categorical values (encoded in the raw data as `"?"`) are kept as
  an explicit `"Unknown"` category instead of being dropped, since a missing
  workclass or occupation can itself be informative.
- `education` is collapsed from 16 raw levels into six ordered attainment
  bands (`No-HS` through `Grad-Degree`) and ordinal-encoded, since more
  schooling is a meaningful direction rather than an unordered label.
- All other categorical fields are one-hot encoded with unseen categories
  tolerated at inference time, and the four numeric fields are scaled with a
  median/IQR-based scaler to stay robust to the heavy skew in capital
  gains/losses.
- All of the above is expressed as a single `scikit-learn` `ColumnTransformer`
  inside a `Pipeline`, rather than a set of separate encoder/scaler objects —
  one fitted object captures the entire preprocessing step.
- Four classical classifiers (logistic regression, k-NN, random forest,
  histogram-based gradient boosting) are tuned with grid search under
  5-fold stratified cross-validation, plus a small PyTorch MLP tuned with
  Optuna as an additional comparison point.
- The strongest model on held-out F1 is refit on top of the already-fitted
  preprocessor and the resulting end-to-end pipeline is the one artifact
  saved to disk.

## Files

| File | Purpose |
|---|---|
| `income_classification.ipynb` | Full EDA → preprocessing → model comparison → save-out workflow |
| `inference.py` | Loads the saved pipeline and exposes `predict(raw_dict)` |
| `app_gui.py` | Small Tkinter form for entering a record and seeing the prediction |
| `artifacts/pipeline.joblib` | Fitted preprocessing + classifier pipeline (created by the notebook) |
| `artifacts/metadata.joblib` | Education grouping/order and expected feature columns |

