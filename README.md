# Mushroom Classification with Feature Selection and Explainable ML

A supervised machine learning project that predicts mushroom class (edible vs. poisonous) from categorical characteristics, then investigates how much predictive performance is retained using only the five most informative original features.

> **Research demonstration only.** Do not use this model to decide whether a real mushroom is safe to eat.

## Project overview

- **Dataset:** [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom), with 22 categorical predictor columns and a binary `class` target.
- **Model:** `ExtraTreesClassifier` (200 trees, balanced class weights).
- **Preprocessing:** One-hot encoding for categorical variables; unknown `stalk-root` values treated as a distinct category.
- **Evaluation:** Stratified five-fold cross-validation using accuracy; no train/test split.
- **Feature selection:** Mutual information and SHAP (mean absolute contribution, aggregated to original features).

## Evaluation design

The script compares three settings:

| Experiment | Features | Evaluation |
| --- | --- | --- |
| Baseline | All original features | Stratified 5-fold CV |
| Mutual information | Top 5 original features selected on each training fold | Stratified 5-fold CV |
| SHAP | Top 5 original features selected on each training fold | Stratified 5-fold CV |

**Why fold-wise selection matters:** The test fold is excluded from feature selection, avoiding optimistic evaluation caused by using the full dataset to choose features. The script also prints full-data rankings for interpretation; those rankings are **not** used to evaluate the five-feature models.

## Run locally

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python mushroom_classification.py
```

The script searches for `mushrooms.csv` alongside the script or in the current working directory. If not found, it attempts to retrieve the UCI Mushroom dataset from its original URL. You can also pass a local CSV path:

```bash
python mushroom_classification.py path/to/mushrooms.csv
```

The CSV must have a `class` column and the expected categorical features.

## Results

The script prints per-fold accuracy, mean accuracy, and standard deviation for all three experiments, plus top-five feature rankings by mutual information and SHAP. **Numerical results are intentionally not claimed here because this repository package has not been executed against the dataset.** Run the script and copy the observed scores into this section before using the numbers on a resume.

| Experiment | Mean 5-fold accuracy |
| --- | --- |
| All features | Run script to populate |
| Top 5 by mutual information | Run script to populate |
| Top 5 by SHAP | Run script to populate |

## Repository files

- `mushroom_classification.py` — complete data loading, modeling, feature selection, and evaluation pipeline.
- `requirements.txt` — Python dependencies.
- `.gitignore` — local environment exclusions.

## Limitations

- Accuracy alone does not show how often poisonous mushrooms are incorrectly predicted as edible. For a safety-critical application, recall, confusion matrices, calibration, and external validation would be essential.
- The classic UCI dataset may not represent mushrooms encountered outside the dataset; high cross-validation accuracy does not establish real-world safety.
- SHAP importance is computed from a fitted tree model; importance scores describe model behavior rather than causal effects.

## Data attribution

Data: UCI Machine Learning Repository, *Mushroom* dataset. See the linked source for dataset details and attribution.
