# Cardiometabolic Phenotype Discovery

An exploratory unsupervised machine-learning comparison using NHANES 2011–2018 data.

## Methods
- Eight biomarkers, 10,616 complete observations in the supplied run
- StandardScaler → PCA (2 components) → K-Means (k=3)
- StandardScaler → TensorFlow autoencoder (3 latent dimensions) → K-Means (k=3)

## Results from the supplied executed Colab run

| Method | Silhouette ↑ | Davies–Bouldin ↓ |
|---|---:|---:|
| PCA + K-Means | **0.3849** | **0.8982** |
| Autoencoder + K-Means | 0.3406 | 1.1052 |

PCA scored better on both internal clustering metrics in the supplied run. See the notebook for biomarker summaries, elbow/silhouette diagnostics, learning curves, and scatter plots. **These are exploratory biomarker clusters, not clinically validated diagnoses or risk predictions.**

## Run in Colab
1. Open `cardiometabolic_phenotype_discovery_clean.ipynb` in Google Colab.
2. Run the first cell and upload your NHANES project ZIP containing `DATASETS/Separate Datasets/` with the XPT files for cycles G–J.
3. Run all cells in order. Outputs are intentionally regenerated from data, not embedded as if the cleaned notebook had been executed.

## Limitations
Complete-case filtering may cause selection bias. NHANES survey weights and complex sampling design were not applied. Autoencoder metrics can vary between runs. Cluster IDs are arbitrary.

## Contributors
Add all project contributors and their respective contributions before publishing.
