## Coffee Tasting Analysis — Streamlit App

This repo analyzes a coffee tasting dataset (Kaggle CoffeeReview) with PCA, linear regression, and classification. The Streamlit app lets you filter coffees, view sensory charts, regression diagnostics, and classification results.

### What’s implemented
- **Streamlit app (`app_coffee.py`)**
  - Sensory scatter plots vs rating with regression lines and Cook’s D diagnostics.
  - Multivariate linear regression (rating ~ aroma, acidity, body, flavor, aftertaste, price if present) with coefficients, R²/MSE/RMSE, predicted vs actual, and residual plots.
  - Classification tab with KNN and Decision Tree (rating binned into Low/Med/High) showing precision/recall/F1 and confusion matrices.
- **PCA (`coffee_pca.py`)**: runs PCA on sensory features; writes loadings, variance, and scores to `data/coffee/`.
- **Scatter generator (`coffee_scatter.py`)**: standalone PNGs of each sensory vs rating with regression lines and a combined lines plot.
- **Multivariate regression script (`coffee_multi_reg.py`)**: saves metrics and plots (pred vs actual, residuals, coefficients) to `data/coffee/`.
- **KNN/Decision Tree script (`coffee_knn_tree.py`)**: trains 80/20 classifiers on binned ratings; writes metrics to `data/coffee/knn_tree_metrics.txt`.

### Running the app
1) Ensure scripts are executable once:
   ```bash
   chmod +x setup.sh run.sh px.sh meta.sh python.sh
   ```
2) Create/update the env (uses conda/mamba):
   ```bash
   ./setup.sh --restart   # or ./setup.sh if env already exists
   ```
3) Run Streamlit:
   ```bash
   ./run.sh
   ```
   Open the printed URL (default `http://localhost:8501`). If you see Matplotlib cache warnings, run with `MPLCONFIGDIR=$(pwd)/.mpl-cache ./run.sh`.

### CLI utilities
- PCA: `./python.sh coffee_pca.py`
- Multivariate regression: `./python.sh coffee_multi_reg.py`
- Scatter PNGs: `./python.sh coffee_scatter.py`
- KNN/Decision Tree: `./python.sh coffee_knn_tree.py`
- Parquet explore: `./px.sh data/coffee/coffee.parquet --info --head 5`

### Coffee PCA (taste attributes)
- Uses the coffee tasting dataset (`data/coffee/coffee.parquet`) with aroma, acidity, body, flavor, aftertaste, and rating (4+ related features required for PCA).
- Run `./python.sh coffee_pca.py` (or `python coffee_pca.py` inside the repo) to generate:
  - `data/coffee/coffee_pca_loadings.csv`
  - `data/coffee/coffee_pca_variance.csv`
  - `data/coffee/coffee_pca_scores.csv`
- Current explained variance ratio: PC1 0.614, PC2 0.123, PC3 0.107, PC4 0.088 (cumulative 0.932). Loadings show PC1 is a "global quality" axis with strong positive weights across all flavor metrics; PC2 is dominated by body.
- The script drops rows with missing flavor metrics before fitting and standardizes features so the PCA isn’t biased by scale differences.

### Data
- Primary file: `data/coffee/coffee.parquet` (15 columns including aroma, acidity, body, flavor, aftertaste, rating, price, notes, etc.). Generated from Kaggle CoffeeReview via `coffee_kaggle.py` (not shown here).

### Outputs of interest
- PCA: `data/coffee/coffee_pca_loadings.csv`, `coffee_pca_variance.csv`, `coffee_pca_scores.csv`
- Regression (standalone script): `data/coffee/multi_reg_metrics.txt`, `multi_reg_pred_vs_actual.png`, `multi_reg_residuals.png`, `multi_reg_coefficients.png`
- Classification: `data/coffee/knn_tree_metrics.txt`
- Scatter PNGs: `data/coffee/scatter_rating_vs_*.png`, `scatter_rating_vs_all_lines.png`

### Environment notes
- Uses local conda env at `./.venv`. To run Python directly: `./python.sh your_script.py`.
- Key dependencies: streamlit (1.29), pandas, pyarrow, scikit-learn, scipy, altair (4.2.2).

### License
MIT License (see `LICENSE`).
