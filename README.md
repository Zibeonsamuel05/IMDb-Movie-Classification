# IMDb Movie Rating Classification (2003–2022)

A machine-learning classification project using the Kaggle dataset **Top 100 Popular Movies from 2003 to 2022 (IMDb)**.

## Domain
Entertainment and Media Analytics — specifically film/movie analytics.

## Project objective
Explore IMDb movie metadata and compare machine-learning classifiers for predicting **IMDb rating categories** from available movie attributes.

> Important terminology: the target labels in this project are derived from the `Rating` column. They represent rating bands, not a direct measure of movie popularity.

## Dataset
- Kaggle dataset: [Top 100 Popular Movies from 2003 to 2022 (IMDb)](https://www.kaggle.com/datasets/georgescutelnicu/top-100-popular-movies-from-2003-to-2022-imdb)
- Expected CSV filename: `movies.csv`
- Expected columns: `Title`, `Rating`, `Year`, `Month`, `Certificate`, `Runtime`, `Directors`, `Stars`, `Genre`, `Filming_location`, `Budget`, `Income`, `Country_of_origin`.

The dataset is not bundled here. Download it from Kaggle and place `movies.csv` in the `data/` folder. Check the dataset's licence and terms before redistributing it.

## Target definitions
### Multiclass
- Poor: IMDb rating < 5
- Average: 5 to < 7
- Good: 7 to < 8.5
- Excellent: 8.5 to 10

### Binary
- Low-rated: IMDb rating < 7
- High-rated: IMDb rating >= 7

## Features and preprocessing
The notebook excludes `Rating` (the source of the target), `Rating_Category` (derived target), and identifier/free-text columns `Title`, `Directors`, `Stars`, and `Filming_location` from the predictors. It uses `Year`, `Month`, `Certificate`, `Runtime`, `Genre`, `Budget`, `Income`, and `Country_of_origin`.

Numeric columns are median-imputed and standardised. Categorical columns are mode-imputed and one-hot encoded. The train/test split is 80/20 with `random_state=42` and stratification.

## Models
- Multiclass: Logistic Regression, Decision Tree, Random Forest
- Binary: K-Nearest Neighbours (KNN), Support Vector Machine (SVM)

## Previously reported evaluation results
These are the results supplied for the project comparison. Re-run the notebook against the original dataset to verify/reproduce them; results can differ if the dataset version, preprocessing, package versions, or data cleaning differ.

| Model | Task | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Multiclass | 0.550 | 0.615 | 0.550 | 0.574 |
| Decision Tree | Multiclass | 0.560 | 0.598 | 0.560 | 0.574 |
| Random Forest | Multiclass | 0.658 | 0.628 | 0.658 | 0.637 |
| KNN | Binary | 0.645 | 0.636 | 0.645 | 0.636 |
| SVM | Binary | 0.740 | 0.738 | 0.740 | 0.739 |

Best reported multiclass model: **Random Forest** (65.8% accuracy; 0.637 weighted F1).

Best reported binary model: **SVM** (74.0% accuracy; 0.739 weighted F1).

Binary and multiclass scores are from different prediction tasks and should not be treated as a direct head-to-head comparison.

## Run the project
1. Install Python 3.10 or newer.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the CSV from Kaggle and save it as `data/movies.csv`.
4. Open `notebooks/movie_rating_classification.ipynb` in Jupyter Notebook or JupyterLab.
5. Run all cells from top to bottom.

Alternatively, run `python src/train_models.py` from the repository root.

## Repository structure
```text
IMDb-Movie-Classification/
├── data/                         # Put the downloaded CSV here (not committed)
├── notebooks/
│   └── movie_rating_classification.ipynb
├── reports/
│   └── model_comparison_report.md
├── results/
│   ├── model_comparison.csv
│   └── model_comparison.png
├── src/
│   └── train_models.py
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Limitations
- These models learn patterns in the supplied dataset; they do not establish causal relationships.
- The target is derived from IMDb ratings, so it should be described as rating classification rather than popularity prediction.
- The reported metrics should be revalidated with the original data and the exact final notebook.
