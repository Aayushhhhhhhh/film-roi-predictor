"""
evaluate.py — Full evaluation report for all trained models.
Usage: python src/evaluate.py
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

ALL_FEATURES = [
    'is_franchise', 'sequel_number', 'release_year',
    'production_budget', 'marketing_budget',
    'director_avg_roi', 'lead_actor_avg_gross',
    'trailer_views_24hr', 'social_buzz_score',
    'pre_release_tracking', 'rotten_tomatoes_score',
    'genre', 'mpaa_rating', 'release_season', 'studio'
]


def run_evaluation(data_path: str = "data/film_roi_dataset.csv",
                   artifact_path: str = "models/model_artifacts.pkl"):

    with open(artifact_path, 'rb') as f:
        arts = pickle.load(f)

    pipelines = arts['pipelines']
    fi        = arts['feature_importances']

    df = pd.read_csv(data_path)
    X  = df[ALL_FEATURES]
    y  = df['roi']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("=" * 58)
    print("  FILM ROI PREDICTION — EVALUATION REPORT")
    print("=" * 58)

    print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("─" * 50)
    for name, pipe in pipelines.items():
        preds = pipe.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        print(f"{name:<22} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")

    print("\nTop 10 Feature Importances (Gradient Boosting):")
    print("─" * 45)
    for feat, imp in fi.head(10).items():
        bar = '█' * int(imp * 120)
        print(f"  {feat:<30} {imp:.4f}  {bar}")

    print("\nSample predictions vs actuals — Gradient Boosting (8 rows):")
    print("─" * 50)
    gb_preds = pipelines['Gradient Boosting'].predict(X_test.head(8))
    actuals  = y_test.head(8).values
    print(f"  {'Actual':>8}  {'Predicted':>10}  {'Abs Error':>10}")
    for a, p in zip(actuals, gb_preds):
        print(f"  {a:>8.3f}  {p:>10.3f}  {abs(a-p):>10.3f}")
    print("=" * 58)


if __name__ == "__main__":
    run_evaluation()
