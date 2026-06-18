"""
train.py — Train all three models and save artifacts.
Usage: python src/train.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

NUMERIC_FEATURES     = [
    'is_franchise', 'sequel_number', 'release_year',
    'production_budget', 'marketing_budget',
    'director_avg_roi', 'lead_actor_avg_gross',
    'trailer_views_24hr', 'social_buzz_score',
    'pre_release_tracking', 'rotten_tomatoes_score'
]
CATEGORICAL_FEATURES = ['genre', 'mpaa_rating', 'release_season', 'studio']
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), CATEGORICAL_FEATURES),
    ])


def train(data_path: str = "data/film_roi_dataset.csv", output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    X  = df[ALL_FEATURES]
    y  = df['roi']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipelines = {
        'Linear Regression': Pipeline([
            ('pre', build_preprocessor()),
            ('reg', LinearRegression())
        ]),
        'Random Forest': Pipeline([
            ('pre', build_preprocessor()),
            ('reg', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('pre', build_preprocessor()),
            ('reg', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))
        ]),
    }

    results = {}
    print(f"\n{'Model':<22} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("─" * 50)

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = {
            'MAE':  round(mean_absolute_error(y_test, preds), 4),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            'R2':   round(r2_score(y_test, preds), 4),
        }
        results[name] = metrics
        print(f"{name:<22} {metrics['MAE']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['R2']:>8.4f}")

    # Feature importances from Gradient Boosting
    gb_pipe  = pipelines['Gradient Boosting']
    ohe_cols = gb_pipe.named_steps['pre'].named_transformers_['cat']\
                      .get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    fi = pd.Series(
        gb_pipe.named_steps['reg'].feature_importances_,
        index=NUMERIC_FEATURES + ohe_cols
    ).sort_values(ascending=False)

    print("\nTop 10 Feature Importances (Gradient Boosting):")
    print(fi.head(10).to_string())

    # Save artifacts
    artifacts = {
        'pipelines':           pipelines,
        'results':             results,
        'feature_importances': fi,
        'feature_cols':        ALL_FEATURES,
    }
    artifact_path = os.path.join(output_dir, 'model_artifacts.pkl')
    with open(artifact_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\n✅ Artifacts saved → {artifact_path}")

    fi.reset_index().rename(columns={'index': 'feature', 0: 'importance'})\
      .to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)

    return artifacts


if __name__ == "__main__":
    train()
