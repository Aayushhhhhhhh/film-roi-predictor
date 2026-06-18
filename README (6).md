"""
app.py  —  Film ROI Prediction Engine
Streamlit app using a full sklearn Pipeline with OneHotEncoder
and 15 features for maximum predictive accuracy.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ─────────────────────────────────────────────
# 1. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Film ROI Predictor", layout="wide", page_icon="🎬")
st.title("🎬 Film ROI Prediction Engine")
st.write("Powered by Gradient Boosting · 15 features · sklearn Pipeline")

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
NUMERIC_FEATURES = [
    'is_franchise', 'sequel_number', 'release_year',
    'production_budget', 'marketing_budget',
    'director_avg_roi', 'lead_actor_avg_gross',
    'trailer_views_24hr', 'social_buzz_score',
    'pre_release_tracking', 'rotten_tomatoes_score'
]
CATEGORICAL_FEATURES = ['genre', 'mpaa_rating', 'release_season', 'studio']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

GENRES   = ['Action', 'Animation', 'Comedy', 'Drama', 'Horror', 'Romance', 'Sci-Fi', 'Thriller']
RATINGS  = ['G', 'PG', 'PG-13', 'R']
SEASONS  = ['Summer', 'Holiday', 'Spring', 'Fall']
STUDIOS  = ['Universal', 'Warner Bros', 'Disney', 'Sony', 'Paramount', 'Lionsgate', 'A24', 'Netflix']

# ─────────────────────────────────────────────
# 3. TRAIN ALL MODELS (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_all_models():
    df = pd.read_csv('data/film_roi_dataset.csv')
    X = df[ALL_FEATURES]
    y = df['roi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), CATEGORICAL_FEATURES),
    ])

    pipelines = {
        'Linear Regression':  Pipeline([('pre', preprocessor), ('reg', LinearRegression())]),
        'Random Forest':      Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))]),
        'Gradient Boosting':  Pipeline([('pre', preprocessor), ('reg', GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))]),
    }

    results = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            'pipeline': pipe,
            'MAE':  round(mean_absolute_error(y_test, preds), 3),
            'RMSE': round(np.sqrt(mean_squared_error(y_test, preds)), 3),
            'R2':   round(r2_score(y_test, preds), 3),
        }

    # Feature importances from GB
    gb_pipe = pipelines['Gradient Boosting']
    ohe_cols = gb_pipe.named_steps['pre'].named_transformers_['cat'].get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feat_names = NUMERIC_FEATURES + ohe_cols
    fi = pd.Series(
        gb_pipe.named_steps['reg'].feature_importances_,
        index=all_feat_names
    ).sort_values(ascending=False)

    return pipelines, results, fi, X_test, y_test


pipelines, results, feature_importances, X_test, y_test = train_all_models()

# ─────────────────────────────────────────────
# 4. SIDEBAR — INPUT PARAMETERS
# ─────────────────────────────────────────────
st.sidebar.header("🎛️ Configure Film Parameters")

genre         = st.sidebar.selectbox("Genre", GENRES)
mpaa_rating   = st.sidebar.selectbox("MPAA Rating", RATINGS, index=2)
release_season = st.sidebar.selectbox("Release Season", SEASONS)
studio        = st.sidebar.selectbox("Studio", STUDIOS)

st.sidebar.markdown("---")

is_franchise  = st.sidebar.checkbox("Part of a Franchise / Sequel?", value=False)
sequel_number = st.sidebar.slider("Sequel Number (0 = original)", 0, 6, 0) if is_franchise else 0
release_year  = st.sidebar.slider("Release Year", 2000, 2030, 2024)

st.sidebar.markdown("---")

budget        = st.sidebar.slider("Production Budget ($M)", 1.0, 300.0, 80.0) * 1_000_000
mkt_budget    = st.sidebar.slider("Marketing Budget ($M)", 1.0, 200.0, 40.0) * 1_000_000

st.sidebar.markdown("---")

director_roi       = st.sidebar.slider("Director's Avg Historical ROI", 0.1, 10.0, 2.0)
lead_actor_gross   = st.sidebar.slider("Lead Actor Avg Gross ($M)", 10.0, 400.0, 150.0) * 1_000_000
trailer_views      = st.sidebar.slider("Trailer Views (24hr)", 500_000, 30_000_000, 8_000_000, step=500_000)
social_buzz        = st.sidebar.slider("Social Buzz Score (0–100)", 0, 100, 70)
tracking           = st.sidebar.slider("Pre-release Tracking Score (0–100)", 0, 100, 70)
rt_score           = st.sidebar.slider("Rotten Tomatoes Score (0–100)", 0, 100, 75)

model_choice = st.sidebar.selectbox("Model", list(pipelines.keys()), index=2)

predict_btn = st.sidebar.button("🚀 Predict ROI", use_container_width=True)

# ─────────────────────────────────────────────
# 5. MAIN AREA — TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🏆 Model Comparison", "🔍 Feature Importance"])

# ── TAB 1: PREDICTION ──
with tab1:
    if predict_btn:
        input_df = pd.DataFrame([{
            'genre':                genre,
            'mpaa_rating':          mpaa_rating,
            'release_season':       release_season,
            'studio':               studio,
            'is_franchise':         int(is_franchise),
            'sequel_number':        sequel_number,
            'release_year':         release_year,
            'production_budget':    budget,
            'marketing_budget':     mkt_budget,
            'director_avg_roi':     director_roi,
            'lead_actor_avg_gross': lead_actor_gross,
            'trailer_views_24hr':   trailer_views,
            'social_buzz_score':    social_buzz,
            'pre_release_tracking': tracking,
            'rotten_tomatoes_score': rt_score,
        }])

        roi = pipelines[model_choice].predict(input_df)[0]
        total_cost    = budget + mkt_budget
        total_revenue = total_cost * (1 + roi)
        net_profit    = total_revenue - total_cost

        st.subheader("📈 Forecasted Financial Performance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI Multiplier",       f"{roi:.2f}×")
        c2.metric("Total Cost",           f"${total_cost/1e6:,.1f}M")
        c3.metric("Projected Revenue",    f"${total_revenue/1e6:,.1f}M")
        c4.metric("Estimated Net Profit", f"${net_profit/1e6:,.1f}M",
                  delta=f"{'▲' if net_profit > 0 else '▼'} {'Profit' if net_profit > 0 else 'Loss'}")

        if roi > 4:
            st.success("🟢 VERDICT: Blockbuster — Exceptional return expected")
        elif roi > 2:
            st.success("🟢 VERDICT: Strong Commercial Hit")
        elif roi > 1:
            st.info("🟡 VERDICT: Profitable / Break-Even Zone")
        else:
            st.error("🔴 VERDICT: Projected Financial Loss")

        # Revenue breakdown bar chart
        st.markdown("#### Revenue Breakdown")
        fig, ax = plt.subplots(figsize=(6, 2.5))
        categories = ['Production\nBudget', 'Marketing\nBudget', 'Projected\nRevenue']
        values     = [budget/1e6, mkt_budget/1e6, total_revenue/1e6]
        colors     = ['#E87C7C', '#F5C26B', '#6DBF8A']
        bars = ax.barh(categories, values, color=colors, height=0.4)
        ax.set_xlabel("USD (Millions)")
        ax.bar_label(bars, fmt='$%.1fM', padding=4, fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Configure film parameters in the sidebar and click **Predict ROI** to get started.")

# ── TAB 2: MODEL COMPARISON ──
with tab2:
    st.subheader("Model Performance on Hold-out Test Set (20%)")
    rows = []
    for name, r in results.items():
        rows.append({'Model': name, 'MAE ↓': r['MAE'], 'RMSE ↓': r['RMSE'], 'R² ↑': r['R2'],
                     'Best?': '✅ Best' if name == 'Gradient Boosting' else ''})
    st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    names = list(results.keys())
    colors = ['#85B7EB', '#85B7EB', '#378ADD']
    for i, metric in enumerate(['MAE', 'RMSE', 'R2']):
        vals = [results[n][metric] for n in names]
        axes[i].bar(names, vals, color=colors)
        axes[i].set_title(metric)
        axes[i].tick_params(axis='x', rotation=15, labelsize=8)
        axes[i].spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── TAB 3: FEATURE IMPORTANCE ──
with tab3:
    st.subheader("Feature Importances — Gradient Boosting")
    top_fi = feature_importances.head(12)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_fi.sort_values().plot.barh(ax=ax, color='#378ADD')
    ax.set_xlabel("Importance Score")
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.dataframe(
        feature_importances.reset_index().rename(columns={'index': 'Feature', 0: 'Importance'})
        .head(15).style.format({'Importance': '{:.4f}'}),
        use_container_width=True
    )
