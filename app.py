import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Film ROI Predictor", layout="wide", page_icon="🎬")

# ── Load & Train (cached so it only runs once) ───────────────────────────────
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data/film_roi_dataset.csv")

    features = [
        "genre", "mpaa_rating", "release_season",
        "production_budget", "marketing_budget",
        "is_franchise", "trailer_views_24hr",
        "pre_release_tracking", "rotten_tomatoes_score",
        "director_avg_roi"
    ]
    target = "roi"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    num_cols = [
        "production_budget", "marketing_budget", "is_franchise",
        "trailer_views_24hr", "pre_release_tracking",
        "rotten_tomatoes_score", "director_avg_roi"
    ]
    cat_cols = ["genre", "mpaa_rating", "release_season"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "mae":  round(mean_absolute_error(y_test, y_pred), 3),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 3),
        "r2":   round(r2_score(y_test, y_pred), 3),
    }

    # Feature importances — map back from OHE to original names
    ohe_cols = (
        model.named_steps["preprocessor"]
             .named_transformers_["cat"]
             .get_feature_names_out(cat_cols)
             .tolist()
    )
    all_feat_names = num_cols + ohe_cols
    raw_fi = pd.Series(
        model.named_steps["regressor"].feature_importances_,
        index=all_feat_names
    )
    # Collapse OHE columns back to original feature name
    fi = {}
    for col in num_cols:
        fi[col] = raw_fi[col]
    for col in cat_cols:
        fi[col] = raw_fi[[c for c in raw_fi.index if c.startswith(col + "_")]].sum()
    fi = pd.Series(fi).sort_values(ascending=False)

    return model, metrics, fi, X_test, y_test, y_pred, features, num_cols, cat_cols


model, metrics, fi, X_test, y_test, y_pred, features, num_cols, cat_cols = load_and_train()

# ── Sidebar — Prediction Inputs ───────────────────────────────────────────────
st.sidebar.header("🎛️ Enter Film Details")

genre          = st.sidebar.selectbox("Genre", ["Action", "Animation", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"])
mpaa_rating    = st.sidebar.selectbox("MPAA Rating", ["G", "PG", "PG-13", "R"], index=2)
release_season = st.sidebar.selectbox("Release Season", ["Summer", "Holiday", "Spring", "Fall"])
is_franchise   = st.sidebar.selectbox("Franchise / Sequel?", ["No", "Yes"])
budget         = st.sidebar.slider("Production Budget ($M)", 1, 300, 80)
mkt_budget     = st.sidebar.slider("Marketing Budget ($M)", 1, 150, 40)
trailer_views  = st.sidebar.slider("Trailer Views in 24hr (M)", 0.5, 30.0, 8.0, step=0.5)
tracking       = st.sidebar.slider("Pre-release Tracking Score (0–100)", 0, 100, 70)
rt_score       = st.sidebar.slider("Rotten Tomatoes Score (0–100)", 0, 100, 75)
director_roi   = st.sidebar.slider("Director's Avg Historical ROI", 0.1, 10.0, 2.0, step=0.1)

predict_clicked = st.sidebar.button("🚀 Predict ROI", use_container_width=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯  Predict ROI",
    "📊  What the Data Tells Us",
    "✅  How Accurate Is Our Model"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("🎬 Film ROI Predictor")
    st.write("Fill in the film details on the left sidebar and click **Predict ROI**.")

    if predict_clicked:
        input_df = pd.DataFrame([{
            "genre":                genre,
            "mpaa_rating":          mpaa_rating,
            "release_season":       release_season,
            "production_budget":    budget * 1_000_000,
            "marketing_budget":     mkt_budget * 1_000_000,
            "is_franchise":         1 if is_franchise == "Yes" else 0,
            "trailer_views_24hr":   int(trailer_views * 1_000_000),
            "pre_release_tracking": tracking,
            "rotten_tomatoes_score": rt_score,
            "director_avg_roi":     director_roi,
        }])

        roi = model.predict(input_df)[0]
        total_cost    = (budget + mkt_budget) * 1_000_000
        total_revenue = total_cost * (1 + roi)
        net            = total_revenue - total_cost

        # Verdict
        if roi > 4:
            st.success("🟢 BLOCKBUSTER — Exceptional return expected")
        elif roi > 2:
            st.success("🟢 STRONG HIT — Profitable film")
        elif roi > 1:
            st.info("🟡 BREAK-EVEN — Marginal profit")
        else:
            st.error("🔴 LOSS — Projected to lose money")

        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROI Multiplier",    f"{roi:.2f}×")
        c2.metric("Total Cost",        f"${(total_cost/1e6):,.1f}M")
        c3.metric("Projected Revenue", f"${(total_revenue/1e6):,.1f}M")
        c4.metric("Net Profit / Loss", f"${(net/1e6):,.1f}M",
                  delta="Profit" if net > 0 else "Loss")

        # Simple bar chart
        st.markdown("#### Cost vs Revenue")
        fig, ax = plt.subplots(figsize=(5, 2.2))
        labels = ["Production", "Marketing", "Projected Revenue"]
        values = [budget, mkt_budget, total_revenue / 1e6]
        colors = ["#E07B7B", "#F0B858", "#5BAD82"]
        bars   = ax.barh(labels, values, color=colors, height=0.45)
        ax.bar_label(bars, fmt="$%.0fM", padding=4, fontsize=9)
        ax.set_xlabel("USD (Millions)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    else:
        st.info("👈 Set your film parameters in the sidebar and click **Predict ROI**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WHAT THE DATA TELLS US
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("📊 What the Data Tells Us")
    st.write("Here are the most important features our model found, what they mean, and why they matter.")

    # Feature importance chart
    st.subheader("Most Important Features for Predicting ROI")
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#378ADD" if i < 3 else "#B4C8E8" for i in range(len(fi))]
    fi.sort_values().plot.barh(ax=ax, color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance (Gradient Boosting)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    # Feature explanations
    st.subheader("What Each Feature Means")

    feature_explanations = {
        "genre": {
            "importance": fi.get("genre", 0),
            "what": "The category of the film — Action, Horror, Comedy, etc.",
            "why": "Genre is the #1 predictor. Horror films have massive ROI on tiny budgets (e.g. Get Out: $4.5M budget → $255M gross). Action films need huge budgets but also earn huge globally.",
            "tip": "Horror and Animation tend to have the highest ROI multipliers."
        },
        "production_budget": {
            "importance": fi.get("production_budget", 0),
            "what": "Total money spent making the film (excluding marketing).",
            "why": "Bigger budgets don't always mean bigger ROI. A $5M horror film can return 10× while a $200M blockbuster might return only 2×.",
            "tip": "Lower-budget films in high-ROI genres (horror, comedy) often outperform expensive ones."
        },
        "trailer_views_24hr": {
            "importance": fi.get("trailer_views_24hr", 0),
            "what": "Number of times the trailer was watched on YouTube in the first 24 hours.",
            "why": "It's the single best early signal of audience excitement before a film releases. High trailer views = high opening weekend.",
            "tip": "10M+ views in 24 hours is a strong indicator of a commercial hit."
        },
        "rotten_tomatoes_score": {
            "importance": fi.get("rotten_tomatoes_score", 0),
            "what": "Percentage of critics who gave the film a positive review (0–100).",
            "why": "Critics influence casual audiences. A high RT score drives word-of-mouth beyond opening weekend, increasing the film's total run.",
            "tip": "Films above 80% on RT tend to have strong legs at the box office."
        },
        "pre_release_tracking": {
            "importance": fi.get("pre_release_tracking", 0),
            "what": "A score (0–100) measuring how much general audiences are aware of and want to see the film before release.",
            "why": "This is what studios track closely. High tracking = high opening weekend. It's measured via surveys weeks before release.",
            "tip": "A tracking score above 75 usually signals a top-5 opening weekend."
        },
        "marketing_budget": {
            "importance": fi.get("marketing_budget", 0),
            "what": "Money spent on ads, trailers, promotions, and distribution (P&A costs).",
            "why": "Marketing drives awareness. But overspending on marketing for a bad film still produces a loss.",
            "tip": "Healthy ratio: marketing budget should be ~40–60% of production budget."
        },
        "director_avg_roi": {
            "importance": fi.get("director_avg_roi", 0),
            "what": "The director's average ROI across their previous films.",
            "why": "A director with a track record of profitable films is a strong signal. Studios bet on proven talent.",
            "tip": "Directors with avg ROI > 2.5× are considered commercially reliable."
        },
        "release_season": {
            "importance": fi.get("release_season", 0),
            "what": "Which time of year the film is released — Summer, Holiday, Spring, or Fall.",
            "why": "Summer (May–Aug) and Holiday (Nov–Dec) seasons have the highest footfall. Films earn 20–30% more in these windows.",
            "tip": "Releasing a big-budget film in Fall is risky — save Summer for tentpoles."
        },
        "mpaa_rating": {
            "importance": fi.get("mpaa_rating", 0),
            "what": "The age rating of the film — G, PG, PG-13, or R.",
            "why": "PG-13 films reach the widest audience (kids + adults). R-rated films limit their audience but can still succeed with the right genre (horror, thriller).",
            "tip": "PG-13 is the sweet spot for maximum box office potential."
        },
        "is_franchise": {
            "importance": fi.get("is_franchise", 0),
            "what": "Whether the film is part of a franchise or an original story (1 = franchise, 0 = original).",
            "why": "Franchise films come with a built-in fanbase. They're more predictable and tend to open higher — but originals can surprise.",
            "tip": "Sequels typically earn 15–25% more than standalone originals."
        },
    }

    # Sort by importance
    sorted_feats = sorted(feature_explanations.items(), key=lambda x: x[1]["importance"], reverse=True)

    for feat_name, info in sorted_feats:
        imp_pct = info["importance"] * 100
        with st.expander(f"**{feat_name.replace('_', ' ').title()}** — {imp_pct:.1f}% importance"):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**Importance:** `{imp_pct:.1f}%`")
                st.progress(min(info["importance"] * 3, 1.0))
            with col2:
                st.markdown(f"**What it is:** {info['what']}")
                st.markdown(f"**Why it matters:** {info['why']}")
                st.markdown(f"💡 **Tip:** {info['tip']}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("✅ How Accurate Is Our Model?")
    st.write("We held back 20% of our data (60 films) for testing. The model never saw these during training.")

    # Metrics in plain English
    st.subheader("Accuracy Metrics — Plain English")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']}")
        st.markdown("""
**What it means:**  
On average, our prediction is off by **{} ROI points**.

If a film's actual ROI is 3.0×, our model predicts somewhere between **{:.1f}× and {:.1f}×**.
        """.format(metrics['mae'], 3.0 - metrics['mae'], 3.0 + metrics['mae']))

    with c2:
        st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']}")
        st.markdown("""
**What it means:**  
RMSE penalises big misses more than MAE. Our score of **{}** means the model avoids large outlier errors.

Lower is better. Anything below 2.0 is solid for ROI prediction.
        """.format(metrics['rmse']))

    with c3:
        st.metric("R² Score", f"{metrics['r2']}")
        st.markdown("""
**What it means:**  
R² measures how much of the ROI variation the model explains.

**{:.0f}%** of ROI variation is explained by our 10 features. Film ROI has inherent randomness — this is expected.
        """.format(metrics['r2'] * 100))

    st.markdown("---")

    # How we measured — train/test split explanation
    st.subheader("How We Measured Accuracy")
    st.markdown("""
We used a method called **Train / Test Split**:

1. **Training set (80% = 240 films)** — the model learns patterns from these films
2. **Test set (20% = 60 films)** — we hide these from the model, then ask it to predict their ROI
3. We compare predictions to the real ROI values and calculate the error

This is like studying from a textbook but being tested on questions you've never seen before.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Predicted vs Actual ROI")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, y_pred, alpha=0.5, color="#378ADD", s=30)
        lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
        ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
        ax.set_xlabel("Actual ROI")
        ax.set_ylabel("Predicted ROI")
        ax.set_title("Each dot = one film from the test set")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.caption("Dots closer to the red line = more accurate predictions.")

    with col2:
        st.subheader("Prediction Error Distribution")
        errors = y_pred - y_test.values
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(errors, bins=20, color="#378ADD", edgecolor="white", alpha=0.85)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Zero error")
        ax.set_xlabel("Prediction Error (Predicted − Actual)")
        ax.set_ylabel("Number of Films")
        ax.set_title("How errors are spread")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
        st.caption("A bell curve centred at 0 means the model is unbiased — it doesn't consistently over or under-predict.")

    st.markdown("---")
    st.subheader("Why Isn't R² Higher?")
    st.info("""
**Film ROI is genuinely hard to predict.** Even Hollywood studios with decades of data can't consistently predict hits.

Reasons R² stays moderate:
- A film can be great on paper but flop due to bad timing, competition, or bad word-of-mouth
- A low-budget horror can go viral and make 20× return unexpectedly
- Our dataset is synthetic — real-world data with more features would improve accuracy significantly

**The model is most useful for:** comparing scenarios, spotting high-risk / high-reward combinations, and avoiding obvious bad decisions.
    """)
