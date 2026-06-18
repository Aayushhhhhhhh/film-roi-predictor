# 🎬 Film ROI Predictor

A Streamlit app that predicts the ROI of a film using machine learning.

## What it does

**Tab 1 — Predict ROI**  
Enter your film details (genre, budget, trailer views, etc.) and get a predicted ROI with a profit/loss verdict.

**Tab 2 — What the Data Tells Us**  
See which features matter most, what each one means, and why it influences ROI.

**Tab 3 — How Accurate Is Our Model**  
Plain-English explanation of MAE, RMSE, and R² with visual charts.

---

## Project Structure

```
film-roi-predictor/
├── app.py                      # Streamlit app — everything is here
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── film_roi_dataset.csv    # 300-film synthetic dataset
```

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/film-roi-predictor.git
cd film-roi-predictor

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

---

## Model

- **Algorithm:** Gradient Boosting Regressor
- **Features:** 10 (genre, budget, marketing budget, trailer views, RT score, tracking score, director ROI, release season, MPAA rating, franchise flag)
- **Target:** ROI = (Total Revenue − Total Cost) / Total Cost
- **Train/Test Split:** 80% / 20%

---

## Features Used

| Feature | Type | Why it matters |
|---|---|---|
| Genre | Categorical | Strongest predictor — horror has highest ROI, action needs huge budget |
| Production Budget | Numeric | More spend ≠ more ROI |
| Marketing Budget | Numeric | Drives awareness, but overspending hurts ROI |
| Trailer Views (24hr) | Numeric | Best early signal of audience excitement |
| Rotten Tomatoes Score | Numeric | Drives word-of-mouth and long-run performance |
| Pre-release Tracking | Numeric | Audience awareness before release |
| Director Avg ROI | Numeric | Track record of profitable films |
| Release Season | Categorical | Summer & Holiday earn 20–30% more |
| MPAA Rating | Categorical | PG-13 reaches the widest audience |
| Franchise / Sequel | Binary | Built-in fanbase = more predictable opening |
