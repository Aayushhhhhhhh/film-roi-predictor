import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# 1. Setup the Web Page
st.set_page_config(page_title="Film ROI Predictor", layout="wide")
st.title("🎬 Film ROI Prediction Engine")
st.write("Powered by Gradient Boosting Machine Learning")

# 2. Load and Train the Model (Cached for speed)
@st.cache_resource
def train_model():
    # Load your exact dataset
    df = pd.read_csv('film_roi_dataset.csv')
    
    features = ['genre', 'production_budget', 'trailer_views_24hr', 'director_avg_roi', 'pre_release_tracking']
    X = df[features]
    y = df['roi']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', ['production_budget', 'trailer_views_24hr', 'director_avg_roi', 'pre_release_tracking']),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), ['genre'])
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    
    model.fit(X, y)
    return model

model = train_model()

# 3. Build the User Interface Sidebar
st.sidebar.header("Configure Film Parameters")
selected_genre = st.sidebar.selectbox("Genre", ['Action', 'Horror', 'Animation', 'Sci-Fi', 'Comedy', 'Drama', 'Thriller'])
budget = st.sidebar.slider("Production Budget ($M)", 1.0, 300.0, 50.0) * 1000000
trailer_views = st.sidebar.slider("Trailer Views (First 24hr)", 1000000, 50000000, 10000000)
director_roi = st.sidebar.slider("Director's Historical ROI", 0.1, 10.0, 2.0)
tracking = st.sidebar.slider("Pre-release Tracking Score", 1, 100, 70)

# 4. Generate the Prediction
if st.sidebar.button("Predict ROI"):
    # Create a dataframe for the single new movie
    input_data = pd.DataFrame({
        'genre': [selected_genre],
        'production_budget': [budget],
        'trailer_views_24hr': [trailer_views],
        'director_avg_roi': [director_roi],
        'pre_release_tracking': [tracking]
    })
    
    # Run data through the pipeline
    prediction = model.predict(input_data)[0]
    
    # 5. Display Results
    st.subheader("Forecasted Financial Performance")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Estimated ROI Multiplier", f"{prediction:.2f}x")
    col2.metric("Total Production Cost", f"${budget/1000000:,.1f}M")
    
    est_revenue = budget + (budget * prediction)
    col3.metric("Projected Total Revenue", f"${est_revenue/1000000:,.1f}M")
    
    if prediction > 2.0:
        st.success("🟢 VERDICT: Strong Commercial Hit")
    elif prediction > 0:
        st.info("🟡 VERDICT: Profitable / Break-Even")
    else:
        st.error("🔴 VERDICT: Projected Financial Loss")
