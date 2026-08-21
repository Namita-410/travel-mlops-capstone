"""
Streamlit app: Travel Insights & Recommendations
---------------------------------------------------
Two tabs:
  1. Flight Price Predictor -- calls the Flask API (or loads the model
     directly if the API isn't running) to predict a flight price.
  2. Hotel Recommendations -- looks up a user's travel history and shows
     the recommender's top picks, plus dataset visualizations.

Run with: streamlit run streamlit_app/app.py
"""

import os
import sys
import joblib
import requests
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "models"))
from train_recommender import recommend_for_user  # reuse the same scoring logic

DATA_DIR = BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
API_URL = os.environ.get("FLIGHT_API_URL", "http://localhost:5000")

st.set_page_config(page_title="Travel Insights", layout="wide")


@st.cache_data
def load_data():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    flights = pd.read_csv(os.path.join(DATA_DIR, "flights.csv"))
    hotels = pd.read_csv(os.path.join(DATA_DIR, "hotels.csv"))
    return users, flights, hotels


@st.cache_resource
def load_recommender():
    return joblib.load(os.path.join(MODELS_DIR, "recommender.joblib"))


@st.cache_resource
def load_regression_pipeline():
    return joblib.load(os.path.join(MODELS_DIR, "flight_price_model.joblib"))


users, flights, hotels = load_data()
rec_artifacts = load_recommender()

st.title("Travel Insights & Recommendations")

tab1, tab2, tab3 = st.tabs(["Flight Price Predictor", "Hotel Recommendations", "Dataset Insights"])

with tab1:
    st.header("Predict a Flight Price")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.selectbox("From", sorted(flights["from"].unique()))
        dest_options = sorted([c for c in flights["to"].unique() if c != origin])
        destination = st.selectbox("To", dest_options)
        flight_type = st.selectbox("Flight type", sorted(flights["flightType"].unique()))
    with col2:
        agency = st.selectbox("Agency", sorted(flights["agency"].unique()))
        time_hrs = st.slider("Flight time (hours)", 0.3, 3.0, 1.5, 0.01)
        distance_km = st.slider("Distance (km)", 100.0, 1000.0, 500.0, 1.0)
        date = st.date_input("Date")

    if st.button("Predict price", type="primary"):
        payload = {
            "from": origin, "to": destination, "flightType": flight_type,
            "agency": agency, "time": time_hrs, "distance": distance_km,
            "date": date.strftime("%m/%d/%Y"),
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=3)
            resp.raise_for_status()
            price = resp.json()["predicted_price"]
            st.success(f"Predicted price via API: **${price:,.2f}**")
        except requests.exceptions.RequestException:
            # API not reachable -> fall back to loading the model directly
            artifact = load_regression_pipeline()
            df = pd.DataFrame([payload])
            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
            df["quarter"] = df["date"].dt.quarter
            df["distance_per_hour"] = df["distance"] / df["time"]
            cols = artifact["categorical_cols"] + artifact["numeric_cols"]
            price = artifact["pipeline"].predict(df[cols])[0]
            st.info(f"API unreachable — predicted locally: **${price:,.2f}**")

with tab2:
    st.header("Hotel Recommendations")
    user_code = st.selectbox("Select a user", sorted(users["code"].unique()))
    user_row = users[users["code"] == user_code].iloc[0]
    st.write(f"**{user_row['name']}** — {user_row['company']}, age {user_row['age']}")

    interactions = rec_artifacts["interactions"]
    if user_code in interactions.index:
        visited = interactions.loc[user_code]
        visited = visited[visited > 0].sort_values(ascending=False)
        st.write("Places already stayed in:", ", ".join(visited.index.tolist()) or "none yet")

    recs = recommend_for_user(rec_artifacts, user_code, top_n=3)
    if recs:
        st.subheader("Top recommendations")
        for place in recs:
            avg_price = rec_artifacts["place_avg_price"][place]
            st.write(f"- **{place}** — avg nightly price ${avg_price:,.2f}")
    else:
        st.write("No new places to recommend — this user has already stayed in every city in the catalog.")

with tab3:
    st.header("Dataset Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.box(flights, x="flightType", y="price", title="Flight price by class")
        st.plotly_chart(fig1, use_container_width=True)
        fig3 = px.histogram(users, x="age", color="gender", title="User age distribution by gender")
        st.plotly_chart(fig3, use_container_width=True)
    with c2:
        route_counts = flights.groupby(["from", "to"]).size().reset_index(name="count")
        fig2 = px.bar(
            route_counts.sort_values("count", ascending=False).head(10),
            x="count", y=route_counts.sort_values("count", ascending=False).head(10)["from"] + " -> " +
              route_counts.sort_values("count", ascending=False).head(10)["to"],
            orientation="h", title="Top 10 busiest routes",
        )
        st.plotly_chart(fig2, use_container_width=True)
        fig4 = px.bar(hotels.groupby("place")["price"].mean().reset_index(),
                      x="place", y="price", title="Average hotel price by city")
        st.plotly_chart(fig4, use_container_width=True)
