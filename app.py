import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and engineered data
model = joblib.load("danger_over_model.joblib")
scaler = joblib.load("danger_over_scaler.joblib")
data = pd.read_csv("over_level_engineered.csv")

FEATURES = [
    "cum_runs_before_over",
    "cum_wickets_before_over",
    "cum_balls_before_over",
    "overs_completed_before",
    "balls_remaining_before_over",
    "run_rate_so_far",
    "req_run_rate",
    "runs_last_over",
    "runs_last_3_overs",
    "wickets_last_3_overs"
]

st.title("⚡ Cricket Danger Over Prediction")

match_id = st.number_input("Match ID", min_value=1, step=1)
innings = st.selectbox("Innings", [1, 2])
over = st.number_input("Over (1–20)", min_value=1, max_value=20)

if st.button("Predict Danger Over"):

    # 🔍 Fetch real over data using Match ID
    row = data[
        (data["Match ID"] == match_id) &
        (data["Innings"] == innings) &
        (data["Over"] == over)
    ]

    if row.empty:
        st.error("❌ Match / Innings / Over not found in dataset")
    else:
        X = row[FEATURES]
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1]

        st.subheader("📌 Result")

        if pred == 1:
            st.error(f"🔥 Danger Over (Probability: {prob:.2f})")
        else:
            st.success(f"🟢 Safe Over (Danger Probability: {prob:.2f})")
