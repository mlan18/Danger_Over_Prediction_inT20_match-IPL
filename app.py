import streamlit as st
import pandas as pd
import joblib

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

#  Match ID (ALL matches shown)
match_ids = sorted(data["Match ID"].unique())
match_id = st.selectbox("Select Match ID", match_ids)

#  Innings
innings = st.selectbox("Select Innings", [1, 2])

filtered_data = data[
    (data["Match ID"] == match_id) &
    (data["Innings"] == innings)
]

#  If innings not played
if filtered_data.empty:
    st.warning("⚠️ This innings was not played (match may be interrupted or abandoned).")
    st.stop()

# Over selection (ONLY valid overs)
available_overs = sorted(filtered_data["Over"].unique())
max_over = max(available_overs)

st.info(f"📊 This innings was played till **over {max_over}**")
over = st.selectbox("Select Over", available_overs)

if st.button("Predict Danger Over"):

    row = filtered_data[filtered_data["Over"] == over]

    X = row[FEATURES]
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("📌 Result")

    if pred == 1:
        st.error(f"🔥 Danger Over\n\nProbability: **{prob:.2f}**")
    else:
        st.success(f"🟢 Safe Over\n\nDanger Probability: **{prob:.2f}**")

