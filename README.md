# ⚡ IPL Danger Over Prediction 

> 🎯 A machine‑learning project to identify **high‑risk (“danger”) overs** in IPL cricket matches using ball‑by‑ball data.

---

## 📌 Project Overview

Cricket matches often change momentum within a single over.  
This project predicts whether an upcoming over is a **Danger Over** based on match context such as:
- Current score
- Wickets fallen
- Run rate pressure
- Recent performance

The system is built using **Python, Machine Learning, and Streamlit** and provides real‑time predictions through a simple web interface.

---

## 🧠 What is a *Danger Over*?

An over is classified as **Danger Over** if:
- 🏏 The batting team scores **high runs**, **or**
- 💥 A wicket falls causing a momentum shift

This helps identify **high‑pressure overs** that significantly impact match outcomes.

---

## 🗂 Dataset

- 📄 **IPL Ball‑by‑Ball Dataset**
- 🔢 ~240,000+ deliveries
- 🏟 Multiple seasons, teams, and venues

Each ball record contains:
- Match ID, Innings, Over, Ball
- Batter & Bowler details
- Runs, Extras, Wickets
- Match context (target, score, wickets)

---

## 🔧 Feature Engineering

The model uses **over‑level engineered features**, including:

- 📈 Cumulative runs before over  
- ❌ Wickets fallen before over  
- ⏳ Balls remaining  
- ⚡ Current run rate  
- 🎯 Required run rate (chasing innings)  
- 🔁 Runs in last over  
- 🔁 Runs in last 3 overs  
- 💥 Wickets in last 3 overs  

> ⚠️ Only features available **before the over starts** are used (no data leakage).

---

## 🤖 Machine Learning Models

- 🌲 **Random Forest Classifier**
- 🚀 **XGBoost Classifier**
- 🔍 Hyperparameter tuning using `RandomizedSearchCV`
- 🛑 Match‑wise splitting using `GroupShuffleSplit` to avoid leakage

### 📊 Model Performance
- Accuracy: **~65–75% (realistic & leakage‑free)**
- Evaluated using:
  - Accuracy
  - Precision / Recall
  - Confusion Matrix

---

## 🖥 Web Application (Streamlit)

An interactive UI built with **Streamlit** allows users to:
- Enter **Match ID, Innings, Over**
- Predict if the over is **Danger** or **Safe**
- View probability with **color‑coded output**
  - 🔴 Red → Danger Over  
  - 🟢 Green → Safe Over  

---

## 🏗 Project Structure

```text
📦 Danger-Over-Prediction
├── 📄 ball_by_ball_ipl.csv
├── 📄 over_level_engineered.csv
├── 📄 01_feature_engineering.py
├── 📄 02_train_model.py
├── 📄 03_predict.py
├── 📄 app.py
├── 📦 danger_over_model.joblib
├── 📦 danger_over_scaler.joblib
└── 📄 README.md
```
### Thank You 🌸

