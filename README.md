# 🏁 F1 Race Predictor & Tactical Dashboard

An end-to-end Data Science and Machine Learning web application that predicts Formula 1 podium finishes using historical race data, advanced feature engineering, and an optimized Gradient Boosting classifier, wrapped in an interactive Streamlit UI.

---

## 🚀 Features

- **Historical Data Integration:** Leverages decades of official F1 data (Ergast API / Kaggle datasets) spanning races, drivers, constructors, and qualifying sessions.
- **Advanced Feature Engineering:** Computes dynamic features such as driver recent form, team pace (rolling averages avoiding data leakage), and qualifying performance metrics.
- **Optimized Machine Learning Model:** Built using **XGBoost Classifier**, fine-tuned with hyperparameter optimization (`RandomizedSearchCV`) and automated optimal threshold tuning to maximize the F1-Score (balancing Precision and Recall).
- **Interactive Web UI:** A clean, responsive dashboard built with **Streamlit** allowing users to simulate race conditions, track difficulties, and driver parameters in real-time.

---

## 🛠️ Tech Stack & Libraries

- **Language:** Python 3.10+
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Web Interface:** Streamlit

---

## 📂 Project Structure

```text
f1-predictor/
│
├── data_prep.ipynb            # Data cleaning and feature engineering notebook
├── app.py                     # Streamlit interactive web application
├── f1_model.pkl               # Trained XGBoost model (serialized)
├── le_driver.pkl              # Label Encoder for drivers
├── le_constructor.pkl         # Label Encoder for constructors
├── le_circuit.pkl             # Label Encoder for circuits
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation


How to Run Locally:

1. Clone the repository:
    git clone [https://github.com/YOUR_USERNAME/f1-race-predictor.git](https://github.com/YOUR_USERNAME/f1-race-predictor.git)
    cd f1-predictor

2. Create and activate a virtual environment:
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On Mac/Linux:
    source venv/bin/activate

3. Install dependencies:
    pip install -r requirements.txt

4. Run the Streamlit application:
    streamlit run app.py


Model Performance & Validation
To prevent overfitting on past historical data, the model uses a Time-Series Split validation strategy (training on data up to 2022, testing on 2023+). The classification threshold is optimized using Precision-Recall curves to ensure high reliability when predicting podium outcomes.
