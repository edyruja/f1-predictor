import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Configurarea paginii
st.set_page_config(page_title="F1 Race Predictor", layout="wide")
st.title("🏁 F1 Race Predictor")

@st.cache_resource
def load_model():
    model = joblib.load('f1_model.pkl')
    le_driver = joblib.load('le_driver.pkl')
    le_constructor = joblib.load('le_constructor.pkl')
    le_circuit = joblib.load('le_circuit.pkl')
    return model, le_driver, le_constructor, le_circuit

model, le_driver, le_constructor, le_circuit = load_model()

# --- NOU: Încărcăm numele complete direct din CSV ---
@st.cache_data
def load_driver_mapping():
    try:
        # Asigură-te că drivers.csv este în același folder cu app.py
        drivers_df = pd.read_csv('drivers.csv')
        # Creăm un dicționar: {'leclerc': 'Charles Leclerc', 'norris': 'Lando Norris'}
        return dict(zip(drivers_df['driverRef'], drivers_df['forename'] + ' ' + drivers_df['surname']))
    except FileNotFoundError:
        return {}

driver_names_dict = load_driver_mapping()

# Funcții de formatare
def format_driver(driver_ref):
    # Caută în dicționar. Dacă nu găsește (fallback), pune doar literă mare
    return driver_names_dict.get(driver_ref, str(driver_ref).replace('_', ' ').title())

def format_circuit(circuit_ref):
    return str(circuit_ref).replace('_', ' ').title()
# ---------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    st.subheader("Track and Race Conditions")
    # Folosim format_circuit aici
    track = st.selectbox("Select Track", le_circuit.classes_, format_func=format_circuit)
    
    
    track_diff = st.slider("Track Difficulty: 1-10", 1, 10, 7)
    race_dist = st.text("Race Distance: 308 km")
    tire_deg = st.slider("Tire Degradation (1-10, lower is better)", 1, 10, 8)
    weather = st.selectbox("Weather Conditions", ["Clear", "Cloudy", "Rain"])

with col2:
    st.subheader("Driver and Car Parameters")
    # Folosim format_driver aici
    driver = st.selectbox("Select Driver", le_driver.classes_, format_func=format_driver)
    grid = st.number_input("Grid Position", min_value=1, max_value=20, value=10)
    
    car_perf = st.slider("Overall Car Performance", 0, 10, 8)
    top_speed = st.slider("Top Speed (km/h)", 300, 360, 335)
    acceleration = st.slider("Acceleration (0-100 km/h in seconds)", 2.0, 3.5, 2.75)
    
    q1_time = st.text_input("Q1 Time (format: m:ss.ms)", "1:41.000")
    q2_time = st.text_input("Q2 Time (format: m:ss.ms)")

st.markdown("---")
st.markdown("""
*Join decades of race, qualifying and driver data, engineer features like grid position, recent form and team pace, then train a gradient boosting classifier to predict who finishes on the podium. Validate it across seasons so it is not just overfitting the past. A good mix of feature engineering and classification on a highly relevant dataset.*
""")

if st.button("🔮 Predict Podium Probability", use_container_width=True):
    try:
        driver_enc = le_driver.transform([driver])[0]
        circuit_enc = le_circuit.transform([track])[0]
        
        simulated_form = 20 - (car_perf * 1.9)
        input_data = np.array([[grid, simulated_form, simulated_form, driver_enc, 0, circuit_enc]])
        
        prob = model.predict_proba(input_data)[0][1] * 100
        
        # Folosim format_driver pentru afișarea rezultatului final
        st.success(f"### 🏆 Șanse ca {format_driver(driver)} să termine pe podium: {prob:.1f}%")
        
        # Schimbăm de la 50 la pragul optim descoperit de model
        if prob >= 60.1:
            st.balloons()
            
    except Exception as e:
        st.error(f"Eroare la predicție: {e}")