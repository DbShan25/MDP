import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set up Streamlit
st.set_page_config(page_title="Multiple Disease Prediction System", layout="wide")
st.title("Multiple Disease Prediction System")
st.header("Health Report Assistance")

# Direct file paths
kidney_model_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/kidney_model_v2.pkl"
liver_model_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/liver_model_v2.pkl"
parkinsons_model_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/parkinsons_model_v2.pkl"

kidney_scaler_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/kidney_scaler.pkl"
liver_scaler_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/liver_scaler.pkl"
parkinsons_scaler_path = "C:/Users/Hxtreme/Jupyter_Notebook_Learning/Project3_V2/Analysis/parkinsons_scaler.pkl"

# Load models
kidney_model = pickle.load(open(kidney_model_path, 'rb'))
liver_model = pickle.load(open(liver_model_path, 'rb'))
parkinsons_model = pickle.load(open(parkinsons_model_path, 'rb'))

# Load scalers (if used during training)
def load_scaler(scaler_path):
    try:
        return pickle.load(open(scaler_path, "rb"))
    except FileNotFoundError:
        return None

kidney_scaler = load_scaler(kidney_scaler_path)
liver_scaler = load_scaler(liver_scaler_path)
parkinsons_scaler = load_scaler(parkinsons_scaler_path)

# Tabs for different diseases
tab1, tab2, tab3 = st.tabs(["Liver Disease Prediction", 
                            "Kidney Disease Prediction", 
                            "Parkinson’s Disease Prediction"])

# **🔹 LIVER DISEASE PREDICTION**
with tab1:
    st.subheader("Liver Disease Health Report Data & Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase', min_value=0, format="%d", key="aspartate_aminotransferase_liver")
    with col2:
        Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase', min_value=0, format="%d", key="alkaline_phosphotase_liver")
    with col3:
        Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase', min_value=0, format="%d", key="alamine_aminotransferase_liver")
    with col4:
        Age = st.number_input('Age', min_value=0, format="%d", key="age_liver")
    with col5:
        Total_Bilirubin = st.number_input('Total Bilirubin', min_value=0.0, format="%.2f", key="total_bilirubin_liver")

    with col1:
        Albumin = st.number_input('Albumin', min_value=0.0, format="%.2f", key="albumin_liver")

    if st.button("Predict Liver Disease"):
        try:
            user_input = np.array([
                Aspartate_Aminotransferase, Alkaline_Phosphotase, Alamine_Aminotransferase,
                Age, Total_Bilirubin, Albumin
            ]).reshape(1, -1)

            #if liver_scaler:
             #   user_input = liver_scaler.transform(user_input)

            prediction = liver_model.predict(user_input)
            st.success("Liver disease: +Ve" if prediction[0] == 1 else "Liver disease: -Ve")

        except Exception as e:
            st.error(f"Error processing input: {e}")

# **🔹 KIDNEY DISEASE PREDICTION**
with tab2:
    st.subheader("Kidney Disease Health Report Data & Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Haemoglobin = st.number_input('Haemoglobin', min_value=0.0, format="%.2f", key="haemoglobin_kidney")
    with col2:
        Packed_Cell_Volume = st.number_input('Packed Cell Volume', min_value=0, format="%d", key="packed_cell_volume_kidney")
    with col3:
        Specific_Gravity = st.number_input('Specific Gravity', min_value=1.0, max_value=1.05, format="%.3f", key="specific_gravity_kidney")
    with col4:
        Albumin = st.number_input('Albumin', min_value=0.0, format="%.2f", key="albumin_kidney")
    with col5:
        Hypertension = st.number_input('Hypertension (0/1)', min_value=0, max_value=1, format="%d", key="hypertension_kidney")

    with col1:
        Diabetes_Mellitus = st.number_input('Diabetes Mellitus (0/1)', min_value=0, max_value=1, format="%d", key="diabetes_mellitus_kidney")
    with col2:
        Blood_Urea = st.number_input('Blood Urea', min_value=0.0, format="%.2f", key="blood_urea_kidney")
    with col3:
        Serum_Creatinine = st.number_input('Serum Creatinine', min_value=0.0, format="%.2f", key="serum_creatinine_kidney")

    if st.button("Predict Kidney Disease"):
        try:
            user_input = np.array([
                Haemoglobin, Packed_Cell_Volume, Specific_Gravity,
                Albumin, Hypertension, Diabetes_Mellitus, Blood_Urea,
                Serum_Creatinine
            ]).reshape(1, -1)

            #if kidney_scaler:
             #   user_input = kidney_scaler.transform(user_input)

            prediction = kidney_model.predict(user_input)
            st.success("Kidney disease: +Ve" if prediction[0] == 1 else "Kidney disease: -Ve")

        except Exception as e:
            st.error(f"Error processing input: {e}")

# **🔹 PARKINSON’S DISEASE PREDICTION**
with tab3:
    st.subheader("Parkinson’s Disease Health Report Data & Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Spread1 = st.number_input('Spread1', min_value=-10.0, format="%.6f", key="spread1_parkinsons")
    with col2:
        PPE = st.number_input('PPE', min_value=0.0, format="%.6f", key="ppe_parkinsons")
    with col3:
        MDVP_Fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, format="%.6f", key="mdvp_fo_parkinsons")
    with col4:
        MDVP_PPQ = st.number_input('MDVP:PPQ', min_value=0.0, format="%.6f", key="mdvp_ppq_parkinsons")
    with col5:
        MDVP_Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, format="%.6f", key="mdvp_jitter_abs_parkinsons")

    with col1:
        Jitter_DDP = st.number_input('Jitter:DDP', min_value=0.0, format="%.6f", key="jitter_ddp_parkinsons")
    with col2:
        MDVP_RAP = st.number_input('MDVP:RAP', min_value=0.0, format="%.6f", key="mdvp_rap_parkinsons")
    with col3:
        Spread2 = st.number_input('Spread2', min_value=0.0, format="%.6f", key="spread2_parkinsons")
    with col4:
        D2 = st.number_input('D2', min_value=0.0, format="%.6f", key="d2_parkinsons")
    with col5:
        HNR = st.number_input('HNR', min_value=0.0, format="%.6f", key="hnr_parkinsons")

    if st.button("Predict Parkinson’s Disease"):
        try:
            user_input = np.array([[
                Spread1, PPE, MDVP_Fo, MDVP_PPQ, MDVP_Jitter_Abs,
                Jitter_DDP, MDVP_RAP, Spread2, D2, HNR
            ]])

            # Apply Scaling (if scaler exists)
            if parkinsons_scaler:
                user_input = parkinsons_scaler.transform(user_input)

            # Get Probability Prediction
            probabilities = parkinsons_model.predict_proba(user_input)
            prob_positive = probabilities[0][1]  # Probability of Parkinson’s (1)
            prob_negative = probabilities[0][0]  # Probability of No Parkinson’s (0)

            # **Print Raw Probability to Debug**
            #st.write(f"🛠 **Debug Info:** Parkinson’s Prob: {prob_positive:.4f}, No Parkinson’s Prob: {prob_negative:.4f}")

            # Adjust threshold (default = 0.6)
            threshold = 0.56
            prediction = 1 if prob_positive > threshold else 0

            # Display Results
            if prediction == 1:
                st.error(f"Parkinson’s Disease **Detected** ")
            else:
                st.success(f"No Parkinson’s Disease ")

        except Exception as e:
            st.error(f"❌ Error processing input: {e}")
