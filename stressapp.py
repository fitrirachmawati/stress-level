import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
import sklearn
import joblib

st.set_page_config(page_title="Stress Level Prediction", layout="wide")

# Define dictionaries for mapping numerical values to strings
gender_dict = {1: 'Male', 0: 'Female'}
occupation_dict = {0: 'Software Engineer', 
                   1: 'Doctor',
                   2: 'Sales Representative',
                   3: 'Teacher',
                   4: 'Nurse',
                   5: 'Engineer',
                   6: 'Accountant',
                   7: 'Scientist',
                   8: 'Lawyer',
                   9: 'Salesperson',
                   10: 'Manager'}
bmi_dict = {1: 'Overweight', 0: 'Normal', 2: 'Obese'}
sleep_disorder_dict = {0: 'None', 1: 'Sleep Apnea', 2: 'Insomnia'}

st.sidebar.header('Input Features')

# Inputan Kategorikal
gender = st.sidebar.selectbox("Select Gender", ["Female", "Male"])
occupation = st.sidebar.selectbox("Select Occupation", ["Software Engineer", "Doctor", "Sales Representative", "Teacher", "Nurse", "Engineer", "Accountant", "Scientist", "Lawyer", "Salesperson", "Manager"])
bmi_category = st.sidebar.selectbox("Select BMI Category", ["Normal", "Overweight","Obese"])
sleep_disorder = st.sidebar.selectbox("Select Sleep Disorder", ["None", "Sleep Apnea", "Insomnia"])

# Create input dictionary with label-to-numeric mapping
label_to_numeric = {
    'Male': 1, 'Female': 0,
    'Software Engineer': 0,
    'Doctor': 1,
    'Sales Representative': 2,
    'Teacher': 3,
    'Nurse': 4,
    'Engineer': 5,
    'Accountant': 6,
    'Scientist': 7,
    'Lawyer': 8,
    'Salesperson': 9,
    'Manager': 10,
    'Overweight': 1,
    'Normal': 0,
    'Obese': 2,
    'None': 0,
    'Sleep Apnea': 1,
    'Insomnia': 2
}

# Create input dictionary
data = {
    'Gender': label_to_numeric[gender],
    'Occupation': label_to_numeric[occupation],
    'BMI Category': label_to_numeric[bmi_category],
    'Sleep Disorder': label_to_numeric[sleep_disorder],
}

# Inputan Numerik
age = st.sidebar.text_input("Age", "30")  # Default value: "30"
sleep_duration = st.sidebar.text_input("Sleep Duration (max 12 hours)", "8.0")  # Default value: "8.0"
physical_activity = st.sidebar.text_input("Physical Activity Level (minutes per day)", "30")  # Default value: "30"
quality_of_sleep = st.sidebar.slider("Quality of Sleep (0-10)", 0, 10, 5)  # Default value: 5
heart_rate = st.sidebar.text_input("Heart Rate (max 100 bpm)", "70")  # Default value: "70"
daily_steps = st.sidebar.text_input("Daily Steps", "5000")  # Default value: "5000"

# Fungsi untuk mengkategorikan Blood Pressure
def categorize_blood_pressure(blood_pressure):
    systolic, diastolic = map(int, blood_pressure.split('/'))

    if systolic < 120 and diastolic < 80:
        return 'Normal'
    elif (120 <= systolic <= 139) or (80 <= diastolic <= 89):
        return 'Pra-hipertensi'
    elif (140 <= systolic <= 159) or (90 <= diastolic <= 99):
        return 'Hipertensi tingkat 1'
    elif (systolic >= 160) or (diastolic >= 100):
        return 'Hipertensi tingkat 2'
    elif (systolic > 140) and (diastolic < 90):
        return 'Hipertensi Sistolik Terisolasi'
    else:
        return 'Lainnya'

# Kamus label numerik untuk Blood Pressure
blood_pressure_label_to_numeric = {
    'Normal': 0,
    'Pra-hipertensi': 1,
    'Hipertensi tingkat 1': 2,
    'Hipertensi tingkat 2': 3,
    'Hipertensi Sistolik Terisolasi': 4,
    'Lainnya': 5
}

# Inputan Blood Pressure
blood_pressure_input = st.sidebar.text_input("Blood Pressure (e.g., 120/80)", "120/80")
blood_pressure_category = categorize_blood_pressure(blood_pressure_input)
blood_pressure_numeric = blood_pressure_label_to_numeric.get(blood_pressure_category, -1)

# Load model menggunakan pickle (lakukan ini sekali saat aplikasi dimulai)
loaded_model = pickle.load(open('best_model_gbr.pkl', 'rb'))

# Fungsi untuk memprediksi Stress Level
def predict_stress_level(input_df):
    # Prediksi Stress Level
    prediction = loaded_model.predict(input_df)
    return prediction[0]

# Menampilkan konten utama
def run():
    st.write("""
    # Stress Level Prediction Dashboard
    This dashboard created by : [@fitrirachmawati](https://www.linkedin.com/in/fitrirachmawati1004/)
    """)

    st.image("stress.png", width=800)

    st.write("""
    The purpose of this application is to predict stress levels based on various input features or factors. It is a valuable tool for users, including professionals in fields such as psychology, healthcare, or anyone interested in assessing stress levels.

    The prediction model takes into account a range of input factors, such as demographic information and lifestyle choices, to provide a comprehensive evaluation of an individual's stress level. Users can input specific data, and the model will generate a prediction, helping individuals and professionals gain insights into stress levels.

    By using this dashboard, users can better understand and predict stress levels in various contexts, aiding in decision-making, healthcare planning, and overall well-being.
    """)

   # Menampilkan hasil input
    st.write("## Hasil Input:")
    st.write(f"Gender: {gender}")
    st.write(f"Age: {age} years")
    st.write(f"Occupation: {occupation}")
    st.write(f"Sleep Duration: {sleep_duration} hours")
    st.write(f"Quality of Sleep: {quality_of_sleep}")
    st.write(f"Physical Activity Level: {physical_activity}")
    st.write(f"BMI Category: {bmi_category}")
    
    if blood_pressure_numeric == -1:
        st.error("Blood Pressure tidak valid. Pastikan format seperti '120/80' dan coba lagi.")
    else:
        st.write(f'Blood Pressure Category: {blood_pressure_category} (Numeric Value: {blood_pressure_numeric})')

    st.write(f"Heart Rate: {heart_rate} bpm")
    st.write(f"Daily Steps: {daily_steps} steps")

    # Menambahkan tombol "Predict Stress Level"
    if st.sidebar.button('Predict Stress Level'):
        # Menggabungkan input pengguna menjadi dataframe
        input_data = {
            'Gender': [label_to_numeric[gender]],
            'Age': [int(age)],
            'Occupation': [label_to_numeric[occupation]],
            'Sleep Duration': [float(sleep_duration)],
            'Quality of Sleep': [int(quality_of_sleep)],
            'Physical Activity Level': [int(physical_activity)],
            'Stress Level': [0],  # Stress Level akan diprediksi, awalnya diatur ke 0
            'BMI Category': [label_to_numeric[bmi_category]],
            'Blood Pressure': [blood_pressure_numeric],  # Menggunakan nilai numerik
            'Heart Rate': [int(heart_rate)],
            'Daily Steps': [int(daily_steps)]
        }

        input_df = pd.DataFrame(input_data)

        # Memprediksi Stress Level menggunakan fungsi predict_stress_level
        predicted_stress_level = predict_stress_level(input_df)

        # Fungsi untuk menggolongkan tingkat stres berdasarkan nilai numerik
        def categorize_stress_level(prediction):
            if prediction < 4.0:
                return "Very Low"
            elif 4.0 <= prediction < 4.2:
                return "Low"
            elif 4.2 <= prediction < 4.4:
                return "Moderate"
            elif 4.4 <= prediction < 4.6:
                return "High"
            elif prediction >= 4.6:
                return "Very High"
            else:
                return "Out of Range"
            
        # Fungsi untuk memberikan saran berdasarkan tingkat stres
        def get_stress_advice(stress_level):
            if stress_level == "Very Low":
                return "You are in a good mental state. Continue to maintain your mental health by exercising regularly and practicing meditation to maintain balance."
            elif stress_level == "Low":
                return "You are experiencing low stress levels. Continue to maintain your healthy routines and avoid situations that could increase stress."
            elif stress_level == "Moderate":
                return "You are experiencing moderate stress levels. Consider finding ways to manage stress, such as regular exercise or meditation."
            elif stress_level == "High":
                return "You are experiencing high stress levels. It's essential to seek support from friends or mental health professionals."
            elif stress_level == "Very High":
                return "You are in a state of very high stress. Seek help from a mental health professional immediately and avoid situations that could increase stress."
            else:
                return "Out of Range"
                

        # Menghitung hasil kualitatif tingkat stres
        qualitative_result = categorize_stress_level(predicted_stress_level)

        # Menampilkan hasil prediksi Stress Level secara kualitatif
        st.subheader('Predicted Stress Level:')
        st.write(f"**{qualitative_result}**")

         # Menampilkan saran berdasarkan tingkat stres
        st.subheader("Advice:")
        st.write(get_stress_advice(qualitative_result))


if __name__ == "__main__" :
    run()
