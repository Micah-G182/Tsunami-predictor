import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import io
from datetime import datetime
import joblib


st.markdown(
    """
<style>
.block-container {
    border: 50px solid purple;
    outline: 10px solid lightgreen;
    padding: 20px 20px;
    border-radius: 10px;
    background-color: rgba(0,0,180,0.3)
    }
p   {
    color: purple;
    }
body {
    background-color: rgba(0,0,180,0.3);
    }
h1  {
    color: green;
    font-family: Arial, sans-serif;
    font-size: 25px;
    }
h2  {
    color: green;
    font-family: Arial, sans-serif;
    font-size: 20px;
    }
h3  {
    color: green;
    font-family: Arial, sans-serif;
    font-size:18px;
    }
.stButton>button {
    background-color: rgba(0,0,40,0.2);
    color: brown;
    border: solid;
    padding: 20px 30px;
    cursor: pointer;
}
    </style>
    """,
    unsafe_allow_html=True,

)
st.set_page_config(page_title="MAAI-MAHIU GIRLS SMART CLINIC 2026 SCIENCE FAIR",layout="centered")
st.markdown("<h1 style='color: maroon;text-decoration: underline;text-decoration-color: green;'MAAI-MAHIU GIRLS SMART CLINIC FOR DISEASE DIAGNOSIS</h1>",unsafe_allow_html=True)
st.write("____________________________________________________")
#loading the dataset
df = pd.read_csv('Training_cleaned_data.csv')
model = joblib.load('smart_clinic_app3.pkl','rb')
#get symptoms list from the dataframe
symptoms_list = df.columns[:-1]
#streamlit app
st.title("MAAI-MAHIU GIRLS' SMART CLINIC DISEASE DIAGNOSIS")

image = Image.open("GeoPeshMMG.png")
st.image(image, caption='MMG Smart Clinic',use_container_width=True)
st.write("---------------------------------------")
#Time-based greeting:
now = datetime.now()
hour = now.hour
if hour < 12:
    greeting = "Good morning"
elif hour < 17:
    greeting = "Good afternoon"
else:
    greeting = "Good evening"
#Displaying welcome greeting in bold
st.write(f"**{greeting} our esteemed doctor. Welcome to Maai-Mahiu Girls' Smart Clinic**")
st.write("===========================================================================")
#Getting patient's information
patient_name = st.text_input("Enter Patient Name")
patient_age = st.number_input("Enter Patient Age")
patient_contact = st.text_input("Enter Patient Contact")

diagnosis_history = []

st.markdown("Select symptoms shown by the patient to predict the disease.")
#Searchable symptoms selection
selected_symptoms = st.multiselect("Search and select symptoms", symptoms_list)
#creation of input vector
if selected_symptoms:
    input_data = [1 if s in selected_symptoms else 0 for s in symptoms_list]
    input_data = np.array(input_data).reshape(1, -1)

    #prediction of the disease:
    if st.button("Predict"):
        if patient_name and patient_age and patient_contact:
            pred = model.predict(input_data)
            #st.write(f"## Predicted Disease: **{pred[0]}**")
            #st.markdown(f"##Predicted Disease: **{pred[0]}**")
            #st.markdown(f"Based on the symptoms: **{','.join(selected_symptoms)}**")
            now = datetime.now()
            diagnosis_time = now.strftime("%Y-%m-%d %H:%M:%S")
            diagnosis_history.append((patient_name,patient_age,patient_contact,pred[0],diagnosis_time))
            #Display of diagnosis result with date and time
            st.write(f"##Predicted Disease: **{pred[0]}**")
            st.write(f"Diagnosis made on: **{diagnosis_time}**")
        else:
            st.warning("Please fill in all patient information")  

#else:
    #st.warning("please select at least one symptom to predict the disease")
if diagnosis_history:
    history_df =pd.DataFrame(diagnosis_history, columns=["Patient Name","Age","Contact Number","Disease Diagnosed","Diagnosis Time"]) 
    st.dataframe(history_df)
    #Allowing users to download diagnosis history as a CSV file
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="Download Diagnosis History",
        data=csv,
        file_name="diagnosis_history.csv",
        mime="text/csv",
    )
else:
    st.write("Diagnosis History:")
    st.write("No diagnosis made yet")    
st.write("______________________________________________________________________")
st.write("Model accuracy score = 93.0%")
st.write("Model Engineers: Georgina and Peninah, 2026 Science and Technology Fair.")
st.write("Thank you.")

