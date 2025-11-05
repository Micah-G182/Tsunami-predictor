import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image
model= pickle.load(open('kn4_model.pkl','rb'))
st.markdown("""

   <style>
      body {
            background-color: lightblue;
            }
            .block-container {
            border: 50px double;
            border-color: green purple;
            background-color: rgba(173,216,230,0.3);
            padding: 20px;
            border-radius: 30px;
            }
            p {
            color: purple;
            }
            h1 {
            text-align: center;
            }
            .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-line: 7px;
            border-color: red;
            border-radius: 5px;
            cursor: pointer;
            }
            .stButton>button:hover {
            background-color: #3e8e41;
            }
    </style>
""",unsafe_allow_html=True
)
st.set_page_config(page_title="micy_G Tsunami Predictor",layout="centered")
st.markdown("<h1 style='color: maroon;text-decoration: underline; text-decoration-color: green;'>MGN Tsunami Predicting Machine Learning Application</h1>",unsafe_allow_html=True)
st.write("____________________________________________________")
st.write("____________________________________________________")
image = Image.open("todd-turner-Af9cNES03LU-unsplash.jpg")
st.image(image, caption='Tsunami Image',use_container_width=True)
with st.form("Prediction form"):
    st.subheader("Enter the following seismic parameters to predict tsunami occurance possibility:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Earthquake Magnitude(Richter Scale):")
        magnitude= st.number_input("Enter magnitude",min_value=6.5,max_value=9.1,value=6.5,step=0.1)
        st.markdown("### Community Decimal Intensity(felt intensity):")
        cdi= st.number_input("Enter intensity",min_value=0.0,max_value=9.0,value=0.0,step=1.0)
        st.markdown("### Modified Mercali Intesity(instrumental):")
        mmi= st.number_input("Enter value",min_value=1.0,max_value=9.0,value=1.0,step=1.0)
        st.markdown("### Event Significance Score:")
        sig= st.number_input("Enter score",min_value=650.0,max_value=2910.0,value=650.0,step=1.0)

    with col2:
        st.markdown("### Number of Seismic Monitoring Stations:")
        nst= st.number_input("Enter stations number",min_value=0.0,max_value=934.0,value=0.0,step=1.0)
        st.markdown("### Distance to Nearest Seismic Station(degrees):")
        dmin= st.number_input("Enter distance",min_value=0.0,max_value=17.7,value=0.0,step=0.001)
        st.markdown("### Azimuthal Gap Between Stations(degrees):")
        gap= st.number_input("Enter gap",min_value=0.0,max_value=239.0,value=0.0,step=1.0)
        st.markdown("### Earthquake Focal Depth(Km):")
        depth= st.number_input("Enter depth",min_value=2.7,max_value=671.0,value=2.7,step=0.001)

    with col3:
        st.markdown("### Epicenter Latitude(WGS84):")
        latitude= st.number_input("Enter latitude",min_value=-61.8,max_value=71.6,value=0.0,step=0.0001)
        st.markdown("### Epicenter Longitude(WGS84):")
        longitude= st.number_input("Enter longitude",min_value=-180.0,max_value=180.0,value=0.0,step=0.0001)
        st.markdown("### Year:")
        Year= st.number_input("Enter year:",min_value=2025.0,max_value=2099.0,value=2025.0,step=1.0)
        st.markdown("### Month:")
        Month= st.number_input("Enter month:",min_value=1.0,max_value=12.0,value=1.0,step=1.0) 
    submitted = st.form_submit_button("Predict")
if submitted:
    input_data= pd.DataFrame([[magnitude,cdi,mmi,sig,nst,dmin,gap,depth,latitude,longitude,Year,Month]],columns=['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude','Year','Month'])
    prediction= model.predict(input_data)[0]
    if prediction == 1:
        st.error("Prediction: High chances of tsunami, take precaution.")
    else:
        st.success("Prediction: No Likelihood of Tsunami") 

st.write("Model accuracy score = 82.8 %")
st.write("_____________________________")
st.write("_____________________________")
st.write("by Micah G.N, e-mail: mgetare24@gmail.com")