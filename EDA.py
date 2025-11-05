import pandas as pd
import numpy as np

df= pd.read_csv(r"C:\Users\USER\Desktop\STREAMLIT\TSUNAMI\earthquake_data_tsunami.csv")
df.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude','Year','Month']]= scaler.fit_transform(df[['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude','Year','Month']])
df.to_csv("earthquake_data_tsunami_cleaned3.csv")