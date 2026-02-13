import pandas as pd
import joblib
import pickle

data = "Training_cleaned_data.csv"
df = pd.read_csv(data)

X = df.drop(columns = ['prognosis'])
y = df['prognosis']
    
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=13)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=5,random_state=13)
model.fit(X_train,y_train)
#save the model:
#joblib.dump(model,"smart_clinic_app.pkl")
with open('smart_clinic_app3.pkl','wb') as f:
    pickle.dump(model,f)
print("model successfully saved")

