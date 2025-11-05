import pandas as pd
df = pd.read_csv('earthquake_data_tsunami_cleaned.csv')
df.head()
X = df[['magnitude','cdi','mmi','sig','nst','dmin','gap','depth','latitude','longitude','Year','Month']]
y = df['tsunami']
X.reset_index(drop=True)
y.reset_index(drop=True)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 15)
knn.fit(X,y)

import pickle
with open('kn4_model.pkl','wb') as model_file:
    pickle.dump(knn,model_file)



