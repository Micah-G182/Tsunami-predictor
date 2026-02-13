import pandas as pd

df = pd.read_csv('Training.csv')
df.head()
df.duplicated()
df.drop_duplicates(inplace=True)
df.shape
df.to_csv("Training_cleaned_data.csv")
