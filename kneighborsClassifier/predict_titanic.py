import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def predict(model, x):
    prediction = model.predict(x)
    proba = model.predict_proba(x)
    return prediction,proba

# Test data
test_data = pd.read_csv('./data/test.csv')
test_data = test_data[['Sex','Age','Pclass']]
test_data.dropna(axis=0, inplace=True)
test_data['Sex'].replace(['male', 'female'],[1,0],inplace=True)

# load the model from local disk
with open('titanic_model.pkl','rb') as f:
    model = pickle.load(f)


# final predicted 
prediction,proba= predict(model,test_data)
print(prediction)
print(proba)

