import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

# data set
titanic = pd.read_csv('./data/train.csv')
titanic = titanic[['Sex','Age','Survived','Pclass']]
titanic.dropna(inplace=True, axis=0)
titanic['Sex'].replace(['male','female'], [1,0],inplace=True)
print(titanic.describe())
print(titanic.head())

y = titanic['Survived']
X = titanic.drop('Survived', axis=1)
print(y)
print(X)
model = KNeighborsClassifier(3)
# train
model.fit(X,y)
score = model.score(X,y)

# Brute force technique: don't use it this technique it is very silly!!!!
def find_optimal_k():
    best_score = 0
    K_MAX = 20
    K_best = 2
    for k in range(2,K_MAX):
        # model constructor
        model = KNeighborsClassifier(k)
        # train
        model.fit(X,y)
        score = model.score(X,y)
        
        if score>best_score:
            best_score = score
            K_best = k
        print(f"k best is: {K_best}")
        print(f"Best score is: {best_score}")




# Save the model in local
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model,f)
#model evaluate
model.score(X,y)

def survie(model, sex=1, age=22, pclass=3):
    x = np.array([sex,age,pclass]).reshape(1,3)
    predictions = model.predict(x)
    return predictions


pred = survie(model)

# END
