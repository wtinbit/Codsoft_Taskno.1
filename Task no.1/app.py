import streamlit as st
import pandas as pd 
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression

pickle_in = open('LogisticRegression.pkl','rb')
model = pickle.load(pickle_in)

st.title('Titanic Survival Prediction')


name = st.text_input("Name of the individual")
Pclass = st.number_input("Pclass:")    
Sex = st.number_input("Sex:")   
Age = st.number_input("Age:")
SibSp = st.number_input("SibSp:")  
Parch = st.number_input("Parch:")         
Fare = st.number_input("Fare:")      
Cabin = st.number_input("Cabin:")      
Embarked_Q = st.number_input("Embarked_Q:")  
Embarked_S = st.number_input("Embarked_S:")
button = st.button("predict")

if button:
    predictions = model.predict([[Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked_Q, Embarked_S]])
if predictions == 0:
    st.write("Sorry,", name,"has not survived")
else:
    st.write("Yes,", name,"has survived")