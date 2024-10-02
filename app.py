import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import sklearn
import dill
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error , mean_absolute_error

path_model = "model.pkl"
path_preprocessor = "preprocessor.pkl"
train_data_path = "Data/train.csv"
test_data_path = "Data/test.csv"

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

with open(path_preprocessor, 'rb') as file_p:
    preprocessor = dill.load(file_p)


with open(path_model, 'rb') as file_m:
    model = dill.load(file_m)


st.write(
    """
    # Flight Price Prediction app
    
    """
)
st.sidebar.header('User Input Features')

def user_input_features():
    airline = st.sidebar.selectbox('Airline',tuple(train_data['airline'].unique()))
    
    date_of_journey = st.sidebar.date_input("Enter date of journey",value=datetime.now().date())
    date_ = date_of_journey.strftime("%Y-%m-%d") # type: ignore
    
    source = st.sidebar.selectbox('Source',tuple(train_data['source'].unique()))
    destination = st.sidebar.selectbox('Destination',tuple(train_data['destination'].unique()))
    dep_time = st.sidebar.time_input("Enter time of departure").strftime("%H:%M:%S")
    arrival_time = st.sidebar.time_input("Enter time of arrival").strftime("%H:%M:%S")
    duration = st.sidebar.number_input("Enter Duration",value=90)
    total_stops = st.sidebar.number_input("Enter total_stops",value=1)
    additional_info = st.sidebar.selectbox('additional_info',tuple(train_data['additional_info'].unique()))
    data = {
        'airline':airline,
        'date_of_journey':date_,
        'source':source,
        'destination':destination,
        'dep_time':dep_time,
        'arrival_time':arrival_time,
        'duration':duration,
        'total_stops':total_stops,
        'additional_info':additional_info
    }
    
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

if st.sidebar.button('Make Prediction'):
    st.write("Input Data :")
    test_X = preprocessor.transform(input_df)
    prediction = model.predict(test_X)[0]
    st.dataframe(input_df)
    st.metric('Predicted Flight Price :',f"{round(prediction,2)}")
    
    test_X = preprocessor.transform(test_data)
    prediction = model.predict(test_X)


    r2 = r2_score(test_data['price'],prediction)
    st.metric("r2_score",round(r2,4))  # type: ignore
    
    x_max = np.max(prediction)
    x_min = np.min(prediction)
    fig,ax = plt.subplots(
        figsize=(7,3),
    )
    st.write('''
             ### Residual Plot
             ''')
    ax.scatter(
        test_data['price'],test_data['price']-prediction,
        c='limegreen', marker='s',
        edgecolor='white'
    )
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
    ax.set_ylabel("Residual")
    ax.set_xlabel("Predicted Values")
    plt.tight_layout()
    st.pyplot(fig)
        
                    
    