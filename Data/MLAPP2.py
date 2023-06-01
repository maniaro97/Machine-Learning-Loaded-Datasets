# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:25:49 2023

@author: manish
"""

import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import pycaret
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment

st.title("Machine Learning Application using Classification and Regression Models")

if os.path.exists("sourcev.csv"):
    df = pd.read_csv("sourcev.csv",index_col=None)

with st.sidebar:
    st.image("https://www.atriainnovation.com/wp-content/uploads/2021/02/portada-1080x675.jpg.webp")
    st.header("Welcome to the Application!")
    st.subheader("This is made for learning machine models. You can do both classification and regression analysis here.")
    st.caption("Choose your parameters below to work on the application.")
    choose=st.radio(":coffee:",['Dataset','Analysis','Training','Download'])
    st.info("I have made this application which helps in building automated machine learning models using streamlit, pandas, pandas_profiling(for EDA) and pycaret library. Hope ypu like it! :)")
    
if choose=="Dataset":
    st.write("Please upload your dataset here. Only .csv files allowed")
    Available_Datasets=[filename for filename in os.listdir()if filename.endswith('.csv')]
    selected_Datasets=st.selectbox('Select Datasets','Available Datasets')
    
    if selected_Datasets:
        df=pd.read_csv(selected_Datasets,index_col=None)
        st.dataframe(df)
        st.success('Dataset Suessfully Loaded')
    else:
        st.error('Error: No Dataset Avaialble')

if choose=="Analysis":
    st.write("Performing profiling on uploaded Dataset using pandas_profiling.")
    if st.sidebar.button("Do Analysis"):
        st.header('Perform Analysis on Data:')
        profile_report = df.profile_report() 
        st_profile_report(profile_report)
    
if choose=="Training":
    st.write("Start Training your Model now. Please choose classification or regression based on your model parameters.")
    choice = st.sidebar.selectbox("Select your Technique:", ["Classification","Regression"])
    target = st.selectbox("Select you Target Variable",df.columns)
    if choice=="Classification":
        if st.sidebar.button("Classification Train"):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_df = s1.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.sidebar.button("Regression Train"):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_df = s2.pull()
            st.info("The Setup data is as follows:")
            st.table(setup_df)
            
            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info("The Comparison of models is as folows:")
            st.table(compare_model)
            
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption("Download your model from here:")
        st.write("Note: the .pkl file will be download from here :)")
        st.download_button("Download the file",f,"Machine Learning model.pkl")