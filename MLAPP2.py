# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 20:25:49 2023

@author: manish
"""

import streamlit as st

page_bg_img = f"""
<style>
[data-testid = "stAppViewContainer"]{{
    background-image: url("https://img.rawpixel.com/s3fs-private/rawpixel_images/website_content/v546batch3-mynt-34-badgewatercolor_1.jpg?w=800&dpr=1&fit=default&crop=default&q=65&vib=3&con=3&usm=15&bg=F4F4F3&ixlib=js-2.2.1&s=89288ef4b47127f7f34a5998b50e4470");
    background-size: cover;
    opacity: 0.9;
    }}
[data-testid = "stSidebar"]{{
    background-color: #E3D3CE;
    opacity: 0.8;
    filter: blur(0.2px);
    }}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html = True)
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import matplotlib.pyplot as plt
import pycaret
from pycaret.classification import setup, compare_models, pull, save_model, ClassificationExperiment
from pycaret.regression import setup, compare_models, pull, save_model, RegressionExperiment

st.title(":orange[**_Machine Learning Application using Classification and Regression Models_**]")
logo='pycaret.png'

    
if os.path.exists("sourcev.csv"):
    df = pd.read_csv("sourcev.csv",index_col=None)

with st.sidebar:
    st.image("https://thedatascientist.com/wp-content/uploads/2019/02/automated_machine_learning-870x435.png")
    st.header(":violet[Welcome to the Application!]")
    st.subheader(":green[This is made for learning machine models. You can do both classification and regression analysis here.]")
    st.caption("**:red[Choose your parameters below to work on the application.]**")
    choose=st.radio(":coffee:",['Dataset','Analysis','Training','Download'])
    st.info("I have made this application which helps in building automated machine learning models using **:red[_streamlit, pandas, pandas_profiling(for EDA) and pycaret library._]** Hope ypu like it! :)")
    
if choose=="Dataset":
    st.write(":blue[_Please upload your dataset here. Only :red[.csv files] allowed_]")
    Available_Datasets=[filename for filename in os.listdir()if filename.endswith('.csv')]
    selected_Datasets=st.selectbox(':brown[Select Datasets]',Available_Datasets)
    
    if selected_Datasets:
        df=pd.read_csv(selected_Datasets,index_col=None)
        df.to_csv("sourcev.csv", index = None)
        st.dataframe(df)
        st.success('**_Dataset Successfully Loaded_**')
    else:
        st.error('**_Error: No Dataset Avaialble_**')

if choose=="Analysis":
    st.write(":green[**_Performing profiling on uploaded Dataset using pandas_profiling._**]")
    if st.button("Do Analysis"):
        st.header('Perform Analysis on Data:')
        profile_report = df.profile_report() 
        st_profile_report(profile_report)

if choose=="Training":
    st.write(":blue[**_Start Training your Model now. Please choose **:green[classification]** or **:red[regression]** based on your model parameters._**]")
    target = st.selectbox(":green[Select you Target Variable:]",df.columns)
    choice = st.selectbox(":blue[Select your Technique:]", ["Classification","Regression"])
    if choice=="Classification":
        if st.button("Classification Train"):
            s1 = ClassificationExperiment()
            s1.setup(data=df, target=target)
            setup_df = s1.pull()
            st.info(":red[**The Setup data is as follows:**]")
            st.table(setup_df)
            
            best_model1 = s1.compare_models()
            compare_model = s1.pull()
            st.info(":red[**The Comparison of models is as folows:**]")
            st.table(compare_model)
            
            best_model1
            s1.save_model(best_model1,"Machine Learning Model")
    else:
        if st.button("Regression Train"):
            s2 = RegressionExperiment()
            s2.setup(data=df, target=target)
            setup_df = s2.pull()
            st.info(":violet[**The Setup data is as follows:**]")
            st.table(setup_df)
            
            best_model2 = s2.compare_models()
            compare_model = s2.pull()
            st.info(":violet[**The Comparison of models is as folows:**]")
            st.table(compare_model)
            
            best_model2
            s2.save_model(best_model2,"Machine Learning Model")

if choose =="Download":
    with open("Machine Learning model.pkl",'rb') as f:
        st.caption(":violet[Download your model from here:]")
        st.write(":red[**_Note: the **:green[.pkl file]** will be download from here :_**]")
        st.download_button("Download the file",f,"Machine Learning model.pkl")
