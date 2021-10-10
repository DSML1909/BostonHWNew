# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 21:28:38 2021

@author: LocalAdmin
"""
import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"https://raw.githubusercontent.com/DSML1909/BostonHWNew/main/housing.csv")

section = st.sidebar.radio('Application Section', ['Data Explorer', 
                                                   'Model Explorer'])
if section == 'Data Explorer':
    st.title('Exploratory Data Analysis')
    st.write(df)
    
    #generating a heatmap to show correlation
    correlation_matrix1 = df.corr().round(2)
    # annot = True to print the values inside the square
    sns.heatmap(data=correlation_matrix1, annot=True)
    
    x_axis = st.sidebar.selectbox('Choose Your X-Axis Category', ['CRIM','AGE','RM', 'LSTAT'])
    
    y_axis = st.sidebar.selectbox('Choose Your Y-Axis Category', ['PRICE'])
    
    chart_type = st.sidebar.selectbox('Choose Your chart type', ['Table', 'Line','Bar','Strip'])
    @st.cache
    def create_grouping(x_axis, y_axis):
        grouping = df.groupby(x_axis)[y_axis].mean()
        return grouping
    
    grouping = create_grouping(x_axis, y_axis)
    
    st.title("Grouped data")
    
    if chart_type == 'Line':
        # make a line chart
        st.line_chart(grouping)
    elif chart_type == 'Bar':
        # make a bar chart
        st.bar_chart(grouping)
    elif chart_type == 'Table':
        # make a table
        st.write(grouping)
    else:
        st.plotly_chart(px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis))

else:

    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.write("""
    # Boston House Price Prediction App
    This app predicts the **Boston House Price**!
    """)
    st.write('---')
    
    # Loads the Boston House Price Dataset
    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    Y = pd.DataFrame(boston.target, columns=["MEDV"])
    
    # Sidebar
    # Header of Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')
    
    def user_input_features():
        CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
        ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
        INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
        CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
        NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
        RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
        AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
        DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
        RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
        TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
        PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
        B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
        LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
        data = {'CRIM': CRIM,
                'ZN': ZN,
                'INDUS': INDUS,
                'CHAS': CHAS,
                'NOX': NOX,
                'RM': RM,
                'AGE': AGE,
                'DIS': DIS,
                'RAD': RAD,
                'TAX': TAX,
                'PTRATIO': PTRATIO,
                'B': B,
                'LSTAT': LSTAT}
        features = pd.DataFrame(data, index=[0])
        return features
    
    df = user_input_features()
    
    # Main Panel
    
    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df)
    st.write('---')
    
    # Build Regression Model
    model = RandomForestRegressor()
    model.fit(X, Y)
    # Apply Model to Make Prediction
    prediction = model.predict(df)
    
    st.header('Prediction of Median House Price Value (MEDV)')
    st.write(prediction)
    st.write('---')