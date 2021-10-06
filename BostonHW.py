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

df = pd.read_csv(r"C:\Users\LocalAdmin\OneDrive\Documents\GitHub\dat-07-26\Homework\Unit3\data\housing.csv")

section = st.sidebar.radio('Application Section', ['Data Explorer', 
                                                   'Model Explorer'])
if section == 'Data Explorer':
    st.title('Exploratory Data Analysis')
    st.write(df.head())
    
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
        st.write("""
        # Boston House Price Prediction App
        This app predicts the **Boston House Price**!
        """)
        st.write('---')

        # Loads the Boston House Price Dataset
        boston = datasets.load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.DataFrame(boston.target, columns=["PRICE"])

        # Sidebar
        # Header of Specify Input Parameters
        st.sidebar.header('Specify Input Parameters')

        def user_input_features():
            CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
            ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
            INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
            CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
            NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
            RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
            AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
            DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
            RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
            TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
            PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
            B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
            LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
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

        st.header('Prediction of PRICE')
        st.write(prediction)
        st.write('---')