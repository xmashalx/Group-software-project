#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 18:01:58 2025

@author: mashalchr
"""
import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive

st.title('Welcome: this is a mixed modelling web application')

#"""Loading the data"""

#"""creating a function for loading data"""
##reference(https://docs.streamlit.io/get-started/tutorials/create-an-app)
def load_data(uploaded_file, nrows=10000):
    data=None
    #loads data from an uploaded file or 
    #prints an error message if file not uploaded
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        return data
    else:
        st.write('please upload a valid dataset')
        return None
    
#"""working on creating a user interface for the user to upload their data"""
st.sidebar.header("Upload your dataset")
uploaded_file = st.sidebar.file_uploader("choose a csv file", type=["csv"])



##also creating tabs for user to naviagte to(tabs to be filled with code later)
tab1, tab2, tab3,tab4 = st.tabs(["Welcome", "your data", "Mixed Model","Diagnostics"])
with tab1:
    #this tab is for helping the user understand the functionality of the web app
    st.header('user guide')
    st.write('the user guide will go here')
    
    
    
data = load_data(uploaded_file) ## uploading the users data 
# this is falaknaaz.

##creating ui for user to select variables
#assigning to variable names to be used later 
#reference (https://medium.com/streamlit/multi-select-all-option-in-streamlit-3c92a0f20526)
if data is not None:
    column_names = data.columns.tolist() ##putting the column names into a list
    st.sidebar.write('Variable selection')
    target=st.sidebar.selectbox("Select Target", column_names)
    features=st.sidebar.multiselect("Select Features", column_names)
    random_vars = st.sidebar.selectbox("Select Random Variable", column_names)
    cov_structure_list=['diagonal','compound symmetry', 'Autoregressive']
    cov_structure=st.sidebar.selectbox('select covariance structure',cov_structure_list)
    random_effects_list=['random slopes','random intercept','random slopes and intercepts']
    random_effect=st.sidebar.selectbox("select random effect type", random_effects_list)
    
##creating a random effect variable to add different random effect types to the model
    if random_effect:
        if random_effect=='random slopes':
            re_formula='1'
        elif random_effect=='random intercept':
            re_formula=' + '.join(features)
        elif random_effect=='random slopes and intercepts':
            re_formula='1 + ' + ' + '.join(features)
        
    if cov_structure:
        if cov_structure=='diagonal':
            cov_struct=Independence()
        elif cov_structure=='compound symmetry':
            cov_struct=Exchangeable()
        elif cov_structure=='Autoregressive':
            cov_struct=Autoregressive()
        else:
            raise ValueError("Invalid covariance structure")


        


    
with tab2:
    #this tab is mainly for exploratory data analysis
    st.header('explore your data')
    #"""displaying a preview of the dataset"""
    if data is not None:
        st.subheader('Raw data')
        st.dataframe(data)
        
        ##using checboxes to allow user to opt in to viewing stats on their data
        show_summary = st.checkbox("calculate Summary Statistics") #ref(https://docs.streamlit.io/develop/concepts/design/buttons)
        if show_summary:
            st.write('summary statistics for your variables')
            st.dataframe(data.describe().style.format(precision=2))
        show_variableinfo = st.checkbox("show Information on variables: data types and missing values")
        if show_variableinfo:
            st.write('Variable types and missing values')
            st.dataframe(pd.DataFrame({ "Column": data.columns,"Data Type": data.dtypes, "Missing Values": data.isnull().sum(),"Unique Values": data.nunique()}))
         
         #using a button to run exploratory data analysis for if the user has chosen their variables   
        st.write('#### please ensure you have made initial variable selections in order to run exploratory data analysis')
        if random_vars and features and target:
            eda_button=st.button("Run Exploratory Data Analysis")
            if eda_button:
                
                #creating visual for dist of numeric variables in user dataset
                st.write("## Exploratory Data Analysis (EDA)")
                st.write("### Histograms of Numeric Columns")
                fig, ax = plt.subplots(figsize=(8, 4))
                data.hist(ax=ax)
                st.pyplot(fig)
                
                ##creating boxplot visuals for target and features grouped by random variable
                st.write(f"### Boxplots of {features} by {random_vars} ")
                for feature in features: #for mixed modelling i am expecting users to want to have multiple predictive features so i loop through all features to visualise spread across grouping variable
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=random_vars,y=feature, data=data, ax=ax)
                    st.pyplot(fig)
                st.write(f"### Boxplots of {target} by {random_vars}")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=random_vars, y=target, data=data, ax=ax)
                st.pyplot(fig)
                
                ##printing summary stats for target and features for each category of grouping variable
                st.write("### Group-Level Summary")
                group_summary = data.groupby(random_vars).agg(["mean", "std", "count"])
                st.dataframe(group_summary)
 
    else:
        st.warning('please upload a valid csv file') #if dataset not valid wont preveiew just error message#"""displaying a preview of the dataset"""
        
        
with tab3: ##this tab is going to be for fitting the model displaying the model results, and evaluating the fit of the model using model stats
    st.header('Fitting your chosen mixed model to your data')
    if data is not None:
        if random_vars and features and target:
            data[target] = pd.to_numeric(data[target], errors='coerce') #making sure target is numerical
            data[random_vars] = data[random_vars].astype('category') ##making sure random variable is categorical before modelling
            formula=f"{target} ~ {' + '.join(features)}" #setting the initial formula
            run_model=st.button("fit mixed model") #creating a button for model deployment 
            if run_model:
                try:
                    ##fitting a fixed effect model for comparison purposes
                    model_fixed=ols(f"{target} ~ {' + '.join(features)}", data=data).fit()
                
                    ##fitting a random effects model
                    model_random = MixedLM.from_formula(formula, data=data, groups=data[random_vars])
                    model_random_fitted = model_random.fit(re_formula=re_formula, cov_struct=cov_struct)
                    #model_random= MixedLM(formula, data, groups=data[random_vars],re_formula=re_formula,cov_struct=cov_struct).fit() #ref (https://www.geeksforgeeks.org/introduction-to-linear-mixed-effects-models/)
                    #st.write('Fixed effect model summary: ')
                    #st.write(model_fixed.summary())
                    #st.write(f'{random_effect} model summary')
                    
                    ##now working to display the output of the model in an interactive user freindly way
                    coeffs = model_random_fitted.params
                    p_values = model_random_fitted.pvalues
                    std_errs = model_random_fitted.bse
                    result_df = pd.DataFrame({
                       'Feature': coeffs.index,
                       'Coefficient': coeffs.values,
                       'Standard Error': std_errs.values,
                       'P-value': p_values.values
                    })
                    result_df['Significant'] = result_df['P-value'].apply(lambda x: 'Yes' if x < 0.05 else 'No') ##highlighting significant values for user
                    st.subheader('Model Coefficients and Statistics') 
                    st.dataframe(result_df, use_container_width=True)
                    #st.write(model_random_fitted.summary())
                    
                    
                    ##generate and display a formula for the users model
                    formula_parts = [f"{coeff:.2f}*{feature}" if coeff > 0 else f"({coeff:.2f})*{feature}" for coeff, feature in zip(coeffs.values, coeffs.index)]
                    formula_str = " + ".join(formula_parts[1:])
                    st.subheader('Generated Model Formula')
                    intercept = model_random_fitted.fe_params.get('Intercept', 0)
                    st.write(f"Predicted {target} ={intercept:.2f} + {formula_str}")
                    #st.write(f"Predicted {target} = {model_random_fitted.intercept:.2f} + {formula_str}")
                    #{intercept:.2f} + 
                    ##now creating a place for the user to evaluate model stats and compare to fixed effects model
                    fixed_aic = model_fixed.aic
                    fixed_bic = model_fixed.bic
                    fixed_log_likelihood = model_fixed.llf
                    random_aic=model_random_fitted.aic
                    random_bic=model_random_fitted.bic
                    random_log_likelihood=model_random_fitted.llf
                    comparison_df = pd.DataFrame({"Model": ["Fixed Effects", "Random Effects"],"AIC": [fixed_aic, random_aic],"BIC": [fixed_bic, random_bic],"Log-Likelihood": [fixed_log_likelihood, random_log_likelihood]})
                    
                    
                
                    st.subheader('Model Comparison: AIC, BIC, and Log-Likelihood')
                    st.dataframe(comparison_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Model fitting failed: {e}")


        
with tab4:
    st.header('Checking all model assumptions are valid')

    
    

    

