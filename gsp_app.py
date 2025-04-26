#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

col1, col2 = st.columns([1, 5])
col1.image('logo.png', width=150)
col2.markdown(
    '''
    <h2 style="line-height: 1.2; margin: 0; padding: 0; display: inline-block; font-size: 38px;">
        Welcome to Blendstat
    </h2>
    <p style="font-size: 28px; margin-top: 5px; font-weight: normal; color: #555;">
        This is a linear mixed modelling website
    </p>
    ''', unsafe_allow_html=True
)
#col2.title('Welcome to Blendstat: this is a mixed modelling web application')


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
    # This tab is for helping the user understand the functionality of the web app
    st.header('User Guide')
    
    # --- Overview ---
    st.subheader('Overview')
    st.write("""
    BlendStat is a mixed modelling web application designed to analyze hierarchical data using Linear Mixed Effects Models (LMMs). 
    Upload your dataset, configure models visually, and gain insights through interactive statistical analysis.
    """)
    
    # --- Key Concepts ---
    st.subheader('Key Concepts')
    with st.expander("What is a Linear Mixed Effects Model?"):
        st.write("""
        **LMMs** handle clustered/hierarchical data by combining:
        - **Fixed effects**: Population-level predictors (e.g., drug dosage for all patients)
        - **Random effects**: Group-specific variations (e.g., baseline differences across hospitals)
        """)
    
    with st.expander("What is Hierarchical Data?"):
        st.write("""
        Data with nested structure:
        - Students → Classrooms → Schools
        - Patients → Doctors → Hospitals
        - Repeated measurements → Individuals
        """)
    
    
    with st.expander("Variable Definitions"):
        st.markdown("""
        - **Target Variable**: Your outcome measure, for example you may be trying to measure blood preasure in patients 
        - **Fixed Predictors**: Variables with consistent effects across all groups (e.g. the age of a patient might affect patients blood pressure  
        consistently regardless of what hospital they go to)
        - **Grouping Variable**: Clustering factor, this is a variable that can group together the rest of the data into sections.  
        For example a column named hospital that includes 3 unique variable values, Hospital A, Hospital B, and Hospital C would be a grouping variable.  
        All patients would fall into one of the three hospitals. It is usefull to analyse this type of data as we can assess the outcome  
        of patients across different hospitals (i.e. assess outcomes across groups)
        - **Random effects**: Including random effects allows for outcomes to vary between groups (e.g. how does a specific drug dosage affect patients in hospitals differently)
        - **Random Intercepts**: Allow baseline outcomes to differ by group (irrespective of any predictor variable) (e.g. we might believe that the baseline blood pressure for patients varies according to the hospital)
        - **Random slopes**: Allow the relationship between a specific predictor and the oucomes to vary across groups (e.g., we might believe that the relationship between drug dosage and blood pressure differs across hospitals,  
        therefore when choosing to select random slopes to be a part of the model you must choose which variable you would like to have a random slope effect)
        **NOTE**: to allow for both random intercept and random slopes in you model you can tick both checkboxes, remember to select a variable to model random slopes
        """)
        
    
    # --- How to Use ---
    st.subheader('Step-by-Step Guide')
    
    # Part A - Uploading Data
    st.markdown("""
    ### A. Uploading Data
    1. **Supported Formats**: CSV/Excel files (<100MB)
    2. **Data Requirements**:
       - One row per observation
       - Columns for target, predictors, and grouping variables
       - Clean missing values beforehand
    3. **Pro Tip**: Start with a sample dataset: (https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv) to practice
    """)
    
    # Part B - Exploring Data
    st.markdown("""
    ### B. Exploring Your Data
    Use the *Data Exploration* tab to:
    - **Preview raw data** 
    - **Summary statistics**: Mean, SD, min/max
    - **Visualizations**:
      - Boxplots by group
      - Scatterplots for relationships
      - Histograms for distributions
    
    **Check for**: Outliers, skewed distributions, and group imbalances
    """)
    
    # Part C - Model Fitting
    st.markdown("""
    ### C. Fitting Your Model
    1. **Variable Selection** (Sidebar):
       - Target: Your outcome variable
       - Fixed Effects: 1+ predictors
       - Grouping Variable: Cluster identifier
       - Random Effects: Choose to allow at least one predictor to have a random effect across groups in your model  
       you can tick the type of random effect you want this variable to have: random intercept, random slope, or check both

    2. **Output Interpretation** Go to the model tab:
    #### 1️⃣ Fixed Effects (Population-Level)
    *What to look for:*
    - **Coefficients**  
      ▸ Positive value → Target variable increases with predictor  
      ▸ Negative value → Target variable decreases with predictor  
      ▸ Near zero → Little effect  
    - **P-values**  
      ▸ *p < 0.05* → Statistically significant (⭐)  
      ▸ *p > 0.05* → Not significant (may need removal)  
    - **Intercept**  
      Baseline prediction when all predictors = 0 (check if meaningful)
     
    #### 2️⃣ Random Effects (Group-Level) 
    - **Intercepts**  
      ▸ Positive → Group baseline > population average  
      ▸ Negative → Group baseline < average  
      ▸ Large magnitude → Big group differences  
    - **Slopes**  
      ▸ Positive → Predictor effect stronger in this group  
      ▸ Negative → Effect weaker/reversed in group  
    - **Variance Components**  
      ▸ High variance → Groups differ substantially  
      ▸ Low variance → Groups behave similarly
    #### 3️⃣ Model Comparison (If Available)
    **Guidelines:**  
    - Lower AIC/BIC → Better model  
    - Higher log-likelihood → Better fit  
    - Prefer simpler models unless complex one improves fit substantially
    """)
    
    # Part D - Diagnostics
    st.markdown("""
    ### D. Model Diagnostics
    Check model validity in the *Diagnostics* tab:
    Here you can check for the following linear mixed model assumptions:
    - linearity 
    - Homoscedasticity
    - normality of residuals 
    - normality of random effects 
    - multicollinearity
    There will be interpretation guidance in this tab to help understand if the model violates any assumptions. 
    """)
    # --- Help ---
    st.subheader('Need Help?')
    st.write("""
    contact support:
    ✉️ Email: 19343366@brookes.ac.uk
    """)


data = load_data(uploaded_file) ## uploading the users data


##creating ui for user to select variables
#assigning to variable names to be used later
#reference (https://medium.com/streamlit/multi-select-all-option-in-streamlit-3c92a0f20526)
if data is not None:
    column_names = data.columns.tolist() ##putting the column names into a list
    st.sidebar.write('Variable selection')
    target=st.sidebar.selectbox("Select Target variable", column_names)
    features=st.sidebar.multiselect("Select predictor variables (fixed effects)", column_names)
    group_vars = st.sidebar.selectbox("Select Grouping Variable", column_names)
    st.sidebar.write('customise random effects: ')
    is_rand_intercept=st.sidebar.checkbox('Include random intercepts')
    is_rand_slopes=st.sidebar.checkbox('Include random slopes')
    if is_rand_slopes:
        random_slopes=st.sidebar.multiselect("select whichvariable will allow for random slopes ", features)



##creating a random effect variable to add different random effect types to the model
    if is_rand_intercept:
        if is_rand_slopes and random_slopes:
            re_formula='1 + ' + ' + '.join(random_slopes)
        else:
            re_formula='1'
    else:
        if is_rand_slopes and random_slopes:
            re_formula= ' + '.join(random_slopes)
        else:
            st.warning('You must choose to include at least one type of random effect')








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
        if group_vars and features and target:
            eda_button=st.button("Run Exploratory Data Analysis")
            if eda_button:

                #creating visual for dist of numeric variables in user dataset
                st.write("## Exploratory Data Analysis (EDA)")
                st.write("### Histograms of Numeric Columns")
                fig, ax = plt.subplots(figsize=(8, 4))
                data.hist(ax=ax,color='#FF69B4')
                st.pyplot(fig)

                ##creating boxplot visuals for target and features grouped by random variable
                st.write(f"### Boxplots of {features} by {group_vars} ")
                for feature in features: #for mixed modelling i am expecting users to want to have multiple predictive features so i loop through all features to visualise spread across grouping variable
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.boxplot(x=group_vars,y=feature, data=data, ax=ax, palette="pastel")
                    st.pyplot(fig)
                st.write(f"### Boxplots of {target} by {group_vars}")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.boxplot(x=group_vars, y=target, data=data, ax=ax,palette="pastel")
                st.pyplot(fig)


                ##printing summary stats for target and features for each category of grouping variable
                st.write("### Group-Level Summary")
                group_summary = data.groupby(group_vars).agg(["mean", "std", "count"])
                st.dataframe(group_summary)

    else:
        st.warning('please upload a valid csv file') #if dataset not valid wont preveiew just error message#"""displaying a preview of the dataset"""

if 'model_random_fitted' not in st.session_state:
    st.session_state.model_random_fitted = None

with tab3: ##this tab is going to be for fitting the model displaying the model results, and evaluating the fit of the model using model stats
    st.header('Fitting your chosen mixed model to your data')
    if data is not None:
        if group_vars and features and target:
            data[target] = pd.to_numeric(data[target], errors='coerce') #making sure target is numerical
            data[group_vars] = data[group_vars].astype('category') ##making sure random variable is categorical before modelling
            formula=f"{target} ~ {' + '.join(features)}" #setting the initial formula
            run_model=st.button("fit mixed model") #creating a button for model deployment
            if run_model:
                try:
                    ##fitting a fixed effect model for comparison purposes
                    model_fixed=ols(f"{target} ~ {' + '.join(features)}", data=data).fit()

                    ##fitting a random effects model
                    model_random=smf.mixedlm(formula, data, groups=data[group_vars],re_formula=re_formula)
                    #model_random = MixedLM.from_formula(formula, data=data, groups=data[group_vars],re_formula=re_formula, cov_struct=cov_struct)
                    model_random_fitted = model_random.fit()
                    #model_random= MixedLM(formula, data, groups=data[random_vars],re_formula=re_formula,cov_struct=cov_struct).fit() #ref (https://www.geeksforgeeks.org/introduction-to-linear-mixed-effects-models/)
                    #st.write('Fixed effect model summary: ')
                    #st.write(model_fixed.summary())
                    #st.write(f'{random_effect} model summary')

                    st.session_state.model_random_fitted = model_random_fitted

                    ##now working to display the output of the model in an interactive user freindly way
                    coeffs = model_random_fitted.params
                    p_values = model_random_fitted.pvalues
                    std_errs = model_random_fitted.bse
                    result_df = pd.DataFrame({
                       'Feature': coeffs.index,
                       'Coefficient': coeffs.values,
                       'Standard Error': std_errs.values,
                       'P-value': p_values.values.round(5)
                    })


                    result_df['Significant'] = result_df['P-value'].apply(lambda x: 'Yes' if x < 0.05 else 'No') ##highlighting significant values for user
                    st.subheader('Model Coefficients and Statistics')
                    st.dataframe(result_df, use_container_width=True)
                    #st.write(model_random_fitted.summary())


                    ##generate and display a formula for the users model
                    formula_parts = [
                        f"{coeff:.2f}*{feature}" if coeff >= 0 else f"({coeff:.2f})*{feature}"
                        for feature, coeff in coeffs.items()
                        if feature != 'Intercept' and not np.isclose(coeff, 0) # Explicitly exclude intercept
                    ]
                    #formula_parts = [f"{coeff:.2f}*{feature}" if coeff > 0 else f"({coeff:.2f})*{feature}" for coeff, feature in zip(coeffs.values, coeffs.index)]
                    formula_str = " + ".join(formula_parts)
                    st.subheader('Generated Model Formula')
                    intercept = model_random_fitted.params.get('Intercept', 0)
                    st.write(f"Predicted {target} ={intercept:.2f} + {formula_str}")
                    #st.write(f"Predicted {target} = {model_random_fitted.intercept:.2f} + {formula_str}")
                    #{intercept:.2f} +

                    ## creating a group specific summary table
                    random_effects_df = pd.DataFrame(model_random_fitted.random_effects).T.reset_index()
                    # Conditional column renaming
                    if re_formula == "1":
                        random_effects_df = random_effects_df.rename(
                            columns={"index": group_vars, "Group": "random_intercept"}
                        )
                    elif re_formula=='1 + ' + ' + '.join(random_slopes):
                        rename_dict = {"index": group_vars, "Group": "random_intercept"}
                        for slope in random_slopes:
                            rename_dict[slope] = f"random_slope_{slope}"
                        random_effects_df = random_effects_df.rename(columns=rename_dict)
                    elif re_formula==' + '.join(random_slopes):
                       rename_dict = {"index": group_vars}
                       for slope in random_slopes:
                           rename_dict[slope] = f"random_slope_{slope}"
                       random_effects_df = random_effects_df.rename(columns=rename_dict)



                    st.subheader("Random Effects by Group")
                    st.dataframe(
                        random_effects_df.style.format(
                            {col: "{:.2f}" for col in random_effects_df.select_dtypes(include=['float64']).columns}
                        ),
                        use_container_width=True
                    )




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
    #creating checkboxes for user to select each assumption check themselves
    lin=st.checkbox('Check model for linearity and homosceadicity')
    norm_resid=st.checkbox('check model for normality of residuals')
    norm_rand=st.checkbox('check model for normality of random effects')
    multi=st.checkbox('check model for no multicollinearity')


    ##the following functions were designed and inspired by the functions in the following reference: https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/
    def lin_and_hom_check(model):
        fitted_values = model.fittedvalues
        residuals = model.resid
        st.write("### Linearity and Homoscedasticity Check:")

        # Plot residuals vs fitted values
        fig, ax = plt.subplots()
        ax.scatter(fitted_values, residuals, alpha=0.7)
        ax.axhline(0, color='red', linestyle='--')
        ax.set_xlabel('Fitted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Linearity and Homoscedasticity Check: Residuals vs Fitted')

        # Display plot in Streamlit
        st.pyplot(fig)

        # Interpretation guidance
        st.write("### Interpretation:")
        st.write("- If No clear pattern it indicates **linearity** is likely satisfied.")
        st.write("- If Even spread of residuals it suggests **homoscedasticity** (constant variance).")
        st.write("- A funnel shape (wider or narrower) suggests a possible **violation** of homoscedasticity.")

    def norm_resid_check(model):
        # Get residuals
        residuals = model.resid

        # Create Q-Q plot
        fig, ax = plt.subplots()
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Normality Check: Q-Q Plot of Residuals")

        # Display plot in Streamlit
        st.pyplot(fig)

        # Interpretation guidance
        st.write("### Interpretation:")
        st.write("- If residuals follow a straight diagonal line, normality is satisfied.")
        st.write("- Deviations from the line suggest potential non-normality.")
        st.write("- Mild deviations are usually acceptable, but severe deviations may indicate issues.")

    def multicollinearity_check(model,data,features):
                # Ensure features are in the dataset
        predictors = data[features]

        # Add constant (intercept) to predictors for VIF calculation
        predictors = sm.add_constant(predictors)

        # Calculate VIF for each selected predictor
        vif_data = pd.DataFrame()
        vif_data["Variable"] = predictors.columns
        vif_data["VIF"] = [variance_inflation_factor(predictors.values, i) for i in range(predictors.shape[1])]

        # Display the VIF table in Streamlit as an interactive dataframe
        st.write("### Variance Inflation Factor (VIF) Check:")
        st.dataframe(vif_data)  # Display interactive VIF table

        # Interpretation guidance
        st.write("### Interpretation:")
        st.write("- A VIF greater than 5 (or 10) suggests high multicollinearity.")
        st.write("- High multicollinearity can make it difficult to assess the individual effect of predictors.")
        st.write("- Consider removing or combining correlated predictors to resolve this issue.")


    def norm_random_effects_check(model):
        # Get random effects from the model
        random_effects = pd.Series([val[0] for val in model.random_effects.values()])

        # Create Q-Q plot for random effects
        fig, ax = plt.subplots()
        stats.probplot(random_effects, dist="norm", plot=ax)
        ax.set_title("Normality Check: Q-Q Plot of Random Effects")

        # Display plot in Streamlit
        st.pyplot(fig)

        # Interpretation guidance
        st.write("### Interpretation:")
        st.write("- If random effects follow a straight diagonal line, normality is satisfied.")
        st.write("- Significant deviations from the line suggest that random effects may not be normally distributed.")
        st.write("- Minor deviations are acceptable, but large deviations could signal problems with model fit.")

    ##running assumptions if user has ticked the checkboxes
    if st.session_state.model_random_fitted:
        # Only perform checks if the model is fitted and stored in session state
        if lin:
            lin_and_hom_check(st.session_state.model_random_fitted)

        if norm_resid:
            norm_resid_check(st.session_state.model_random_fitted)

        if norm_rand:
            norm_random_effects_check(st.session_state.model_random_fitted)

        if multi:
            multicollinearity_check(st.session_state.model_random_fitted, data, features)
    else:
        st.write("Please fit the model first.")







