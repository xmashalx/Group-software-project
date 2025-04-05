# Linear Mixed Modelling Web application

A user-friendly web application designed to help users **fit and analyse Linear Mixed Effect Models (LMEMs)** using their own datasets — all without needing to write any code.

This tool is especially helpful for:
- Students and researchers new to statistical modelling
- Users with little to no programming experience
- Quick, exploratory analysis on custom datasets

---

## 🌟 Features

- 📁 Upload your own `.csv` dataset
- 🧮 Fit linear mixed models with fixed and random effects
- 📊 View model summaries and statistical output
- 🎛️ Intuitive interface – no coding required

---
## Built with 
This web application was built using the following major frameworks and libraries:

***Streamlit*** - For building the web app interface and making it interactive.

***NumPy*** - A package for numerical computations, used for handling data.

***Pandas*** - A powerful data manipulation library used to handle and process data.

***Matplotlib & Seaborn*** - For data visualization and plotting.

***Statsmodels*** - Used for statistical modeling, including Linear Mixed Models.

***Scipy*** - For statistical functions and support.

**<u>Other Tools:</u>**

***Python*** - The primary programming language for this project.

***GitHub*** - For version control and repository hosting.

## 🚀 1. Getting Started

### 1.1 Install Python 
make sure python is installed in your computer You can download it from:
👉 https://www.python.org/downloads/ 

### 1.2 Download the project 
You can get the project files by cloning the GitHub repository:
1)Open a terminal or command prompt
2)Run the following command : git clone https://github.com/xmashalx/Group-software-project.git
3) navigate into the project folder: cd Group-software-project

### 1.3 Create a virtual environemnt (optional but recomended)
***On Windows***
run the following commands:

python -m venv env

env\Scripts\activate

***On macOS/Linux***
run the following commands:

python3 -m venv env

source env/bin/activate

### 1.4 Install the required packages
run the following command: pip install -r requirements.txt

### 1.5 Run the application 
run the following command: streamlit run gsp_app.py

### 1.6 Using the web application 
To test the Linear Mixed Modelling Web Application, you’ll need a dataset to upload. 
For demonstration purposes, you can use the sleepstudy.csv dataset — a commonly used dataset in linear mixed-effects modeling.
You can download the dataset from the following link: https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv
Once you have the dataset, upload it to the web application when prompted, select your variables, and start fitting your model.
There is a user guide on the website to help you if you get stuck 


## 2. dataset requirements
If you are using your own dataset please read the following to undertand requirements for the dataset
### 2.1 Required Format 
File Type: CSV file (e.g., yourdata.csv).
Columns:
- Dependent Variable (Outcome): This is the variable you are trying to predict or explain (e.g., reaction_time, sleep_duration).
- Independent Variables: These can be continuous or categorical predictors (e.g., age, group).
- Subject ID (Grouping Factor): A unique identifier for each subject or unit in your data (e.g., subject_id). This is important for the mixed model, as it accounts for random effects.

### 2.2 Data Types
- Numerical Columns: Ensure that continuous variables (e.g., age, reaction_time) are in numeric format.
- Categorical Columns: Ensure that categorical variables (e.g., group) are in text format (no numbers or special characters).
- Missing Data: Try to avoid missing data in your dataset. If necessary, you can remove rows with missing values or impute them prior to uploading.

## 3. Acknowledgements 
The developers of this project are:

Mashal Hussain - 19343366@brookes.ac.uk

FalaKnaaz Khan - 19335056@brookes.ac.uk

Mukul Katkar - 19310486@brookes.ac.uk

Vaibhav Lakhani - 19326141@brookes.ac.uk

Additionally, we would like to thank our Product Owner and Project Supervisor: 

Dr.Eleni Elia – For their guidance, feedback, and role as the project owner.
For helping us navigate and fulfill the project requirements. Their insights were invaluable throughout the development process.
