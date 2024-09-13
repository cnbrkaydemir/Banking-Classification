import streamlit as st
import numpy as np
from utils import *


jobs = {"admin":0, "blue-collar":1, "entrepreneur":2, "housemaid":3, "management":4, "retired":5,"self-employed":6, "services":7, "technician":8, "unemployed":9, "Unknown":10}


marital_status = {"divorced":0, "maried":1, "single":2, "unknown":3}

education_status = {"Basic 4 years old":0,"Basic 6 years old":1, "Basic 9 years old":2, "Highschool":3, "Illiterate":4, "Professional Course":5, "University Degree":6, "Unknown":7}

default_dict = {"no":0, "unknown":1, "yes":2}

housing_dict = {"no":0, "unknown":1, "yes":2}

loan_dict = {"no":0, "unknown":1, "yes":2}

contact_dict = {"cellular":0, "telephone":1}

month_dict = {"January":0, "February":1, "March":2, "April":3, "May":4, "June":5, "July":6, "August":7, "September":8, "October":9, "November":10, "December":11 }

day_dict = {"Friday":0 ,"Monday":1, "Thursday":2, "Tuesday":3, "Wednesday":4}

poutcome_dict = {"failure":0, "nonexistant":1, "success":2}



st.title("Banking Classification")

st.write("Fill in the form with customer information to predict if they will subscribe to term deposit.")

result = 0

with st.form("banking_marketing"):

    age = st.number_input('Age')

    job = st.selectbox(
     "Select Job",
     (jobs.keys()),
        
    )

    maritial = st.selectbox(
     "Maritial Status",
     (marital_status.keys())   
    )

    education = st.selectbox(
        "Education Status",
        (education_status.keys())
    )


    default = st.selectbox(
        "Has credit in default ?",
        (default_dict.keys())   
    )

    housing = st.selectbox(
        "Has housing loan ?",
        (housing_dict.keys())   
    )

    loan = st.selectbox(
        "Has personal loan ?",
        (loan_dict.keys())   
    )

    contact = st.selectbox(
        "Method of contact ?",
        (contact_dict.keys())
    )

    month = st.selectbox(
        "Last contact month of the year ?",
        (month_dict.keys())
    )

    day = st.selectbox(
        "Last contact day ?",
        (day_dict.keys())
    )

    duration = st.number_input('Total duration of the last contact with this client ?')

    campaign = st.number_input('Number of contacts performed during this campaign and for this client ?')

    p_days = st.number_input('Number of days that passed by after the client was last contacted from a previous campaign ?')

    previous = st.number_input('Number of contacts performed before this campaign and for this client ?')

    p_out = st.selectbox(
        "Outcome of the previous marketing campaign ?",
        (poutcome_dict.keys())
    )

    emp_var_rate = st.number_input('Employee variance rate for this client ?')

    cons_price_idx = st.number_input('Consumer price index ?')

    cons_conf_idx = st.number_input('Consumer confidence index ?')
    
    euribor = st.number_input('Euribor 3 month rate ?')
    
    nr_employed = st.number_input('Number of employees ?')

    totalLoan = loan_dict[loan] + housing_dict[housing] 


    submit_form = st.form_submit_button(label="Predict", help="Click to predict!")
    
    if submit_form:
        result = predict([np.log1p(age), jobs[job], marital_status[maritial], education_status[education], default_dict[default],
                              housing_dict[housing], loan_dict[loan], contact_dict[contact], month_dict[month], day_dict[day],
                              np.log1p(duration), np.log1p(campaign), p_days, np.log1p(previous), poutcome_dict[p_out], emp_var_rate, cons_price_idx, cons_conf_idx,
                                euribor,nr_employed ** 2, totalLoan
                              ])

        if result > 0.5:
            st.markdown("<h4>Positive: Looks like the customer will subscribe to the term deposit.<h4/>", unsafe_allow_html=True)

        else:
            st.markdown("<h4>Negative: Unfortunately the customer will likely not subscribe to the term deposit.</h4>", unsafe_allow_html=True)
    