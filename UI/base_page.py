import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import time
import pandas as pd 

class Page():
    def __init__(self):
        print("Switching to page one")




    def user_input(self):
        age = st.slider('Age', 0, 130, 5)
        sex = st.selectbox(
            'Sex',
            ('Male','Female'))
        country = st.selectbox(
            'Country',
            ('Spain','France',"Germany"))
        
        tenure = st.slider('Tenure', 0, 20, 1)

        balance_eur = st.number_input('Balance in EUR')

        nb_product = st.slider('NB products', 0, 10, 1)

        has_credit_card = st.selectbox(
            'has credit card ?',
            ('Yes','No'))
        if has_credit_card == 'Yes' : 
            has_credit_card = 1 
        else :
            has_credit_card = 0
        
        is_active_member = st.selectbox(
            'Is active member ?',
            ('Yes','No'))


        if is_active_member == 'Yes' : 
            is_active_member = 1 
        else :
            is_active_member = 0
        
        credit_score = st.number_input('Credit Score')
        
        estimated_salary = st.number_input('Estimated Salary')

        feedback = st.text_area('Feedback', ''' Enter feedback ''')


        data = {
            'Age': age,
            'Gender': sex,
            'Country': country,
            'Tenure': tenure,
            'Balance (EUR)': balance_eur,
            'NumberOfProducts': nb_product,
            'HasCreditCard': has_credit_card,
            'IsActiveMember': is_active_member,
            'CreditScore': credit_score,
            'EstimatedSalary': estimated_salary,
            'CustomerFeedback': feedback
        }

        df = pd.DataFrame([data])
        return df 
 