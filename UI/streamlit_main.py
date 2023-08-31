import streamlit as st
import os
from base_page import Page

import pandas as pd 
import requests
import shap
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt 

PAGES = {
    "Prediction": 1,

}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]


if page == 1:
    try:
        first_page=Page()
        df = first_page.user_input()
        button_clicked = st.button("Send Request")
        if button_clicked :
            data = {'file': ('data.csv', df.to_csv(index=False), 'text/csv')}
            url = 'http://127.0.0.1:5000/predict'
            response = requests.post(url, files=data).json()
            data = response.get("data")
            names = response.get("data_name")
            new_df = pd.DataFrame(data,names)
            shap_values = np.array(response.get("shap_values"))
            predicted_case = '***The user is leaving the bank***' if response.get("predictions")[0] == 1 else '***The user is staying in the bank***'
            st.markdown(predicted_case)
            st.title("Shap based interpretability of the classifier")
            plt.figure()
            shap.summary_plot(np.array(response.get("shap_values")), data , feature_names=names  , plot_type="bar")
            st.pyplot(plt)
            st.title("Integrated Gradient based interpretation of the roberta model")
            st.markdown(response.get("interpretations")[0], unsafe_allow_html=True)
    except Exception as e :
        print(e)
        st.write("La requête ne renvoies rien, merci de vérifier les paramètres")
