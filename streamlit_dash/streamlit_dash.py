import streamlit as st
import requests
import base64
import time
import altair as alt
import pandas as pd
from streamlit_autorefresh import st_autorefresh
import numpy as np
from utils import convertRGBtoHex


data_url = 'http://localhost:5000/observability_data'
uuid_url = 'http://localhost:5000/get_open_experiment'


st.set_page_config(
    page_title = 'Color matching dashboard',
    layout='wide'
)


@st.cache_data
def get_uuid():
    uniqueid = None
    while uniqueid is None:
        # if we start streamlit dash before campaign, want it to chill here 
        # until campaign gets going 
        r = requests.get(uuid_url)
        uniqueid = r.json()['uuid']
        time.sleep(5)
    
    return uniqueid

st.title('Color matching campaign monitoring')

placeholder = st.empty()

def rgb2hex(r,g,b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

while True:
    uniqueid = get_uuid()
    data_r = requests.post(data_url, json = {'uuid':uniqueid})
    data = data_r.json()

    image = data['image']
    
    image_jpeg = base64.b64decode(image.encode('ascii'))

    observed_rgb = data['observed_rgb']

    with placeholder.container():

        image_fig, chart_fig = st.columns(2)
        chart2_fig, chart3_fig = st.columns(2)

        image_fig.add_rows

        with image_fig:
            st.header('Most recent image')
            st.image(image_jpeg)

        with chart_fig:
            st.header('backtesting curve')

            
            print('most recent color')
            print(observed_rgb[-1])
            hex_codes = [convertRGBtoHex(rgb[0], rgb[1], rgb[2]) for rgb in observed_rgb]

            loss = data['model_prediction']['y_true']
            ind = np.arange(1, len(hex_codes)+1).tolist()

            backtest_data = pd.DataFrame({'trial_ind':ind, 'loss':loss, 'color':hex_codes})            


            backtest_curve = alt.Chart(
                backtest_data,
                width = 800,
                height = 600
            ).mark_circle(
                size = 100
            ).encode(
                x = 'trial_ind',
                y = 'loss',
                color = alt.value('color')
            )

            st.altair_chart(backtest_curve)


        with chart2_fig:
            st.header('GPR Model parity plot')
            y_pred = data['model_prediction']['y_pred']
            y_true = data['model_prediction']['y_true']
            parity_data = pd.DataFrame({'y_true':y_true, 'y_predicted':y_pred})

            alt_chart = alt.Chart(
                parity_data,
                width = 400, 
                height = 400
            ).mark_circle(
                size = 100
            ).encode(
                x = 'y_true',
                y = 'y_predicted'
            )

            st.altair_chart(alt_chart)

        with chart3_fig:
            st.header('Held for future use')


    time.sleep(10)






