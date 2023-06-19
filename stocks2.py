import streamlit as st
import pandas as pd 
import numpy as np 
# USING YFINANCE TO SCRAPE DATA FROM THE INTERNET
import yfinance as yf
from sklearn.metrics import r2_score
from datetime import date
from plotly import graph_objs as go
from sklearn.model_selection import train_test_split
from prophet import Prophet
import plotly.express as px


st.set_page_config(layout='wide')

# SETTING THE TITLE OF THE PAGE
st.title("STOCK PREDICTION MODEL V2")


# SETTING THE START DATE AND END DATE TO LOAD THE DATA BETWEEEN THE TIME FRAME
START = '2018-04-10'
TODAY = date.today().strftime('%Y-%m-%d')

# MAKING A FUNCTION TO LOAD THE DATA BASED ON SELECTED STOCK BETWEEN THE TIME FRAME
def load_data(ticker):
    data = yf.download(ticker, start=START,end=TODAY)
    data.reset_index(inplace=True)
    return data

# DEFINING A FUNCTION FOR PLOTTING OPENING AND CLOSING PRICE OVER THE YEARS
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close')) 
    fig.layout.update (xaxis_rangeslider_visible=True) 
    st.plotly_chart (fig)

def plot_ema_ma():
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data2['Date'][-47:],y=data2['MA7'][-47:],name='MA7',line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=data['Date'][-40:],y=data['Close'][-40:], name='closing price',line=dict(color='lightblue')))
    fig2.add_trace(go.Scatter(x=data2['Date'][-47:],y=data2['EMA'][-47:], name='EMA',line=dict(color='green')))
    
    fig2.layout.update(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig2)



def prophet_plt():
  fig3 = go.Figure()
  fig3.add_scatter(x=data3['ds'][-40:],y=data3['Close'][-40:],mode='lines',name='Actual')
  fig3.add_scatter(x=prediction['ds'][-47:],y=prediction['trend'][-47:], mode='lines', name='Predicted')
  fig3.add_trace(px.scatter(prediction[-47:],x='ds',y='yhat_upper', opacity=0.2).data[0])
  fig3.add_trace(px.scatter(prediction[-47:],x='ds',y='yhat_lower', opacity=0.2).data[0])
  model.plot(prediction)
  fig3.layout.update(xaxis_rangeslider_visible=False)
  st.plotly_chart(fig3)

with st.container():
  col1, col2 = st.columns((1,1))

  with col1:
    st.write('**DISCLAIMER:** Stocks are one of the most unpredictive investments, If you are considering investing in stocks, you should do your research and speak with a financial advisor and not rely on this model solely.')

  with col2:
    # SELECTING THE STOCKS I WANT TO MONITOR 
    stocks = ('HINDUNILVR.NS','RELIANCE.NS','TCS.NS','INFY','HDB','AWL.NS','GRINFRA.NS','KPRMILL.NS','MENONBE.NS','TANLA.NS')
    selected_stocks = st.selectbox("Select stock name:", stocks)

    # USING THE FUNCTION TO LOAD DATA FOR THE SELECTED STOCK
    data = load_data(selected_stocks)

  with col1:  
    data2 = data
    # CREATING THE COLUMNS WITH ROLLING MEAN COLUMNS OF 3, 7, 12 DAY PERIODS
    data2['MA3'] = data2['Close'].rolling(window=3).mean()
    data2['MA12'] = data2['Close'].rolling(window=12).mean()
    data2['MA7'] = data2['Close'].rolling(window=7).mean()
    
    # CALCULATING EMA AND CREATING A COLUMN
    data3 = data2
    data3['EMA'] = data3['Close'].ewm(span=10, adjust=False).mean()

    # FORECASTING EMA & MA
    for i in range(7):
      data2 = data2.append({'Close': data2['MA7'].iloc[-1]}, ignore_index=True)
      data2['MA7'] = data2['Close'].rolling(7).mean()
      data2['EMA'] = data2['Close'].ewm(span=10, adjust=False).mean()
    
    last_date = data['Date'].max()
    next_7_days = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq='B')
    next_7_days_data = pd.DataFrame({'Date': next_7_days})

    data2.loc[len(data) : len(data) + 7, 'Date'] = next_7_days[:7]
    st.subheader('EMA & MA graph:')
    plot_ema_ma() # PLOTTING EMA MA GRAPH

    st.subheader('Historical data')
    plot_raw_data() # PLOTTING HISTORICAL DATA

    st.subheader('Raw data')
    st.write(data.tail(10))

  with col2:
    
    data3[['ds','y']] = data[['Date','Close']]
    model = Prophet(weekly_seasonality=False) # LOADING MODEL
    model.fit(data3) # FITTING MODEL
    future = model.make_future_dataframe(periods=7) # MAKING DATAFRAME WITH FUTURE VALUES
    prediction = model.predict(future) # PREDICTION
    

    st.subheader('Future predictions by Prophet')
    prophet_plt() # PLOTTING PROPHET PLOT


    st.write(prediction.tail(10))


