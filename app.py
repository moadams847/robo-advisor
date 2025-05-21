import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# Streamlit app configuration
st.set_page_config(layout="wide")
st.title("Portfolio Analysis Tool")

# Sidebar controls
with st.sidebar:
    st.header("Portfolio Configuration")
    risk_level = st.select_slider("Risk Level", options=[1, 2, 3, 4], value=3)
    
    investment_objective = st.selectbox(
        "Investment Objective",
        ["Capital Preservation", "Balanced Growth", "Dynamic Growth",  ],
        index=1
    )
    portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=1000, step=1000)
    confidence_level = st.slider("Confidence Level for VaR", 0.90, 0.99, 0.95, 0.01)
    years = st.slider("Analysis Period (Years)", 1, 10, 3)

# Asset allocation based on risk and objective
# Strategy = Fixed Income 1 (for any objective when risk_level == 1)
if risk_level == 1 and investment_objective in ("Capital Preservation", "Balanced Growth", "Dynamic Growth"):
    tickers = ['EMB', 'VWOB', 'BNDX', 'AGG', 'TFI']
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

#strategy = Income 2
elif risk_level == 2 and investment_objective == "Capital Preservation":
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.3, 0.3, 0.3, 0.05, 0.05]

#strategy = Balanced 3
elif risk_level == 2 and investment_objective == "Balanced Growth":
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.166, 0.166, 0.166, 0.25, 0.25]
    
#strategy = Balanced 3
elif risk_level == 2 and investment_objective == "Dynamic Growth":
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.166, 0.166, 0.166, 0.25, 0.25]
    

#strategy = Balanced 3
elif risk_level == 3 and investment_objective == "Balanced Growth":
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.166, 0.166, 0.166, 0.25, 0.25]
    
#strategy = Growth 4
elif risk_level == 4 and investment_objective == "Dynamic Growth":
    tickers = ['VTI', 'VTV', 'VOE', 'VEA', 'IEFA']
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    
# Date range calculation
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=365 * years)

# Download data and process
@st.cache_data
def get_portfolio_data(tickers, start_date, end_date):
    adjusted_close = pd.DataFrame()
    full_names = []
    expense_ratios = []
    
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adjusted_close[ticker] = data['Close']
        
        etf = yf.Ticker(ticker)
        info = etf.info
        full_names.append(info.get('shortName', ticker))
        expense_ratios.append(info.get('annualReportExpenseRatio', 0.0))
    
    return adjusted_close, full_names, expense_ratios

adjusted_close, full_names, expense_ratios = get_portfolio_data(tickers, start_date, end_date)

# Calculate returns
log_returns = np.log(adjusted_close / adjusted_close.shift(1)).dropna()
portfolio_returns = (log_returns * weights).sum(axis=1)

# Performance metrics
average_return = portfolio_returns.mean()
annualized_return = (1 + average_return) ** 252 - 1
historical_variance = portfolio_returns.var()
annualized_variance = historical_variance * 252
DividendAmount = round(annualized_variance * portfolio_value, 2)

# VaR calculation
day_window = 5
range_returns = portfolio_returns.rolling(window=day_window).sum().dropna()
var = -np.percentile(range_returns, 100 - (confidence_level * 100)) * portfolio_value

# Display results
col1, col2 = st.columns(2)

with col1:
    st.header("Portfolio Allocation")
    
    # Create DataFrame for display
    portfolio_df = pd.DataFrame({
        'Ticker': tickers,
        'Name': full_names,
        'Weight': weights,

    })
    
    st.dataframe(portfolio_df.style.format({'Weight': '{:.0%}'}))
    
    # Pie chart
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    labels = [f"{t}\n({n.split(' ')[0]})" for t, n in zip(tickers, full_names)]
    ax1.pie(weights, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#c2c2f0'])
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    st.header("Performance Metrics")
    
    metrics_df = pd.DataFrame({
        'Metric': ['Dividend Yield', 'Dividen dAmount', 'Annualized Volatility',  
                  f'{day_window}-day {confidence_level:.0%} VaR'],
        'Value': [f"{annualized_return*100:.2f}%", DividendAmount,
                 f"{np.sqrt(annualized_variance)*100:.2f}%",
                 f"${var:,.2f}"]
    })
    st.dataframe(metrics_df)
    
    # Returns plot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    portfolio_returns.plot(ax=ax2)
    ax2.set_title("Daily Returns")
    ax2.set_ylabel("Return")
    ax2.grid(True)
    ax2.grid(True)
    st.pyplot(fig2)
    
 
# VaR distribution plot
st.header(f"Distribution of Portfolio {day_window}-day Returns (Dollar Value)")
fig3, ax3 = plt.subplots(figsize=(10, 5))
range_returns_dollar = range_returns * portfolio_value
ax3.hist(range_returns_dollar, bins=50, alpha=0.7, color='blue')
ax3.axvline(-var, color='red', linestyle='dashed', linewidth=2)
ax3.set_xlabel("Returns ($)")
ax3.set_ylabel("Frequency")
ax3.grid(True)
st.pyplot(fig3)

# Display raw data if needed
if st.checkbox("Show raw data"):
    st.subheader("Adjusted Close Prices")
    st.dataframe(adjusted_close)
    
    st.subheader("Log Returns")
    st.dataframe(log_returns)