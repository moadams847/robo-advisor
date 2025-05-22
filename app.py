import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# Streamlit config
st.set_page_config(layout="wide")
st.title("Robo Advisor by Enron Group")

st.subheader("Analyze your investment portfolio with our tool.")

# Sidebar Inputs
with st.sidebar:
    st.header("Portfolio Configuration")
    risk_level = st.select_slider("Risk Level", options=["low", "Moderate", "Considerable", "High"], value="Considerable")
    investment_objective = st.selectbox(
        "Investment Objective",
        ["Capital Preservation", "Balanced Growth", "Dynamic Growth"],
        index=1
    )
    portfolio_value = st.number_input("Portfolio Value ($)", min_value=1000, value=1000, step=1000)
    confidence_level = st.slider("Confidence Level for VaR", 0.90, 0.99, 0.95, 0.01)
    years = st.slider("Analysis Period (Years)", 1, 10, 3)
    
    placeholder = st.empty()
    placeholder.info("Deciding on your investment strategy...")
    
    st.write("This tool helps you analyze your investment portfolio based on your risk level and investment objective.")
    
    st.write(" Developed by **Mohammed Adams** ")
    
# Portfolio Strategy
tickers, weights = [], []

if risk_level == "low":
    investmentStrategy = "Fixed income"
    placeholder.success("Fixed income strategy selected.")
    tickers = ['EMB', 'VWOB', 'BNDX', 'AGG', 'TFI']
    weights = np.repeat(1/5, 5)

elif risk_level == "Moderate" and investment_objective == "Capital Preservation":
    investmentStrategy = "Income"
    placeholder.success("Income strategy selected.")
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.3, 0.3, 0.3, 0.05, 0.05]

elif risk_level == "Moderate" and investment_objective in ["Balanced Growth", "Dynamic Growth"]:
    investmentStrategy = "Balanced Growth"
    placeholder.success("Balanced Growth strategy selected.")
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.166, 0.166, 0.166, 0.25, 0.25]

elif risk_level == "Considerable" and investment_objective == "Balanced Growth":
    investmentStrategy = "Balanced Growth"
    placeholder.success("Balanced Growth strategy selected.")
    tickers = ['EMB', 'VWOB', 'BNDX', 'VEA', 'IEFA']
    weights = [0.166, 0.166, 0.166, 0.25, 0.25]

elif risk_level == "High" and investment_objective == "Dynamic Growth":
    investmentStrategy = "Growth"
    placeholder.success("Growth strategy selected.")
    tickers = ['VTI', 'VTV', 'VOE', 'VEA', 'IEFA']
    weights = np.repeat(1/5, 5)

# Proceed only if tickers and weights are set
if tickers and len(weights) > 0:
    
    # Dates
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365 * years)

    @st.cache_data
    def get_portfolio_data(tickers, start, end):
        prices = pd.DataFrame()
        names, expenses = [], []

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end)
                prices[ticker] = data['Close']
                info = yf.Ticker(ticker).info
                names.append(info.get('shortName', ticker))
                expenses.append(info.get('annualReportExpenseRatio', 0.0))
            except Exception as e:
                st.error(f"Error downloading {ticker}: {e}")
                prices[ticker] = np.nan
                names.append(ticker)
                expenses.append(0.0)
        
        return prices.dropna(axis=1), names, expenses

    prices, full_names, expense_ratios = get_portfolio_data(tickers, start_date, end_date)

    # Check if we have price data
    if not prices.empty:
        weights = np.array(weights[:len(prices.columns)])
        log_returns = np.log(prices / prices.shift(1)).dropna()
        portfolio_returns = (log_returns * weights).sum(axis=1)

        avg_daily_return = portfolio_returns.mean()
        annual_return = (1 + avg_daily_return) ** 252 - 1
        annual_variance = portfolio_returns.var() * 252
        dividend_amount = round(annual_variance * portfolio_value, 2)

        # Value at Risk
        day_window = 5
        rolling_returns = portfolio_returns.rolling(window=day_window).sum().dropna()
        var = -np.percentile(rolling_returns, 100 - confidence_level * 100) * portfolio_value

        # Display Outputs
        col1, col2 = st.columns(2)

        with col1:
            st.header("Portfolio Allocation")
            df_alloc = pd.DataFrame({
                "Ticker": prices.columns,
                "Name": full_names[:len(prices.columns)],
                "Weight": weights
            })
            st.dataframe(df_alloc.style.format({"Weight": "{:.0%}"}))

            fig1, ax1 = plt.subplots()
            ax1.pie(weights, labels=[f"{t}\n({n.split()[0]})" for t, n in zip(prices.columns, full_names)],
                    autopct='%1.1f%%', startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

        with col2:
            st.header("Performance Metrics")
            df_metrics = pd.DataFrame({
                "Metric": ["Annual Return", "Dividend Amount", "Annual Volatility", f"{day_window}-Day {int(confidence_level*100)}% VaR"],
                "Value": [
                    f"{annual_return*100:.2f}%",
                    f"${dividend_amount:,.2f}",
                    f"{np.sqrt(annual_variance)*100:.2f}%",
                    f"${var:,.2f}"
                ]
            })
            st.dataframe(df_metrics)

            fig2, ax2 = plt.subplots()
            portfolio_returns.plot(ax=ax2)
            ax2.set_title("Daily Returns")
            ax2.grid(True)
            st.pyplot(fig2)

        st.header(f"Distribution of {day_window}-Day Returns ($)")
        fig3, ax3 = plt.subplots()
        ax3.hist(rolling_returns * portfolio_value, bins=50, color="blue", alpha=0.7)
        ax3.axvline(-var, color="red", linestyle="--", label="VaR")
        ax3.set_xlabel("Return ($)")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        st.pyplot(fig3)

        # Show raw data
        if st.checkbox("Show raw data"):
            st.subheader("Adjusted Close Prices")
            st.dataframe(prices)

            st.subheader("Log Returns")
            st.dataframe(log_returns)

    else:
        st.error("No valid price data was fetched. Please try different tickers or a shorter time period.")
else:
    st.warning("Please select a valid combination of risk level and investment objective to generate a portfolio.")
