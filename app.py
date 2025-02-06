from flask import Flask, request, jsonify
import yfinance as yf
from pypfopt import risk_models, expected_returns, EfficientFrontier, exceptions
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

app = Flask(__name__)

def fetch_historical_data(tickers, period="1y"):
    """
    Fetch historical data for given tickers using yfinance
    """
    data = pd.DataFrame()
    for ticker in tickers:
        try:
            stock_data = yf.download(ticker, period=period)
            data[ticker] = stock_data['Adj Close']
        except Exception as e:
            return None, f"Error fetching data for {ticker}: {str(e)}"
    return data, None

def optimize_portfolios(historical_data):
    """
    Run portfolio optimization and return results
    """
    try:
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(historical_data)
        S = risk_models.sample_cov(historical_data)

        # Filter out assets with near-zero variance
        non_zero_variance = S.columns[(S.var() > 1e-6)]
        mu = mu[non_zero_variance]
        S = S.loc[non_zero_variance, non_zero_variance]

        portfolios = {}
        metrics = {}

        # Define optimization methods
        methods = {
            "Max Sharpe Ratio": ("max_sharpe", {}),
            "Minimum Volatility": ("min_volatility", {}),
            "Target Volatility 20%": ("efficient_risk", {"target_volatility": 0.20})
        }

        # Run optimizations
        for name, (method, kwargs) in methods.items():
            try:
                ef = EfficientFrontier(mu, S, solver="ECOS")
                weights = getattr(ef, method)(**kwargs)
                cleaned_weights = ef.clean_weights()
                
                # Store portfolio weights
                portfolios[name] = {
                    k: round(v * 100, 2) 
                    for k, v in cleaned_weights.items() 
                    if v > 0
                }
                
                # Store performance metrics
                expected_return, volatility, sharpe = ef.portfolio_performance()
                metrics[name] = {
                    "expected_annual_return": round(expected_return * 100, 2),
                    "annual_volatility": round(volatility * 100, 2),
                    "sharpe_ratio": round(sharpe, 2)
                }
                
            except exceptions.OptimizationError as e:
                return None, f"Optimization error for {name}: {str(e)}"

        return {"portfolios": portfolios, "metrics": metrics}, None

    except Exception as e:
        return None, f"Error in optimization process: {str(e)}"

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Endpoint to perform portfolio optimization
    Expected JSON payload: {
        "tickers": ["AAPL", "GOOGL", ...],
        "period": "1y"  # optional, defaults to 1y
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'tickers' not in data:
            return jsonify({
                "error": "Missing required parameter: tickers"
            }), 400

        tickers = data['tickers']
        period = data.get('period', '1y')

        # Validate tickers
        if not isinstance(tickers, list) or not tickers:
            return jsonify({
                "error": "Tickers must be a non-empty list"
            }), 400

        # Fetch historical data
        historical_data, error = fetch_historical_data(tickers, period)
        if error:
            return jsonify({"error": error}), 400

        # Run optimization
        result, error = optimize_portfolios(historical_data)
        if error:
            return jsonify({"error": error}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
