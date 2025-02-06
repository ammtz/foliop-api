from flask import Flask, request, jsonify
from pypfopt import risk_models, expected_returns, EfficientFrontier, exceptions
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Configuration
PICKLE_PATH = os.getenv('PICKLE_PATH', 'historical_data.pkl')

def get_historical_data(tickers, period="5y"):
    """
    Get historical data from pickle file for the specified tickers
    """
    try:
        # Load the full dataset
        df = pd.read_pickle(PICKLE_PATH)
        
        # Validate that all requested tickers are in the dataset
        missing_tickers = set(tickers) - set(df.columns)
        if missing_tickers:
            return None, f"Missing data for tickers: {', '.join(missing_tickers)}"
        
        # Filter for requested tickers
        df_filtered = df[tickers]
        
        # Filter for requested time period
        if period.endswith('y'):
            years = int(period.replace('y', ''))
            start_date = df_filtered.index[-1] - pd.DateOffset(years=years)
            df_filtered = df_filtered[df_filtered.index >= start_date]
        
        return df_filtered, None

    except FileNotFoundError:
        return None, f"Historical data file not found at {PICKLE_PATH}"
    except Exception as e:
        return None, f"Error loading historical data: {str(e)}"

def optimize_portfolio(historical_data, strategy, target_volatility=None):
    """
    Run portfolio optimization for a specific strategy and return results
    """
    try:
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(historical_data)
        S = risk_models.sample_cov(historical_data)

        # Filter out assets with near-zero variance
        non_zero_variance = S.columns[(S.var() > 1e-6)]
        mu = mu[non_zero_variance]
        S = S.loc[non_zero_variance, non_zero_variance]

        # Initialize EfficientFrontier
        ef = EfficientFrontier(mu, S, solver="ECOS")

        # Apply the requested strategy
        if strategy == "max_sharpe":
            weights = ef.max_sharpe()
        elif strategy == "min_volatility":
            weights = ef.min_volatility()
        elif strategy == "efficient_risk":
            if target_volatility is None:
                return None, "Target volatility is required for efficient_risk strategy"
            weights = ef.efficient_risk(target_volatility=target_volatility)
        else:
            return None, f"Invalid strategy: {strategy}"

        # Clean weights and get performance metrics
        cleaned_weights = ef.clean_weights()
        expected_return, volatility, sharpe = ef.portfolio_performance()

        result = {
            "weights": {
                k: round(v * 100, 2) 
                for k, v in cleaned_weights.items() 
                if v > 0
            },
            "metrics": {
                "expected_annual_return": round(expected_return * 100, 2),
                "annual_volatility": round(volatility * 100, 2),
                "sharpe_ratio": round(sharpe, 2)
            }
        }

        return result, None

    except exceptions.OptimizationError as e:
        return None, f"Optimization error: {str(e)}"
    except Exception as e:
        return None, f"Error in optimization process: {str(e)}"

@app.route('/optimize', methods=['POST'])
def optimize():
    """
    Endpoint to perform portfolio optimization using stored historical data
    """
    try:
        data = request.get_json()
        
        # Validate required parameters
        if not data or 'tickers' not in data or 'strategy' not in data:
            return jsonify({
                "error": "Missing required parameters: tickers and strategy required"
            }), 400

        tickers = data['tickers']
        strategy = data['strategy']
        period = data.get('period', '1y')
        target_volatility = data.get('target_volatility')

        # Validate tickers
        if not isinstance(tickers, list) or not tickers:
            return jsonify({
                "error": "Tickers must be a non-empty list"
            }), 400

        # Validate strategy
        valid_strategies = ["max_sharpe", "min_volatility", "efficient_risk"]
        if strategy not in valid_strategies:
            return jsonify({
                "error": f"Invalid strategy. Must be one of: {', '.join(valid_strategies)}"
            }), 400

        # Validate target_volatility for efficient_risk strategy
        if strategy == "efficient_risk":
            if target_volatility is None:
                return jsonify({
                    "error": "target_volatility is required for efficient_risk strategy"
                }), 400
            if not isinstance(target_volatility, (int, float)) or target_volatility <= 0:
                return jsonify({
                    "error": "target_volatility must be a positive number"
                }), 400

        # Fetch historical data from pickle file
        historical_data, error = get_historical_data(tickers, period)
        if error:
            return jsonify({"error": error}), 400

        # Run optimization
        result, error = optimize_portfolio(historical_data, strategy, target_volatility)
        if error:
            return jsonify({"error": error}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
