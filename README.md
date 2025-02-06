# foliop-api


## See example API calls
### For max Sharpe ratio
curl -X POST http://localhost:5000/optimize \
     -H "Content-Type: application/json" \
     -d '{
           "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"],
           "strategy": "max_sharpe"
         }'

### For efficient risk with target volatility
curl -X POST http://localhost:5000/optimize \
     -H "Content-Type: application/json" \
     -d '{
           "tickers": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA", "TSLA"],
           "strategy": "efficient_risk",
           "target_volatility": 0.20
         }'
