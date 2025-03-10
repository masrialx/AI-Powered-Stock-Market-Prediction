from flask import Flask, jsonify, request
import yfinance as yf
from src.database import db, Stock, StockPrice, EarningsCalendar, NewsSentiment, Prediction, Trade, Selection, init_db
from src.model import fetch_stock_data, fetch_earnings_calendar, fetch_news, preprocess_data, grok_api_predict, score_stock, select_top_stocks, narrow_to_top_5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

init_db(app)

# Helper function to calculate advanced metrics
def calculate_performance_metrics(df):
    if df.empty or len(df) < 2:
        return None
    
    daily_returns = df['Close'].pct_change().dropna()
    price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    momentum = price_change / len(df)  # Average daily momentum
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    return {
        "price_change": round(price_change, 2),
        "volatility": round(volatility, 4),
        "momentum": round(momentum, 4),
        "sharpe_ratio": round(sharpe_ratio, 4)
    }

# Step 1: Data Collection & Preprocessing with Fresh Data
@app.route("/collect", methods=["POST"])
def collect_data():
    if not request.is_json:
        return jsonify({"error": "Request must have Content-Type 'application/json'"}), 415
    
    data = request.get_json()
    tickers = data.get("tickers", [])
    force_refresh = data.get("force_refresh", False)
    
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400
    
    try:
        unique_tickers = set(tickers)
        warnings = []
        
        for ticker in unique_tickers:
            ticker = ticker.upper()
            if not ticker.isalnum():
                print(f"Skipping invalid ticker: {ticker}")
                continue
            
            stock = yf.Ticker(ticker)
            info = stock.info
            stock_entry = Stock.query.filter_by(ticker=ticker).first()
            last_updated = stock_entry.last_updated if stock_entry else None
            
            if not stock_entry:
                stock_entry = Stock(ticker=ticker, name=info.get("longName", ""), 
                                   sector=info.get("sector", ""), exchange=info.get("exchange", ""))
                db.session.add(stock_entry)
                db.session.commit()
                print(f"Added new stock: {ticker}")
            
            df = fetch_stock_data(ticker, since=last_updated if not force_refresh else None)
            if not df.empty:
                df = preprocess_data(df)
                for _, row in df.iterrows():
                    date_str = row['Date'].strftime('%Y-%m-%d')
                    existing = StockPrice.query.filter_by(stock_id=stock_entry.stock_id, date=date_str).first()
                    if not existing:
                        price = StockPrice(stock_id=stock_entry.stock_id, 
                                          date=date_str,
                                          open=row['Open'], high=row['High'], 
                                          low=row['Low'], close=row['Close'], 
                                          volume=int(row['Volume']))
                        db.session.add(price)
                        print(f"Added price data for {ticker} on {date_str}")
            
            earnings = fetch_earnings_calendar(ticker, since=last_updated if not force_refresh else None)
            for e in earnings:
                if e['earnings_date']:
                    existing = EarningsCalendar.query.filter_by(stock_id=stock_entry.stock_id, earnings_date=e['earnings_date']).first()
                    if not existing:
                        earnings_entry = EarningsCalendar(stock_id=stock_entry.stock_id, 
                                                         earnings_date=e['earnings_date'],
                                                         estimated_eps=e['estimated_eps'],
                                                         actual_eps=e['actual_eps'])
                        db.session.add(earnings_entry)
                        print(f"Added earnings for {ticker} on {e['earnings_date']}")
            
            news_data = fetch_news(ticker, since=last_updated if not force_refresh else None)
            if not news_data and last_updated is None:
                warnings.append(f"No news data retrieved for {ticker}")
            for news in news_data:
                existing = NewsSentiment.query.filter_by(stock_id=stock_entry.stock_id, source=news['source'], date=news['date']).first()
                if not existing:
                    news_entry = NewsSentiment(stock_id=stock_entry.stock_id, 
                                              source=news['source'], date=news['date'], 
                                              sentiment_score=news['sentiment_score'])
                    db.session.add(news_entry)
                    print(f"Added news for {ticker} from {news['source']} on {news['date']}")
            
            db.session.commit()
            print(f"Committed data for {ticker}")
        
        response = {"message": f"Data collected for {len(unique_tickers)} tickers"}
        if warnings:
            response["warnings"] = warnings
        return jsonify(response), 201
    except Exception as e:
        db.session.rollback()
        print(f"Error during data collection: {str(e)}")
        return jsonify({"error": f"Data collection failed: {str(e)}"}), 500

# Step 2 & 3: Select Top 10 and Narrow to Top 5
@app.route("/select", methods=["POST"])
def select_stocks():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400
    
    try:
        since = None if force_refresh else (datetime.utcnow() - timedelta(days=1))
        top_stocks_with_scores = select_top_stocks(tickers, since=since)
        if not top_stocks_with_scores:
            return jsonify({"error": "No valid stocks found"}), 404
        
        top_10 = [ticker for ticker, _ in top_stocks_with_scores]
        top_5 = narrow_to_top_5(top_stocks_with_scores)
        
        for ticker in top_10:
            selection = Selection(user_id=user_id, ticker=ticker, is_top_5=(ticker in top_5))
            db.session.add(selection)
        db.session.commit()
        print(f"Stored selections for user {user_id}: top_10={top_10}, top_5={top_5}")
        
        return jsonify({"top_10": top_10, "top_5": top_5}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Stock selection failed: {str(e)}"}), 500

# Enhanced Endpoint: Evaluate Selections with Powerful Calculations and Grok AI
@app.route("/evaluate_selections", methods=["POST"])
def evaluate_selections():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400
    
    try:
        # Step 1: Collect fresh data
        collect_response = app.test_client().post("/collect", json={"tickers": tickers, "force_refresh": force_refresh})
        if collect_response.status_code != 201:
            return collect_response.get_json(), collect_response.status_code
        
        # Step 2: Select top stocks
        select_response = app.test_client().post("/select", json={"tickers": tickers, "user_id": user_id, "force_refresh": force_refresh})
        if select_response.status_code != 200:
            return select_response.get_json(), select_response.status_code
        
        top_10_tickers = select_response.get_json()["top_10"]
        top_5_tickers = select_response.get_json()["top_5"]
        
        # Step 3: Simulate 2-day hold with powerful calculations
        performance = {}
        for ticker in top_10_tickers:
            df = fetch_stock_data(ticker, period="3d")  # Last 3 days to simulate 2-day hold
            metrics = calculate_performance_metrics(df)
            if metrics:
                performance[ticker] = {
                    "initial_price": float(df['Close'].iloc[0]),
                    "final_price": float(df['Close'].iloc[-1]),
                    "price_change": metrics["price_change"],
                    "volatility": metrics["volatility"],
                    "momentum": metrics["momentum"],
                    "sharpe_ratio": metrics["sharpe_ratio"]
                }
            else:
                performance[ticker] = {"error": "Insufficient data"}
        
        # Calculate market health score
        top_5_performance = {k: v for k, v in performance.items() if k in top_5_tickers and "error" not in v}
        if top_5_performance:
            market_health_score = (
                0.4 * np.mean([p["price_change"] for p in top_5_performance.values()]) / 100 +  # 40% weight
                0.3 * np.mean([p["sharpe_ratio"] for p in top_5_performance.values()]) +         # 30% weight
                0.2 * np.mean([p["momentum"] for p in top_5_performance.values()]) -            # 20% weight
                0.1 * np.mean([p["volatility"] for p in top_5_performance.values()])            # 10% penalty for volatility
            )
            market_going_well = market_health_score > 0  # Boolean value
        else:
            market_health_score = 0
            market_going_well = False  # Boolean value
        
        # Step 4: Grok AI predictions and market analysis
        predictions = {}
        market_analysis = ""
        for ticker in top_5_tickers:
            df = fetch_stock_data(ticker, period="3d")
            if not df.empty:
                news_data = fetch_news(ticker, since=datetime.utcnow() - timedelta(days=2))
                earnings_data = fetch_earnings_calendar(ticker, since=datetime.utcnow() - timedelta(days=2))
                latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].tolist()
                
                # Grok prediction
                pred, conf, expl = grok_api_predict(ticker, latest_data, news_data, earnings_data)
                predictions[ticker] = {
                    "prediction": pred,
                    "confidence": round(conf * 100, 2),
                    "explanation": expl
                }
                
                # Grok market analysis (simulated for now, replace with real API call)
                performance_data = performance.get(ticker, {})
                grok_payload = f"""
                Analyze the market for {ticker} based on:
                - Price Change: {performance_data.get('price_change', 'N/A')}%
                - Volatility: {performance_data.get('volatility', 'N/A')}
                - Momentum: {performance_data.get('momentum', 'N/A')}
                - Sharpe Ratio: {performance_data.get('sharpe_ratio', 'N/A')}
                - Recent News Sentiment: {sum(n['sentiment_score'] for n in news_data) / len(news_data) if news_data else 'N/A'}
                - Earnings Data: {earnings_data if earnings_data else 'N/A'}
                Provide a concise market outlook.
                """
                market_analysis += f"{ticker}: Simulated Grok analysis - Market outlook is {'positive' if pred == 'Buy' else 'cautious' if pred == 'Hold' else 'negative'} based on {performance_data.get('price_change', 'N/A')}% price change, sentiment, and trends.\n"
        
        # Prepare result
        result = {
            "user_id": user_id,
            "evaluation_date": datetime.utcnow().isoformat(),
            "top_10_performance": performance,
            "top_5_predictions": predictions,
            "market_health_score": round(market_health_score, 4),
            "market_going_well": bool(market_going_well),  # Explicitly ensure it’s a boolean
            "market_analysis": market_analysis.strip()
        }
        
        # Debug: Print the result to inspect its contents
        print("Result before jsonify:", result)
        
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500
    
    
# Step 4: AI-Based Prediction with Grok API
@app.route("/predict/<ticker>", methods=["GET"])
def predict_stock(ticker):
    ticker = ticker.upper()
    force_refresh = request.args.get("force_refresh", "false").lower() == "true"
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        last_updated = stock.last_updated if not force_refresh else None
        df = fetch_stock_data(ticker, since=last_updated)
        df = preprocess_data(df)
        if df.empty:
            return jsonify({"error": f"No new data for {ticker} since last update"}), 404
        
        news_data = fetch_news(ticker, since=last_updated)
        earnings_data = fetch_earnings_calendar(ticker, since=last_updated)
        latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume', 'SMA10', 'RSI', 'MACD', 'BB_upper', 'BB_lower']].tolist()
        
        prediction, confidence, explanation = grok_api_predict(ticker, latest_data, news_data, earnings_data)
        existing_pred = Prediction.query.filter_by(stock_id=stock.stock_id, date=df.iloc[-1]['Date'].strftime('%Y-%m-%d')).first()
        if not existing_pred:
            pred_entry = Prediction(stock_id=stock.stock_id, 
                                   date=df.iloc[-1]['Date'].strftime('%Y-%m-%d'),
                                   predicted_movement=prediction, 
                                   confidence_score=confidence,
                                   explanation=explanation)
            db.session.add(pred_entry)
        else:
            existing_pred.predicted_movement = prediction
            existing_pred.confidence_score = confidence
            existing_pred.explanation = explanation
        db.session.commit()
        
        return jsonify({
            "ticker": ticker,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "explanation": explanation,
            "source": "Grok API"
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Step 5: Monitoring
@app.route("/monitor/<ticker>", methods=["GET"])
def monitor_stock(ticker):
    ticker = ticker.upper()
    days = request.args.get("days", 2, type=int)
    
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        pred = Prediction.query.filter_by(stock_id=stock.stock_id).order_by(Prediction.date.desc()).first()
        if not pred:
            return jsonify({"error": "No prediction found"}), 404
        
        df = fetch_stock_data(ticker, period=f"{days}d")
        if df.empty:
            return jsonify({"error": f"No recent data for {ticker}"}), 404
        
        latest_close = df.iloc[-1]['Close']
        pred_close = StockPrice.query.filter_by(stock_id=stock.stock_id, date=pred.date).first()
        if not pred_close:
            return jsonify({"error": f"No price data for prediction date {pred.date}"}), 404
        
        price_change = ((latest_close - pred_close.close) / pred_close.close) * 100
        status = "Correct" if (pred.predicted_movement == "Buy" and price_change > 0) or \
                             (pred.predicted_movement == "Sell" and price_change < 0) else "Incorrect"
        
        return jsonify({
            "ticker": ticker,
            "prediction": pred.predicted_movement,
            "price_change": round(price_change, 2),
            "status": status
        }), 200
    except Exception as e:
        return jsonify({"error": f"Monitoring failed: {str(e)}"}), 500

# Step 6: Trading Alerts
@app.route("/trade/<ticker>", methods=["POST"])
def execute_trade(ticker):
    ticker = ticker.upper()
    data = request.get_json()
    user_id = data.get("user_id")
    action = data.get("action")
    
    if not user_id or not isinstance(user_id, int) or not action or action not in ["Buy", "Sell"]:
        return jsonify({"error": "Valid user_id and action (Buy/Sell) required"}), 400
    
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        df = fetch_stock_data(ticker, period="1d")
        if df.empty:
            return jsonify({"error": f"No current price data for {ticker}"}), 404
        current_price = df.iloc[-1]['Close']
        
        pred = Prediction.query.filter_by(stock_id=stock.stock_id).order_by(Prediction.date.desc()).first()
        if pred:
            pred_date = datetime.strptime(pred.date, '%Y-%m-%d')
            if (datetime.now() - pred_date).days <= 5:
                pred_close = StockPrice.query.filter_by(stock_id=stock.stock_id, date=pred.date).first()
                if pred_close:
                    price_change = ((current_price - pred_close.close) / pred_close.close) * 100
                    if price_change >= 5:
                        action = "Sell"
                    elif price_change <= -3:
                        action = "Sell"
        
        trade = Trade(user_id=user_id, stock_id=stock.stock_id, action=action, 
                     price=current_price, date=datetime.now().strftime('%Y-%m-%d'))
        db.session.add(trade)
        db.session.commit()
        
        alert = "Profit target hit" if price_change >= 5 else "Stop-loss triggered" if price_change <= -3 else "Manual trade"
        return jsonify({
            "message": f"{action} executed for {ticker}",
            "ticker": ticker,
            "action": action,
            "price": current_price,
            "alert": alert
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Trade failed: {str(e)}"}), 500

# Full Automation with Grok API
@app.route("/run", methods=["POST"])
def run_full_process():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400
    
    try:
        collect_response = app.test_client().post("/collect", json={"tickers": tickers, "force_refresh": force_refresh})
        if collect_response.status_code != 201:
            return collect_response.get_json(), collect_response.status_code
        
        select_response = app.test_client().post("/select", json={"tickers": tickers, "user_id": user_id, "force_refresh": force_refresh})
        if select_response.status_code != 200:
            return select_response.get_json(), select_response.status_code
        top_5 = select_response.get_json()["top_5"]
        if not top_5:
            return jsonify({"error": "No top 5 stocks selected"}), 404
        
        predictions = {}
        for ticker in top_5:
            pred_response = app.test_client().get(f"/predict/{ticker}?force_refresh={force_refresh}")
            if pred_response.status_code == 200:
                predictions[ticker] = pred_response.get_json()
        
        trades = {}
        for ticker in top_5:
            monitor_response = app.test_client().get(f"/monitor/{ticker}")
            if monitor_response.status_code == 200:
                monitor_data = monitor_response.get_json()
                if monitor_data["prediction"] in ["Buy", "Sell"]:
                    trade_response = app.test_client().post(f"/trade/{ticker}", 
                                                           json={"user_id": user_id, "action": monitor_data["prediction"]})
                    if trade_response.status_code == 201:
                        trades[ticker] = trade_response.get_json()
        
        return jsonify({
            "message": "Full process completed with Grok API",
            "top_5": top_5,
            "predictions": predictions,
            "trades": trades
        }), 200
    except Exception as e:
        return jsonify({"error": f"Full process failed: {str(e)}"}), 500

# Existing Endpoint: Get all stocks from database
@app.route("/stocks", methods=["GET"])
def get_all_stocks():
    try:
        stocks = Stock.query.all()
        if not stocks:
            return jsonify({"message": "No stocks found in database"}), 404
        
        result = [
            {
                "ticker": stock.ticker,
                "name": stock.name,
                "sector": stock.sector,
                "exchange": stock.exchange,
                "last_updated": stock.last_updated.isoformat()
            } for stock in stocks
        ]
        return jsonify({"stocks": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve stocks: {str(e)}"}), 500

# New Endpoint: Get All Selected Stocks Data
@app.route("/selected_stocks", methods=["GET"])
def get_all_selected_stocks():
    try:
        tickers = request.args.get("tickers")
        if tickers:
            tickers = tickers.split(",")
        else:
            stocks = Stock.query.all()
            if not stocks:
                return jsonify({"message": "No stocks found in database"}), 404
            tickers = [stock.ticker for stock in stocks]
        
        since = datetime.utcnow() - timedelta(days=30)
        top_stocks_with_scores = select_top_stocks(tickers, since=since)
        if not top_stocks_with_scores:
            return jsonify({"error": "No valid stocks selected"}), 404
        
        top_10_tickers = [ticker for ticker, _ in top_stocks_with_scores]
        top_5_tickers = narrow_to_top_5(top_stocks_with_scores)
        
        top_10_details = []
        for ticker in top_10_tickers:
            result, error = score_stock(ticker, since=since)
            if result:
                top_10_details.append({
                    "ticker": result["ticker"],
                    "composite_score": result["composite_score"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "momentum": result["momentum"],
                    "grok_score": result["grok_score"],
                    "news_sentiment": result["news_sentiment"],
                    "x_sentiment": result["x_sentiment"],
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "explanation": result["explanation"]
                })
        
        top_5_details = [stock for stock in top_10_details if stock["ticker"] in top_5_tickers]
        
        return jsonify({
            "top_10": top_10_details,
            "top_5": top_5_details,
            "count": len(top_10_details)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve selected stocks: {str(e)}"}), 500

# New Endpoint: Get Specific Selected Stock Data
@app.route("/selected_stocks/<ticker>", methods=["GET"])
def get_selected_stock(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found in database"}), 404
        
        stocks = Stock.query.all()
        tickers = [s.ticker for s in stocks]
        if not tickers:
            return jsonify({"error": "No stocks available for selection"}), 404
        
        since = datetime.utcnow() - timedelta(days=30)
        top_stocks_with_scores = select_top_stocks(tickers, since=since)
        if not top_stocks_with_scores:
            return jsonify({"error": "No valid stocks selected"}), 404
        
        top_10_tickers = [t for t, _ in top_stocks_with_scores]
        if ticker not in top_10_tickers:
            return jsonify({"error": f"Stock {ticker} not in selected top stocks"}), 404
        
        result, error = score_stock(ticker, since=since)
        if result:
            return jsonify({
                "ticker": result["ticker"],
                "composite_score": result["composite_score"],
                "sharpe_ratio": result["sharpe_ratio"],
                "momentum": result["momentum"],
                "grok_score": result["grok_score"],
                "news_sentiment": result["news_sentiment"],
                "x_sentiment": result["x_sentiment"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "explanation": result["explanation"]
            }), 200
        else:
            return jsonify({"error": error}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve selected stock {ticker}: {str(e)}"}), 500

# Existing Endpoints (unchanged)
@app.route("/stocks/<ticker>", methods=["GET"])
def get_stock(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        since = datetime.utcnow() - timedelta(days=30)
        result, error = score_stock(ticker, since=since)
        if result:
            return jsonify({
                "ticker": result["ticker"],
                "composite_score": result["composite_score"],
                "sharpe_ratio": result["sharpe_ratio"],
                "momentum": result["momentum"],
                "grok_score": result["grok_score"],
                "news_sentiment": result["news_sentiment"],
                "x_sentiment": result["x_sentiment"],
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "explanation": result["explanation"]
            }), 200
        else:
            return jsonify({"error": error}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to fetch stock {ticker}: {str(e)}"}), 500

@app.route("/stock_prices/<ticker>", methods=["GET"])
def get_stock_prices(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        prices = StockPrice.query.filter_by(stock_id=stock.stock_id).order_by(StockPrice.date.asc()).all()
        if not prices:
            return jsonify({"message": f"No price data found for {ticker}"}), 404
        
        result = [
            {
                "date": price.date,
                "open": price.open,
                "high": price.high,
                "low": price.low,
                "close": price.close,
                "volume": price.volume,
                "last_updated": price.last_updated.isoformat()
            } for price in prices
        ]
        return jsonify({"ticker": ticker, "prices": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve stock prices for {ticker}: {str(e)}"}), 500

@app.route("/earnings/<ticker>", methods=["GET"])
def get_earnings(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        earnings = EarningsCalendar.query.filter_by(stock_id=stock.stock_id).order_by(EarningsCalendar.earnings_date.asc()).all()
        if not earnings:
            return jsonify({"message": f"No earnings data found for {ticker}"}), 404
        
        result = [
            {
                "earnings_date": e.earnings_date,
                "estimated_eps": e.estimated_eps,
                "actual_eps": e.actual_eps,
                "last_updated": e.last_updated.isoformat()
            } for e in earnings
        ]
        return jsonify({"ticker": ticker, "earnings": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve earnings for {ticker}: {str(e)}"}), 500

@app.route("/news/<ticker>", methods=["GET"])
def get_news(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        news = NewsSentiment.query.filter_by(stock_id=stock.stock_id).order_by(NewsSentiment.date.asc()).all()
        if not news:
            return jsonify({"ticker": ticker, "news": [], "count": 0, "message": f"No news data found for {ticker}"}), 200
        
        result = [
            {
                "source": n.source,
                "date": n.date,
                "sentiment_score": n.sentiment_score,
                "last_updated": n.last_updated.isoformat()
            } for n in news
        ]
        return jsonify({"ticker": ticker, "news": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve news for {ticker}: {str(e)}"}), 500

@app.route("/data/<ticker>", methods=["GET"])
def get_all_data(ticker):
    ticker = ticker.upper()
    try:
        stock = Stock.query.filter_by(ticker=ticker).first()
        if not stock:
            return jsonify({"error": f"Stock {ticker} not found"}), 404
        
        stock_info = {
            "ticker": stock.ticker,
            "name": stock.name,
            "sector": stock.sector,
            "exchange": stock.exchange,
            "last_updated": stock.last_updated.isoformat()
        }
        
        prices = StockPrice.query.filter_by(stock_id=stock.stock_id).order_by(StockPrice.date.asc()).all()
        prices_data = [
            {"date": p.date, "open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume, "last_updated": p.last_updated.isoformat()}
            for p in prices
        ]
        
        earnings = EarningsCalendar.query.filter_by(stock_id=stock.stock_id).order_by(EarningsCalendar.earnings_date.asc()).all()
        earnings_data = [
            {"earnings_date": e.earnings_date, "estimated_eps": e.estimated_eps, "actual_eps": e.actual_eps, "last_updated": e.last_updated.isoformat()}
            for e in earnings
        ]
        
        news = NewsSentiment.query.filter_by(stock_id=stock.stock_id).order_by(NewsSentiment.date.asc()).all()
        news_data = [
            {"source": n.source, "date": n.date, "sentiment_score": n.sentiment_score, "last_updated": n.last_updated.isoformat()}
            for n in news
        ]
        
        return jsonify({
            "stock": stock_info,
            "prices": prices_data,
            "earnings": earnings_data,
            "news": news_data
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve data for {ticker}: {str(e)}"}), 500

@app.route("/all_news", methods=["GET"])
def get_all_news():
    try:
        news = db.session.query(NewsSentiment, Stock.ticker).join(Stock, NewsSentiment.stock_id == Stock.stock_id).order_by(NewsSentiment.date.asc()).all()
        if not news:
            return jsonify({"news": [], "count": 0, "message": "No news data found in database"}), 200
        
        result = [
            {
                "ticker": stock_ticker,
                "source": n.source,
                "date": n.date,
                "sentiment_score": n.sentiment_score,
                "last_updated": n.last_updated.isoformat()
            } for n, stock_ticker in news
        ]
        return jsonify({"news": result, "count": len(result)}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to retrieve all news: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)