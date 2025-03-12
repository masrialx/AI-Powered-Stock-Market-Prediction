from flask import Flask, jsonify, request
from flasgger import Swagger, swag_from  # Import Flasgger for Swagger
import yfinance as yf
from src.database import db, Stock, StockPrice, EarningsCalendar, NewsSentiment, Prediction, Trade, Selection, init_db
from src.model import fetch_stock_data, fetch_earnings_calendar, fetch_news, preprocess_data, grok_api_predict, score_stock, select_top_stocks, narrow_to_top_5
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Swagger
swagger = Swagger(app, template={
    "swagger": "2.0",
    "info": {
        "title": "Smart Stock Picker API",
        "description": "API for collecting stock data, selecting top stocks, evaluating predictions, monitoring performance, and executing trades using Grok AI.",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["http"],
    "consumes": ["application/json"],
    "produces": ["application/json"]
})

# Initialize database
init_db(app)

# Helper function for performance metrics
def calculate_performance_metrics(df):
    if df.empty or len(df) < 2:
        return None
    daily_returns = df['Close'].pct_change().dropna()
    price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100
    volatility = daily_returns.std() * np.sqrt(252)
    momentum = price_change / len(df)
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() != 0 else 0
    return {
        "price_change": round(price_change, 2),
        "volatility": round(volatility, 4),
        "momentum": round(momentum, 4),
        "sharpe_ratio": round(sharpe_ratio, 4)
    }

# Step 1: Data Collection
@app.route("/collect", methods=["POST"])
@swag_from({
    'tags': ['Data Collection'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'tickers': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of stock tickers (e.g., ["AAPL", "TSLA"])'},
                    'force_refresh': {'type': 'boolean', 'default': False, 'description': 'Force refresh of data'}
                },
                'required': ['tickers']
            }
        }
    ],
    'responses': {
        '201': {
            'description': 'Data collected successfully',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'warnings': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        },
        '400': {'description': 'Invalid input'},
        '415': {'description': 'Request must be JSON'},
        '500': {'description': 'Server error'}
    }
})
def collect_data():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415
    data = request.get_json()
    tickers = data.get("tickers", [])
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400

    try:
        unique_tickers = set(tickers)
        warnings = []
        with app.app_context():
            for ticker in unique_tickers:
                ticker = ticker.upper()
                if not ticker.isalnum():
                    warnings.append(f"Invalid ticker: {ticker}")
                    continue

                stock = Stock.query.filter_by(ticker=ticker).first()
                last_updated = stock.last_updated if stock and not force_refresh else None

                yf_stock = yf.Ticker(ticker)
                info = yf_stock.info
                if not stock:
                    stock = Stock(ticker=ticker, name=info.get("longName", ""), 
                                  sector=info.get("sector", ""), exchange=info.get("exchange", ""))
                    db.session.add(stock)
                stock.last_updated = datetime.utcnow()
                db.session.commit()

                df = fetch_stock_data(ticker, since=last_updated)
                if not df.empty:
                    df = preprocess_data(df)
                    for _, row in df.iterrows():
                        date_str = row['Date'].strftime('%Y-%m-%d')
                        if not StockPrice.query.filter_by(stock_id=stock.stock_id, date=date_str).first():
                            price = StockPrice(stock_id=stock.stock_id, date=date_str, open=row['Open'], 
                                               high=row['High'], low=row['Low'], close=row['Close'], volume=int(row['Volume']))
                            db.session.add(price)

                earnings = fetch_earnings_calendar(ticker, since=last_updated)
                for e in earnings:
                    if e['earnings_date'] and not EarningsCalendar.query.filter_by(stock_id=stock.stock_id, earnings_date=e['earnings_date']).first():
                        earnings_entry = EarningsCalendar(stock_id=stock.stock_id, earnings_date=e['earnings_date'], 
                                                          estimated_eps=e['estimated_eps'], actual_eps=e['actual_eps'])
                        db.session.add(earnings_entry)

                news_data = fetch_news(ticker, since=last_updated)
                if not news_data and not last_updated:
                    warnings.append(f"No news data for {ticker}")
                for news in news_data:
                    if not NewsSentiment.query.filter_by(stock_id=stock.stock_id, source=news['source'], date=news['date']).first():
                        news_entry = NewsSentiment(stock_id=stock.stock_id, source=news['source'], date=news['date'], 
                                                   sentiment_score=news['sentiment_score'])
                        db.session.add(news_entry)

                db.session.commit()
                print(f"Data collected for {ticker}")

        return jsonify({"message": f"Data collected for {len(unique_tickers)} tickers", "warnings": warnings}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Data collection failed: {str(e)}"}), 500

# Step 2 & 3: Select Top Stocks
@app.route("/select", methods=["POST"])
@swag_from({
    'tags': ['Stock Selection'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'tickers': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of stock tickers'},
                    'user_id': {'type': 'integer', 'default': 1, 'description': 'User ID'},
                    'force_refresh': {'type': 'boolean', 'default': False, 'description': 'Force refresh'}
                },
                'required': ['tickers']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Top stocks selected',
            'schema': {
                'type': 'object',
                'properties': {
                    'top_10': {'type': 'array', 'items': {'type': 'string'}},
                    'top_5': {'type': 'array', 'items': {'type': 'string'}}
                }
            }
        },
        '400': {'description': 'Invalid input'},
        '404': {'description': 'No valid stocks found'},
        '500': {'description': 'Server error'}
    }
})
def select_stocks():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400

    try:
        since = None if force_refresh else (datetime.utcnow() - timedelta(days=30))
        top_stocks = select_top_stocks(tickers, since=since)
        if not top_stocks:
            return jsonify({"error": "No valid stocks found"}), 404

        top_10 = [ticker for ticker, _ in top_stocks]
        top_5 = narrow_to_top_5(top_stocks)

        with app.app_context():
            for ticker in top_10:
                if not Selection.query.filter_by(user_id=user_id, ticker=ticker).first():
                    db.session.add(Selection(user_id=user_id, ticker=ticker, is_top_5=(ticker in top_5)))
            db.session.commit()

        return jsonify({"top_10": top_10, "top_5": top_5}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Stock selection failed: {str(e)}"}), 500

# Step 4: Evaluate and Predict
@app.route("/evaluate", methods=["POST"])
@swag_from({
    'tags': ['Evaluation'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'tickers': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of stock tickers'},
                    'user_id': {'type': 'integer', 'default': 1, 'description': 'User ID'},
                    'force_refresh': {'type': 'boolean', 'default': False, 'description': 'Force refresh'}
                },
                'required': ['tickers']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Evaluation completed',
            'schema': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'integer'},
                    'top_5': {'type': 'array', 'items': {'type': 'string'}},
                    'predictions': {'type': 'object'},
                    'performance': {'type': 'object'},
                    'market_health_score': {'type': 'number'},
                    'market_going_well': {'type': 'boolean'}
                }
            }
        },
        '400': {'description': 'Invalid input'},
        '500': {'description': 'Server error'}
    }
})
def evaluate_selections():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400

    try:
        collect_resp = app.test_client().post("/collect", json={"tickers": tickers, "force_refresh": force_refresh})
        if collect_resp.status_code != 201:
            return collect_resp.get_json(), collect_resp.status_code

        select_resp = app.test_client().post("/select", json={"tickers": tickers, "user_id": user_id, "force_refresh": force_refresh})
        if select_resp.status_code != 200:
            return select_resp.get_json(), select_resp.status_code

        top_5 = select_resp.get_json()["top_5"]
        predictions = {}
        performance = {}
        for ticker in top_5:
            df = fetch_stock_data(ticker, period="3d")
            if not df.empty:
                metrics = calculate_performance_metrics(df)
                news_data = fetch_news(ticker, since=datetime.utcnow() - timedelta(days=2))
                earnings_data = fetch_earnings_calendar(ticker, since=datetime.utcnow() - timedelta(days=2))
                latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].tolist()

                pred, conf, expl = grok_api_predict(ticker, latest_data, news_data, earnings_data)
                predictions[ticker] = {"prediction": pred, "confidence": round(conf * 100, 2), "explanation": expl}
                performance[ticker] = metrics if metrics else {"error": "Insufficient data"}

                with app.app_context():
                    stock = Stock.query.filter_by(ticker=ticker).first()
                    pred_date = df.iloc[-1]['Date'].strftime('%Y-%m-%d')
                    if not Prediction.query.filter_by(stock_id=stock.stock_id, date=pred_date).first():
                        db.session.add(Prediction(stock_id=stock.stock_id, date=pred_date, 
                                                  predicted_movement=pred, confidence_score=conf, explanation=expl))
                    db.session.commit()

        top_5_perf = {k: v for k, v in performance.items() if k in top_5 and "error" not in v}
        market_health_score = (
            0.4 * np.mean([p["price_change"] for p in top_5_perf.values()]) / 100 +
            0.3 * np.mean([p["sharpe_ratio"] for p in top_5_perf.values()]) +
            0.2 * np.mean([p["momentum"] for p in top_5_perf.values()]) -
            0.1 * np.mean([p["volatility"] for p in top_5_perf.values()])
        ) if top_5_perf else 0
        market_going_well = market_health_score > 0

        result = {
            "user_id": user_id,
            "top_5": top_5,
            "predictions": predictions,
            "performance": performance,
            "market_health_score": round(market_health_score, 4),
            "market_going_well": bool(market_going_well)
        }
        print(f"Evaluation result: {result}")
        
        if not market_going_well:
            print("Market not going well, restarting process...")
            return app.test_client().post("/evaluate", json={"tickers": tickers, "user_id": user_id, "force_refresh": True}).get_json(), 200

        return jsonify(result), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

# Step 5: Enhanced Monitor Endpoint with Grok AI
@app.route("/monitor/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Monitoring'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol (e.g., AAPL)'
        },
        {
            'name': 'days',
            'in': 'query',
            'type': 'integer',
            'default': 2,
            'required': False,
            'description': 'Number of days to monitor'
        }
    ],
    'responses': {
        '200': {
            'description': 'Monitoring result with Grok AI analysis',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'initial_prediction': {'type': 'string'},
                    'initial_confidence': {'type': 'number'},
                    'price_change': {'type': 'number'},
                    'status': {'type': 'string'},
                    'performance': {'type': 'object'},
                    'latest_prediction': {'type': 'string'},
                    'latest_confidence': {'type': 'number'},
                    'market_outlook': {'type': 'string'}
                }
            }
        },
        '404': {'description': 'Stock or data not found'},
        '500': {'description': 'Server error'}
    }
})
def monitor_stock(ticker):
    ticker = ticker.upper()
    days = request.args.get("days", 2, type=int)
    try:
        with app.app_context():
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
            metrics = calculate_performance_metrics(df)
            status = "Correct" if (
                (pred.predicted_movement == "Buy" and price_change > 0) or
                (pred.predicted_movement == "Sell" and price_change < 0) or
                (pred.predicted_movement == "Hold" and abs(price_change) < 2)
            ) else "Incorrect"

            news_data = fetch_news(ticker, since=datetime.utcnow() - timedelta(days=days))
            earnings_data = fetch_earnings_calendar(ticker, since=datetime.utcnow() - timedelta(days=days))
            latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].tolist()
            grok_pred, grok_conf, grok_expl = grok_api_predict(ticker, latest_data, news_data, earnings_data)

            sentiment_avg = sum(n['sentiment_score'] for n in news_data) / len(news_data) if news_data else 0
            market_outlook = f"Grok analysis for {ticker}: Market outlook is {'positive' if grok_pred == 'Buy' else 'cautious' if grok_pred == 'Hold' else 'negative'} based on {price_change:.2f}% price change, sentiment ({sentiment_avg:.2f}), and recent trends. {grok_expl}"

            latest_date = df.iloc[-1]['Date'].strftime('%Y-%m-%d')
            if latest_date != pred.date:
                new_pred = Prediction(stock_id=stock.stock_id, date=latest_date, 
                                      predicted_movement=grok_pred, confidence_score=grok_conf, explanation=grok_expl)
                db.session.add(new_pred)
                db.session.commit()

            return jsonify({
                "ticker": ticker,
                "initial_prediction": pred.predicted_movement,
                "initial_confidence": round(pred.confidence_score * 100, 2),
                "price_change": round(price_change, 2),
                "status": status,
                "performance": metrics,
                "latest_prediction": grok_pred,
                "latest_confidence": round(grok_conf * 100, 2),
                "market_outlook": market_outlook
            }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Monitoring failed: {str(e)}"}), 500

# Step 6: Execute Trades
@app.route("/trade/<ticker>", methods=["POST"])
@swag_from({
    'tags': ['Trading'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        },
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'user_id': {'type': 'integer', 'description': 'User ID'},
                    'action': {'type': 'string', 'enum': ['Buy', 'Sell'], 'description': 'Trade action'}
                },
                'required': ['user_id', 'action']
            }
        }
    ],
    'responses': {
        '201': {
            'description': 'Trade executed',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'price': {'type': 'number'}
                }
            }
        },
        '400': {'description': 'Invalid input'},
        '404': {'description': 'Stock not found'},
        '500': {'description': 'Server error'}
    }
})
def execute_trade(ticker):
    ticker = ticker.upper()
    data = request.get_json()
    user_id = data.get("user_id")
    action = data.get("action")
    if not user_id or action not in ["Buy", "Sell"]:
        return jsonify({"error": "Valid user_id and action (Buy/Sell) required"}), 400

    try:
        with app.app_context():
            stock = Stock.query.filter_by(ticker=ticker).first()
            if not stock:
                return jsonify({"error": f"Stock {ticker} not found"}), 404

            df = fetch_stock_data(ticker, period="1d")
            if df.empty:
                return jsonify({"error": f"No current price data for {ticker}"}), 404
            current_price = df.iloc[-1]['Close']

            trade = Trade(user_id=user_id, stock_id=stock.stock_id, action=action, 
                          price=current_price, date=datetime.utcnow().strftime('%Y-%m-%d'))
            db.session.add(trade)
            db.session.commit()

            return jsonify({"message": f"{action} executed for {ticker}", "price": current_price}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Trade failed: {str(e)}"}), 500

# Full Automation Endpoint
@app.route("/run", methods=["POST"])
@swag_from({
    'tags': ['Automation'],
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'tickers': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of stock tickers'},
                    'user_id': {'type': 'integer', 'default': 1, 'description': 'User ID'},
                    'force_refresh': {'type': 'boolean', 'default': False, 'description': 'Force refresh'}
                },
                'required': ['tickers']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Full process completed',
            'schema': {
                'type': 'object',
                'properties': {
                    'message': {'type': 'string'},
                    'top_5': {'type': 'array', 'items': {'type': 'string'}},
                    'predictions': {'type': 'object'},
                    'trades': {'type': 'object'},
                    'market_going_well': {'type': 'boolean'}
                }
            }
        },
        '400': {'description': 'Invalid input'},
        '500': {'description': 'Server error'}
    }
})
def run_full_process():
    data = request.get_json()
    tickers = data.get("tickers", [])
    user_id = data.get("user_id", 1)
    force_refresh = data.get("force_refresh", False)
    if not tickers or not isinstance(tickers, list):
        return jsonify({"error": "Valid tickers list required"}), 400

    try:
        collect_resp = app.test_client().post("/collect", json={"tickers": tickers, "force_refresh": force_refresh})
        if collect_resp.status_code != 201:
            return collect_resp.get_json(), collect_resp.status_code

        eval_resp = app.test_client().post("/evaluate", json={"tickers": tickers, "user_id": user_id, "force_refresh": force_refresh})
        if eval_resp.status_code != 200:
            return eval_resp.get_json(), eval_resp.status_code

        result = eval_resp.get_json()
        top_5 = result["top_5"]
        predictions = result["predictions"]

        trades = {}
        for ticker in top_5:
            monitor_resp = app.test_client().get(f"/monitor/{ticker}")
            if monitor_resp.status_code == 200:
                monitor_data = monitor_resp.get_json()
                if monitor_data["status"] == "Correct" and monitor_data["latest_prediction"] in ["Buy", "Sell"]:
                    trade_resp = app.test_client().post(f"/trade/{ticker}", json={"user_id": user_id, "action": monitor_data["latest_prediction"]})
                    if trade_resp.status_code == 201:
                        trades[ticker] = trade_resp.get_json()

        return jsonify({
            "message": "Full process completed",
            "top_5": top_5,
            "predictions": predictions,
            "trades": trades,
            "market_going_well": result["market_going_well"]
        }), 200
    except Exception as e:
        return jsonify({"error": f"Full process failed: {str(e)}"}), 500

# Get All Stocks
@app.route("/stocks", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'responses': {
        '200': {
            'description': 'List of all stocks',
            'schema': {
                'type': 'object',
                'properties': {
                    'stocks': {
                        'type': 'array',
                        'items': {
                            'type': 'object',
                            'properties': {
                                'ticker': {'type': 'string'},
                                'name': {'type': 'string'},
                                'sector': {'type': 'string'},
                                'exchange': {'type': 'string'},
                                'last_updated': {'type': 'string'}
                            }
                        }
                    },
                    'count': {'type': 'integer'}
                }
            }
        },
        '404': {'description': 'No stocks found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get All Selected Stocks Data
@app.route("/selected_stocks", methods=["GET"])
@swag_from({
    'tags': ['Stock Selection'],
    'parameters': [
        {
            'name': 'tickers',
            'in': 'query',
            'type': 'string',
            'required': False,
            'description': 'Comma-separated list of tickers (e.g., AAPL,TSLA)'
        }
    ],
    'responses': {
        '200': {
            'description': 'Selected stocks data',
            'schema': {
                'type': 'object',
                'properties': {
                    'top_10': {'type': 'array', 'items': {'type': 'object'}},
                    'top_5': {'type': 'array', 'items': {'type': 'object'}},
                    'count': {'type': 'integer'}
                }
            }
        },
        '404': {'description': 'No stocks found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get Specific Selected Stock Data
@app.route("/selected_stocks/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Selection'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'Selected stock data',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'composite_score': {'type': 'number'},
                    'sharpe_ratio': {'type': 'number'},
                    'momentum': {'type': 'number'},
                    'grok_score': {'type': 'number'},
                    'news_sentiment': {'type': 'number'},
                    'x_sentiment': {'type': 'number'},
                    'prediction': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'explanation': {'type': 'string'}
                }
            }
        },
        '404': {'description': 'Stock not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get Stock Details
@app.route("/stocks/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'Stock details',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'composite_score': {'type': 'number'},
                    'sharpe_ratio': {'type': 'number'},
                    'momentum': {'type': 'number'},
                    'grok_score': {'type': 'number'},
                    'news_sentiment': {'type': 'number'},
                    'x_sentiment': {'type': 'number'},
                    'prediction': {'type': 'string'},
                    'confidence': {'type': 'number'},
                    'explanation': {'type': 'string'}
                }
            }
        },
        '404': {'description': 'Stock not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get Stock Prices
@app.route("/stock_prices/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'Stock price history',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'prices': {'type': 'array', 'items': {'type': 'object'}},
                    'count': {'type': 'integer'}
                }
            }
        },
        '404': {'description': 'Stock or prices not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get Earnings
@app.route("/earnings/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'Earnings data',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'earnings': {'type': 'array', 'items': {'type': 'object'}},
                    'count': {'type': 'integer'}
                }
            }
        },
        '404': {'description': 'Stock or earnings not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get News
@app.route("/news/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'News data',
            'schema': {
                'type': 'object',
                'properties': {
                    'ticker': {'type': 'string'},
                    'news': {'type': 'array', 'items': {'type': 'object'}},
                    'count': {'type': 'integer'}
                }
            }
        },
        '404': {'description': 'Stock not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get All Data for a Stock
@app.route("/data/<ticker>", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'parameters': [
        {
            'name': 'ticker',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'Stock ticker symbol'
        }
    ],
    'responses': {
        '200': {
            'description': 'All data for a stock',
            'schema': {
                'type': 'object',
                'properties': {
                    'stock': {'type': 'object'},
                    'prices': {'type': 'array', 'items': {'type': 'object'}},
                    'earnings': {'type': 'array', 'items': {'type': 'object'}},
                    'news': {'type': 'array', 'items': {'type': 'object'}}
                }
            }
        },
        '404': {'description': 'Stock not found'},
        '500': {'description': 'Server error'}
    }
})
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

# Get All News
@app.route("/all_news", methods=["GET"])
@swag_from({
    'tags': ['Stock Data'],
    'responses': {
        '200': {
            'description': 'All news data',
            'schema': {
                'type': 'object',
                'properties': {
                    'news': {'type': 'array', 'items': {'type': 'object'}},
                    'count': {'type': 'integer'}
                }
            }
        },
        '500': {'description': 'Server error'}
    }
})
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