import pandas as pd
import numpy as np
import yfinance as yf
import requests
from textblob import TextBlob
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GROK_API_KEY = os.getenv('GROK_API_KEY')
GROK_API_URL = os.getenv('GROK_API_URL')

def fetch_stock_data(ticker, period="2mo", since=None):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if df.empty:
            raise ValueError(f"No stock data for {ticker}")
        
        df.reset_index(inplace=True)
        if df['Date'].dtype.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df = df.drop_duplicates(subset=['Date'])
        
        if since:
            df = df[df['Date'] > since]
        
        if df.empty or len(df) < 2:
            raise ValueError(f"Insufficient data for {ticker} after filtering")
        
        return df
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}")

def fetch_earnings_calendar(ticker, since=None):
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if not cal:
            return []
        
        earnings_date = None
        estimated_eps = None
        
        if isinstance(cal, dict) and 'Earnings Date' in cal:
            earnings_dates = cal['Earnings Date']
            if isinstance(earnings_dates, list) and earnings_dates:
                earnings_date_raw = earnings_dates[0]
            else:
                earnings_date_raw = earnings_dates
            
            if hasattr(earnings_date_raw, 'strftime'):
                earnings_date = earnings_date_raw.strftime('%Y-%m-%d')
            else:
                earnings_date = str(earnings_date_raw)[:10] if earnings_date_raw else None
            
            estimated_eps = cal.get('Earnings Average', None)
        
        if earnings_date and (not since or datetime.strptime(earnings_date, '%Y-%m-%d') > since):
            return [{"earnings_date": earnings_date, "estimated_eps": estimated_eps, "actual_eps": None}]
        
        return []
    except Exception as e:
        raise Exception(f"Error fetching earnings for {ticker}: {str(e)}")

def fetch_news(ticker, since=None):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        end_date = datetime.now()
        min_start_date = end_date - timedelta(days=10)  # Adjust based on your NewsAPI tier
        
        start_date = since if since is not None else min_start_date
        if start_date < min_start_date:
            start_date = min_start_date
        
        stock = yf.Ticker(ticker)
        company_name = stock.info.get("longName", ticker).split(" ")[0]
        query = f"{ticker} OR {company_name}"
        
        print(f"Fetching news for {ticker} with query '{query}' from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        articles = newsapi.get_everything(q=query, 
                                         from_param=start_date.strftime('%Y-%m-%d'), 
                                         to=end_date.strftime('%Y-%m-%d'), 
                                         language='en', 
                                         sort_by='relevancy', 
                                         page_size=20)
        
        if articles['status'] != 'ok':
            print(f"NewsAPI error for {ticker}: {articles}")
            return []
        
        print(f"Raw NewsAPI response for {ticker}: totalResults={articles['totalResults']}, articles={len(articles['articles'])}")
        news_data = []
        seen = set()
        for article in articles['articles']:
            key = (article['source']['name'], article['publishedAt'][:10])
            if key not in seen and (not since or datetime.strptime(article['publishedAt'][:10], '%Y-%m-%d') > since):
                sentiment = TextBlob(article['title'] + " " + (article['description'] or "")).sentiment.polarity
                news_data.append({
                    "source": article['source']['name'],
                    "date": article['publishedAt'][:10],
                    "sentiment_score": sentiment
                })
                seen.add(key)
        print(f"Fetched {len(news_data)} news articles for {ticker}: {news_data}")
        return news_data if news_data else []
    except Exception as e:
        print(f"Warning: Failed to fetch news for {ticker}: {str(e)}")
        return []

def preprocess_data(df):
    try:
        if df.empty or not all(col in df for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("Incomplete stock data")
        
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-6)
        
        df['SMA10'] = SMAIndicator(df['Close'], window=10).sma_indicator()
        df['RSI'] = pd.Series(pd.Series(df['Close']).diff().apply(lambda x: x if x > 0 else 0).rolling(window=14).mean() / 
                              (pd.Series(-df['Close'].diff().apply(lambda x: -x if x < 0 else 0)).rolling(window=14).mean() + 1e-6) * 100 + 100)
        macd = MACD(df['Close'])
        df['MACD'] = macd.macd()
        bb = BollingerBands(df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        
        return df.dropna()
    except Exception as e:
        raise Exception(f"Error preprocessing data: {str(e)}")

def grok_api_predict(ticker, stock_data, news_data, earnings_data, step="predict"):
    try:
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "ticker": ticker,
            "stock_data": stock_data,
            "news_sentiment": news_data,
            "earnings_data": earnings_data,
            "step": step
        }
        # Uncomment when real Grok API is available
        # response = requests.post(GROK_API_URL, json=payload, headers=headers, timeout=10)
        # response.raise_for_status()
        # result = response.json()
        
        # Simulated Grok API
        if step == "select":
            score = np.random.uniform(0, 1)
            return score
        else:
            simulated_response = {
                "prediction": np.random.choice(["Buy", "Sell", "Hold"]),
                "confidence": round(np.random.uniform(0.6, 0.95), 2),
                "explanation": f"Analysis of {ticker} trends, sentiment, and earnings"
            }
            return simulated_response["prediction"], simulated_response["confidence"], simulated_response["explanation"]
    except requests.exceptions.RequestException as e:
        raise Exception(f"Grok API request failed for {ticker}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in Grok API prediction for {ticker}: {str(e)}")

def score_stock(ticker, since=None):
    try:
        end_date = datetime.utcnow()
        if since is None:
            since = end_date - timedelta(days=30)
        
        if isinstance(since, pd.Timestamp):
            since = since.to_pydatetime().replace(tzinfo=None)
        elif isinstance(since, datetime):
            since = since.replace(tzinfo=None)
        
        df = fetch_stock_data(ticker, since=since)
        if df.empty or len(df) < 2:
            return None, f"Insufficient stock data for {ticker}"
        
        momentum = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]
        daily_returns = df['Close'].pct_change().dropna()
        volatility = daily_returns.std()
        sharpe_ratio = daily_returns.mean() / volatility if volatility != 0 else 0
        
        news_data = fetch_news(ticker, since=since)
        news_sentiment = sum(n['sentiment_score'] for n in news_data) / len(news_data) if news_data else 0
        
        latest_data = df.iloc[-1][['Open', 'High', 'Low', 'Close', 'Volume']].tolist()
        earnings_data = fetch_earnings_calendar(ticker, since=since)
        prediction, confidence, explanation = grok_api_predict(ticker, latest_data, news_data, earnings_data)
        grok_score = confidence * (1 if prediction == "Buy" else -1 if prediction == "Sell" else 0)
        
        x_sentiment = 0.1 if grok_score > 0 else -0.1 if grok_score < 0 else 0
        
        composite_score = (
            0.3 * sharpe_ratio + 
            0.2 * momentum + 
            0.3 * grok_score + 
            0.1 * news_sentiment + 
            0.1 * x_sentiment
        )
        
        return {
            "ticker": ticker,
            "sharpe_ratio": sharpe_ratio,
            "momentum": momentum,
            "grok_score": grok_score,
            "news_sentiment": news_sentiment,
            "x_sentiment": x_sentiment,
            "composite_score": composite_score,
            "prediction": prediction,
            "confidence": confidence,
            "explanation": explanation,
            "volume": float(df['Volume'].mean())  # Added for narrow_to_top_5 compatibility
        }, None
    except Exception as e:
        return None, str(e)

def select_top_stocks(tickers, since=None):
    top_stocks = []
    end_date = datetime.utcnow()
    
    if since is None:
        since = end_date - timedelta(days=30)
    
    if isinstance(since, pd.Timestamp):
        since = since.to_pydatetime().replace(tzinfo=None)
    elif isinstance(since, datetime):
        since = since.replace(tzinfo=None)
    
    print(f"Processing {len(tickers)} tickers: {tickers}")
    
    for ticker in tickers:
        result, error = score_stock(ticker, since=since)
        if result:
            top_stocks.append((ticker, result["composite_score"]))
            print(f"Scored {ticker}: sharpe={result['sharpe_ratio']:.4f}, momentum={result['momentum']:.4f}, "
                  f"grok={result['grok_score']:.4f}, news={result['news_sentiment']:.4f}, x={result['x_sentiment']:.4f}, "
                  f"composite={result['composite_score']:.4f}")
        else:
            print(f"Skipping {ticker} due to error: {error}")
    
    if not top_stocks:
        print("No valid stocks found")
        return []
    
    top_stocks.sort(key=lambda x: x[1], reverse=True)
    available_count = len(top_stocks)
    max_selection = min(available_count, 10)
    top_selection = top_stocks[:max_selection]
    print(f"Selected {max_selection} stocks from {available_count} valid options: {[t[0] for t in top_selection]}")
    
    return top_selection

def narrow_to_top_5(top_stocks):
    if not top_stocks:
        return []
    
    # Fetch full data for top stocks to use volume and sentiment
    top_stock_details = []
    for ticker, score in top_stocks:
        result, _ = score_stock(ticker)
        if result:
            top_stock_details.append(result)
    
    if len(top_stock_details) < 5:
        return [stock["ticker"] for stock in top_stock_details]
    
    top_5 = sorted(top_stock_details, 
                   key=lambda x: x["composite_score"] + x["news_sentiment"] * 0.5 + x["volume"] * 0.000001, 
                   reverse=True)[:5]
    return [stock["ticker"] for stock in top_5]