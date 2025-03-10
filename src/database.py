from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Stock(db.Model):
    __tablename__ = 'stocks'
    stock_id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(10), unique=True, nullable=False)
    name = db.Column(db.String(100))
    sector = db.Column(db.String(50))
    exchange = db.Column(db.String(50))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Stock(ticker={self.ticker}, name={self.name})>"

class StockPrice(db.Model):
    __tablename__ = 'stock_prices'
    price_id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Integer)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<StockPrice(ticker_id={self.stock_id}, date={self.date}, close={self.close})>"

class EarningsCalendar(db.Model):
    __tablename__ = 'earnings_calendar'
    earnings_id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    earnings_date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    estimated_eps = db.Column(db.Float)
    actual_eps = db.Column(db.Float)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<EarningsCalendar(stock_id={self.stock_id}, earnings_date={self.earnings_date})>"

class NewsSentiment(db.Model):
    __tablename__ = 'news_sentiment'
    news_id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    source = db.Column(db.String(100), nullable=False)
    date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    sentiment_score = db.Column(db.Float, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<NewsSentiment(stock_id={self.stock_id}, source={self.source}, date={self.date})>"

class Prediction(db.Model):
    __tablename__ = 'predictions'
    prediction_id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    predicted_movement = db.Column(db.String(10), nullable=False)  # Buy/Sell/Hold
    confidence_score = db.Column(db.Float, nullable=False)
    explanation = db.Column(db.Text)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction(stock_id={self.stock_id}, date={self.date}, predicted_movement={self.predicted_movement})>"

class Trade(db.Model):
    __tablename__ = 'trades'
    trade_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.stock_id'), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # Buy/Sell
    price = db.Column(db.Float, nullable=False)
    date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Trade(user_id={self.user_id}, stock_id={self.stock_id}, action={self.action}, date={self.date})>"

class Selection(db.Model):
    __tablename__ = 'selections'
    selection_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # To identify user
    ticker = db.Column(db.String(10), nullable=False)
    selection_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_top_5 = db.Column(db.Boolean, default=False)  # True if in top_5
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Selection(user_id={self.user_id}, ticker={self.ticker}, selection_date={self.selection_date}, is_top_5={self.is_top_5})>"

def init_db(app):
    with app.app_context():
        db.init_app(app)
        db.create_all()
        print("Database tables created or verified.")

