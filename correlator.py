import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def merge_sentiment_with_prices(sentiment_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if sentiment_df.empty:
        return pd.DataFrame()

    # Get unique dates from sentiment
    sentiment_df['date_only'] = pd.to_datetime(sentiment_df['date']).dt.date
    start_date = sentiment_df['date_only'].min()
    end_date = sentiment_df['date_only'].max() + pd.Timedelta(days=1)

    # Download stock prices using yfinance
    price_df = yf.download(ticker, start=start_date, end=end_date)
    price_df = price_df[['Close']].reset_index()
    price_df['date_only'] = price_df['Date'].dt.date

    # Aggregate sentiment scores
    sentiment_agg = sentiment_df.groupby('date_only').agg(
        avg_sentiment=('weighted_score', 'mean'),
        count=('headline', 'count')
    ).reset_index()

    # Merge with price
    merged = pd.merge(price_df, sentiment_agg, on='date_only', how='left')
    return merged

def plot_sentiment_vs_price(df: pd.DataFrame, ticker: str, output_dir: str) -> str:
    if df.empty or 'avg_sentiment' not in df.columns:
        return ""

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title(f"{ticker} - Avg Sentiment vs Closing Price")

    ax1.plot(df['date_only'], df['Close'], color='blue', label='Close Price')
    ax2 = ax1.twinx()
    ax2.plot(df['date_only'], df['avg_sentiment'], color='green', label='Avg Sentiment')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color='blue')
    ax2.set_ylabel('Avg Sentiment', color='green')

    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.tight_layout()
    path = f"{output_dir}/{ticker}_sentiment_vs_price.png"
    plt.savefig(path)
    plt.clf()
    return path