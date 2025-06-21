import time
from typing import List, Dict, Optional, Tuple
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import logging
import os
import torch
import argparse
from tqdm import tqdm
import json

# =======================
# Configure logging
# =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =======================
# Configuration
# =======================
class Config:
    """Configuration class for the sentiment analyzer"""
    DEFAULT_TICKERS = ["AAPL", "TSLA", "GOOGL", "NVDA", "AMZN", "MSFT", "META", "AMD"]
    MAX_HEADLINES = 15
    MAX_RETRIES = 3
    OUTPUT_DIR = "sentiment_data"
    MODEL_NAME = "ProsusAI/finbert"
    HEADLESS = True
    REQUEST_TIMEOUT = 30
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    @classmethod
    def load_from_file(cls, filepath: str = "config.json"):
        """Load configuration from JSON file"""
        if os.path.exists(filepath):
            with open(filepath) as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)

# Load configuration
Config.load_from_file()

# =======================
# Sentiment Analysis
# =======================
class SentimentAnalyzer:
    """Handles sentiment analysis using FinBERT model"""
    _instance = None
    label_map = {"positive": 1, "neutral": 0, "negative": -1}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the sentiment analysis pipeline"""
        try:
            logger.info("Loading FinBERT sentiment model...")
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=False,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load sentiment model: {str(e)}")
            raise

    def analyze(self, text: str) -> Optional[Dict]:
        """Analyze text sentiment"""
        try:
            if not text or not isinstance(text, str):
                logger.warning(f"Invalid text for analysis: {text}")
                return None

            # Skip common non-news headlines
            skip_phrases = ["Search the web", "Oops", "Trending Tickers", "More from"]
            if any(phrase in text for phrase in skip_phrases):
                return None

            result = self.pipeline(text)[0]
            label = result['label'].lower()
            score = round(result['score'], 4)
            weighted_score = round(self.label_map[label] * score, 4)

            return {
                "label": label,
                "score": score,
                "weighted_score": weighted_score,
                "original_text": text
            }
        except Exception as e:
            logger.error(f"Error analyzing text '{text[:50]}...': {str(e)}")
            return None

# =======================
# Web Scraper
# =======================
class YahooFinanceScraper:
    """Handles scraping of Yahoo Finance news"""
    def __init__(self):
        self.driver = self._init_driver()

    def _init_driver(self) -> webdriver.Chrome:
        """Initialize and return a configured Chrome WebDriver"""
        options = Options()
        if Config.HEADLESS:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument(f'user-agent={Config.USER_AGENT}')

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(Config.REQUEST_TIMEOUT)
        return driver

    def _accept_cookies(self):
        """Handle cookie consent popup if present"""
        try:
            consent_buttons = self.driver.find_elements(By.CSS_SELECTOR, 'button[type=submit]')
            if consent_buttons:
                consent_buttons[0].click()
                time.sleep(1)
        except Exception as e:
            logger.warning(f"Could not handle cookie consent: {str(e)}")

    def scrape_news(self, ticker: str, max_headlines: int = Config.MAX_HEADLINES) -> pd.DataFrame:
        """Scrape news headlines for a given ticker"""
        try:
            url = f"https://finance.yahoo.com/quote/{ticker}/news"
            logger.info(f"Scraping {url}")

            self.driver.get(url)
            
            # Wait for either headlines or cookie consent to appear
            WebDriverWait(self.driver, 15).until(
                lambda d: d.find_elements(By.CSS_SELECTOR, 'h3') or 
                         d.find_elements(By.CSS_SELECTOR, 'button[type=submit]')
            )

            self._accept_cookies()

            # Wait for headlines to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h3'))
            )

            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            headlines = []

            # Filter only valid news headlines (excluding promotional content)
            valid_h3s = [
                h3 for h3 in soup.select('h3')
                if h3.find_parent('a') and '/news/' in h3.find_parent('a').get('href', '')
            ][:max_headlines]

            analyzer = SentimentAnalyzer()
            
            for h3 in valid_h3s:
                try:
                    headline = h3.get_text(strip=True)
                    if not headline:
                        continue

                    parent_link = h3.find_parent('a')
                    link = (
                        "https://finance.yahoo.com" + parent_link['href'] 
                        if parent_link and parent_link['href'].startswith('/') 
                        else parent_link['href'] if parent_link else ''
                    )

                    summary_tag = h3.find_next('p')
                    summary = (
                        summary_tag.get_text(strip=True) 
                        if summary_tag and summary_tag.name == 'p' 
                        else ''
                    )

                    sentiment = analyzer.analyze(headline)
                    if not sentiment:
                        continue

                    headlines.append({
                        'ticker': ticker,
                        'headline': headline,
                        'summary': summary,
                        'link': link,
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': datetime.now().isoformat(),
                        'sentiment_label': sentiment['label'],
                        'sentiment_score': sentiment['score'],
                        'weighted_score': sentiment['weighted_score']
                    })
                except Exception as e:
                    logger.warning(f"Error processing headline: {str(e)}")
                    continue

            return pd.DataFrame(headlines)

        except Exception as e:
            logger.error(f"Failed to scrape {ticker}: {str(e)}")
            return pd.DataFrame()

    def close(self):
        """Clean up the WebDriver"""
        if self.driver:
            self.driver.quit()

# =======================
# Data Processing
# =======================
class DataProcessor:
    """Handles data processing and visualization"""
    @staticmethod
    def save_results(df: pd.DataFrame, ticker: str, output_dir: str = Config.OUTPUT_DIR) -> str:
        """Save results to CSV file and return file path"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{output_dir}/{ticker}_sentiment_{timestamp}.csv"
        df.to_csv(filename, index=False)
        return filename

    @staticmethod
    def generate_visualizations(df: pd.DataFrame, ticker: str, output_dir: str = Config.OUTPUT_DIR):
        """Generate visualizations for the sentiment data"""
        if df.empty:
            return

        os.makedirs(output_dir, exist_ok=True)
        
        # Sentiment distribution plot
        plt.figure(figsize=(10, 6))
        ax = df['sentiment_label'].value_counts().plot(
            kind='bar', 
            color=['green', 'gray', 'red'],
            alpha=0.7
        )
        
        plt.title(f"{ticker} - Sentiment Distribution", pad=20)
        plt.xlabel("Sentiment", labelpad=10)
        plt.ylabel("Count", labelpad=10)
        
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(str(p.get_height()), 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')
        
        plt.tight_layout()
        plot_path = f"{output_dir}/{ticker}_sentiment_plot.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.clf()
        
        return plot_path

    @staticmethod
    def calculate_stats(df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the data"""
        if df.empty:
            return {}
            
        pos_count = len(df[df['sentiment_label'] == 'positive'])
        neg_count = len(df[df['sentiment_label'] == 'negative'])
        neu_count = len(df[df['sentiment_label'] == 'neutral'])
        avg_score = df['weighted_score'].mean()
        
        return {
            'positive': pos_count,
            'negative': neg_count,
            'neutral': neu_count,
            'average_score': round(avg_score, 4),
            'total': len(df)
        }

# =======================
# Main Application
# =======================
# Complete FinBERT-Based Market Sentiment Analysis Tool with yfinance Integration

from correlator import merge_sentiment_with_prices, plot_sentiment_vs_price
from faiss_search import SemanticNewsSearch


def main(tickers: List[str], output_dir: str = Config.OUTPUT_DIR):
    """Main function to scrape and analyze news for multiple tickers"""
    scraper = YahooFinanceScraper()
    processor = DataProcessor()
    searcher = SemanticNewsSearch()

    try:
        for ticker in tqdm(tickers, desc="Processing tickers"):
            logger.info(f"Processing {ticker}...")

            try:
                df = scraper.scrape_news(ticker)

                if not df.empty:
                    # Save sentiment results
                    filename = processor.save_results(df, ticker, output_dir)
                    logger.info(f"Saved {len(df)} headlines to {filename}")

                    # Generate sentiment distribution plot
                    dist_plot = processor.generate_visualizations(df, ticker, output_dir)
                    logger.info(f"Saved sentiment plot to {dist_plot}")

                    # Correlate with stock price using yfinance
                    price_df = merge_sentiment_with_prices(df, ticker)
                    if not price_df.empty:
                        price_plot = plot_sentiment_vs_price(price_df, ticker, output_dir)
                        logger.info(f"Saved correlation plot to {price_plot}")
                    else:
                        logger.warning(f"No price data available for {ticker}")

                    # Semantic Search Indexing
                    searcher.build_index(headlines=df['headline'].tolist(), metadata=df.to_dict('records'), ticker=ticker)
                    logger.info(f"Built semantic index for {ticker}")

                    # Calculate and log stats
                    stats = processor.calculate_stats(df)
                    logger.info(
                        f"Sentiment summary for {ticker}:\n"
                        f"Positive: {stats['positive']} | "
                        f"Neutral: {stats['neutral']} | "
                        f"Negative: {stats['negative']}\n"
                        f"Average score: {stats['average_score']:.2f} | "
                        f"Total headlines: {stats['total']}"
                    )
                else:
                    logger.warning(f"No valid headlines found for {ticker}")

            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

    finally:
        scraper.close()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Financial News Sentiment Analysis")
    parser.add_argument(
        '--tickers', 
        type=str, 
        default=",".join(Config.DEFAULT_TICKERS),
        help="Comma-separated list of tickers to analyze"
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default=Config.OUTPUT_DIR,
        help="Directory to store output files"
    )
    return parser.parse_args()
if __name__ == "__main__":
    import sys
    if "ipykernel" in sys.argv[0] or "spyder" in sys.modules:
        # Running inside Jupyter or IDE — use defaults
        tickers = Config.DEFAULT_TICKERS
        main(tickers)
    else:
        args = parse_arguments()
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
        main(tickers, args.output_dir)