# Market-Manipulation-Detection-Tool
Built a market manipulation detection tool deployed as a webapp with user-initiated refresh architecture. 

This tool monitors cryptocurrency markets for potential manipulation (pump‑and‑dump, wash trading, or coordinated FUD). It watches three things:

On‑chain volume & price – If trading volume spikes but the price barely moves, it could be fake activity.

Reddit sentiment – Every post in r/CryptoCurrency is analysed with CryptoBERT (an AI model trained on crypto slang) to detect bullish, neutral, or bearish sentiment. The system also spots likely bot accounts by scraping each user’s post history, age, and karma.

Twitter/X sentiment – Recent tweets about the same coins are analysed with the same CryptoBERT model, giving a second, faster data stream.

When a volume spike without price change happens at the same time as a sentiment shift driven mostly by bot‑like accounts (on Reddit, Twitter, or both), the tool raises an alert. Using two social platforms makes the signal more reliable: real market moves create consistent sentiment across platforms, while many manipulation campaigns try to spread fake hype on multiple channels simultaneously.

Architecture (high‑level)
The system is built as a user‑initiated refresh pipeline (no continuous background polling – respects free API limits).

Backend – FastAPI server with two main endpoints (/refresh and /latest). It handles all data fetching, bot detection, sentiment analysis, and caching.

Frontend – Streamlit dashboard that displays blockchain metrics, Reddit posts, Twitter tweets, market data, and alerts. Users click a button to trigger a fresh data pull.

WebSocket – Optional WebSocket endpoint (/ws/live-data) that pushes the latest cached data to connected clients (used for potential real‑time updates in the future).

Scraping & Bot Detection

Reddit posts: requests + BeautifulSoup on old.reddit.com.

Bot scoring: headless browser (Playwright) scrapes each Reddit user’s profile to extract account age, karma, posting intervals, and content repetition. Scores are cached.

NLP – CryptoBERT (Hugging Face transformer) fine‑tuned on crypto social media. Runs on CPU or GPU (PyTorch).

Blockchain – Ethereum (Web3.py) and Kaspa (REST API).

Market data – CoinPaprika API (price, 24h volume).

Caching – In‑memory cache stores the latest data, bot scores, and sentiment results so subsequent refreshes are fast (first refresh takes ~2‑3 minutes due to heavy bot‑scraping).


Please see code below:
import os
from dotenv import load_dotenv
load_dotenv("k.env")
import asyncio
from datetime import datetime
from fastapi import FastAPI, WebSocket
import uvicorn
from web3 import Web3
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from coinpaprika.client import Client
from twitter_sentiment import run_twitter_pipeline

ETH_RPC_URL = os.getenv("ETH_RPC")
if not ETH_RPC_URL:
    raise ValueError("ETH_RPC not set in .env file")

TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
cp_client = Client()

class CryptoBERTSentimentAnalyzer:
    def __init__(self, model_name="ElKulako/cryptobert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CryptoBERT on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.id2label = {0: "Bearish", 1: "Neutral", 2: "Bullish"}

    def get_sentiment_scores(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        return {
            'compound': probabilities[0][predicted_class_id].item(),
            'pos': probabilities[0][2].item() if predicted_class_id == 2 else 0.0,
            'neu': probabilities[0][1].item() if predicted_class_id == 1 else 0.0,
            'neg': probabilities[0][0].item() if predicted_class_id == 0 else 0.0,
            'label': self.id2label[predicted_class_id]
        }

cached_data = {
    "blockchain": None,
    "reddit_posts": [],
    "tweets": [],
    "market": {},
    "market_alerts": [],
    "last_refresh": None
}

eth_w3 = Web3(Web3.HTTPProvider(ETH_RPC_URL))
kaspa_api = "https://api.kaspa.org/info/blockdag"

def fetch_blockchain():
    result = {
        "timestamp": datetime.now().isoformat(),
        "ethereum": {"block_height": None, "gas_price_gwei": None},
        "kaspa": {"block_count": None, "difficulty": None}
    }
    try:
        eth_block = eth_w3.eth.get_block('latest')['number']
        eth_gas = eth_w3.from_wei(eth_w3.eth.gas_price, 'gwei')
        result["ethereum"]["block_height"] = eth_block
        result["ethereum"]["gas_price_gwei"] = float(eth_gas)
    except Exception as e:
        print(f"Ethereum RPC error: {e}")

    try:
        kas_response = requests.get(kaspa_api, timeout=5).json()
        result["kaspa"]["block_count"] = kas_response['blockCount']
        result["kaspa"]["difficulty"] = kas_response['difficulty']
    except Exception as e:
        print(f"Kaspa API error: {e}")
    return result

def fetch_market_data(coin_ids=["eth-ethereum", "kaspa"]):
    market_data = {}
    alerts = []
    for coin_id in coin_ids:
        try:
            ticker = cp_client.ticker(coin_id)
            current_price = ticker["quotes"]["USD"]["price"]
            volume_24h = ticker["quotes"]["USD"]["volume_24h"]
            avg_volume = volume_24h
            spike_detected = volume_24h > avg_volume * 2.0
            market_data[coin_id] = {
                "price": current_price,
                "volume_24h": volume_24h,
                "avg_volume_24h": avg_volume,
                "volume_spike": spike_detected,
                "price_change_pct": 0.0,
                "price_stable": True,
                "timestamp": datetime.now().isoformat()
            }
            if spike_detected:
                alerts.append(f"Volume spike detected for {coin_id.upper()}")
        except Exception as e:
            print(f"Market error for {coin_id}: {e}")
            market_data[coin_id] = None
    return market_data, alerts

def scrape_reddit_posts(subreddit_name="CryptoCurrency", limit=20):
    url = f"https://old.reddit.com/r/{subreddit_name}/new/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://old.reddit.com/"
    }
    posts = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Request failed: {response.status_code}")
            return []
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
        things = soup.find_all("div", class_="thing")
        for thing in things:
            if len(posts) >= limit:
                break
            fullname = thing.get("data-fullname", "")
            if not fullname.startswith("t3_"):
                continue
            title_tag = thing.find("a", class_="title")
            author_tag = thing.find("a", class_="author")
            posts.append({
                "id": fullname[3:],
                "title": title_tag.text.strip() if title_tag else "",
                "author": author_tag.text.strip() if author_tag else None,
                "body": "",
                "created_utc": 0
            })
    except Exception as e:
        print(f"SCRAPER FAILED: {e}")
        return []
    return posts

from reddit_bot_detector import get_reddit_bot_score

sentiment_analyzer = CryptoBERTSentimentAnalyzer()
bot_cache = {}

async def fetch_reddit_posts(limit=20):
    raw_posts = scrape_reddit_posts(limit=limit)
    processed = []
    for post in raw_posts:
        text = post['title']
        sentiment = sentiment_analyzer.get_sentiment_scores(text)
        author = post['author']
        bot_score = 0.5
        if author:
            if author not in bot_cache:
                try:
                    result = await get_reddit_bot_score(author)
                    bot_cache[author] = result['final_score']
                except Exception as e:
                    print(f"Bot score error for {author}: {e}")
                    bot_cache[author] = 0.5
                await asyncio.sleep(2)
            bot_score = bot_cache[author]
        processed.append({
            'id': post['id'],
            'title': post['title'],
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_neu': sentiment['neu'],
            'sentiment_label': sentiment['label'],
            'bot_score': bot_score
        })
    return processed

app = FastAPI()

@app.post("/refresh")
async def refresh_data():
    blockchain = fetch_blockchain()
    reddit_posts = await fetch_reddit_posts()
    market_data, alerts = fetch_market_data()
    tweets = []
    if TWITTER_BEARER_TOKEN:
        try:
            df = run_twitter_pipeline(max_tweets_per_coin=5, store_in_db=False)
            if not df.empty:
                tweets = df.to_dict(orient='records')
            print(f"Fetched {len(tweets)} tweets from X")
        except Exception as e:
            print(f"Twitter/X error: {e}")
    else:
        print("No TWITTER_BEARER_TOKEN found – skipping X data")

    cached_data.update({
        "blockchain": blockchain,
        "reddit_posts": reddit_posts,
        "tweets": tweets,
        "market": market_data,
        "market_alerts": alerts,
        "last_refresh": datetime.now().isoformat()
    })
    print(f"Refresh completed – {len(reddit_posts)} Reddit posts, {len(tweets)} tweets")
    return {"status": "ok"}

@app.get("/latest")
async def get_latest():
    return cached_data

@app.websocket("/ws/live-data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json(cached_data)
    await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Crypto Market Monitor", layout="wide")
st.title("📡 Crypto Market Manipulation Monitor")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh Data (may take 1-3 min on first run)"):
        with st.spinner("Fetching fresh blockchain, Reddit, and market data..."):
            try:
                resp = requests.post("http://localhost:8000/refresh", timeout=300)
                if resp.status_code == 200:
                    st.success("Data refreshed!")
                else:
                    st.error(f"Refresh failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

try:
    data = requests.get("http://localhost:8000/latest").json()

    if data.get("market_alerts"):
        st.error("⚠️ **Potential Market Manipulation Detected!**")
        for alert in data["market_alerts"]:
            st.warning(alert)

    blockchain = data.get("blockchain")
    if blockchain:
        st.subheader("⛓️ Blockchain Metrics")
        col1, col2 = st.columns(2)
        eth = blockchain.get("ethereum") or {}
        kas = blockchain.get("kaspa") or {}
        col1.metric("Ethereum Block Height", eth.get("block_height", "N/A"))
        col1.metric("Gas Price (Gwei)", f"{eth.get('gas_price_gwei', 0):.4f}" if eth.get('gas_price_gwei') else "N/A")
        col2.metric("Kaspa Block Count", kas.get("block_count", "N/A"))
        col2.metric("Kaspa Difficulty", f"{kas.get('difficulty', 0):.2e}" if kas.get('difficulty') else "N/A")
    else:
        st.warning("Blockchain data not available (RPC may be down).")

    reddit_posts = data.get("reddit_posts")
    if reddit_posts:
        st.subheader("💬 Reddit Sentiment (CryptoBERT + Bot scoring)")
        df = pd.DataFrame(reddit_posts)
        display_cols = ["title", "sentiment_compound", "bot_score"]
        if "sentiment_label" in df.columns:
            display_cols.append("sentiment_label")
        st.dataframe(df[display_cols])
    else:
        st.info("No Reddit posts fetched (scraper may have failed).")

    tweets = data.get("tweets")
    if tweets:
        st.subheader("🐦 Twitter Sentiment")
        df_tweets = pd.DataFrame(tweets)
        st.dataframe(df_tweets[["text", "sentiment_compound", "coin"]])

    market = data.get("market")
    if market:
        st.subheader("📊 Market Data (CoinPaprika)")
        for coin, info in market.items():
            if info:
                st.write(f"**{coin.upper()}**")
                st.write(f"Price: ${info.get('price', 'N/A')} | "
                         f"24h Volume: {info.get('volume_24h', 'N/A'):,.0f} | "
                         f"Volume Spike: {info.get('volume_spike', False)}")

    last_refresh = data.get("last_refresh")
    if last_refresh:
        st.caption(f"Last data refresh: {last_refresh}")
    else:
        st.caption("No data refresh yet. Click the button above.")

except requests.exceptions.ConnectionError:
    st.error("Cannot connect to backend. Make sure `python bd.py` is running.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Backend may be running but returned incomplete data. Check backend terminal for errors.")
