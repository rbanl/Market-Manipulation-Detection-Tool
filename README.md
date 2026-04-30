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

Blockchain – Ethereum, Kaspa and Bitcoin.

Market data – CoinPaprika API (price, 24h volume).

Caching – In‑memory cache stores the latest data, bot scores, and sentiment results so subsequent refreshes are fast (first refresh takes ~2‑3 minutes due to heavy bot‑scraping).


Please see my code below:

import time
import random
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

COINGECKO_IDS = {
    "ethereum": "ethereum",
    "kaspa": "kaspa"
}

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
    "manipulation_alerts": [],
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

def fetch_market_data(coin_ids=["eth-ethereum", "kas-kaspa"]):
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

def get_historical_data(coin_name, hours=24):
    gecko_id = COINGECKO_IDS.get(coin_name)
    if not gecko_id:
        return None, None
    url = f"https://api.coingecko.com/api/v3/coins/{gecko_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 1,
        "interval": "hourly"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"CoinGecko error for {coin_name}: {resp.status_code}")
            return None, None
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if len(prices) < 2 or len(volumes) < 2:
            return None, None
        if len(prices) > hours:
            prices = prices[-hours:]
            volumes = volumes[-hours:]
        return prices, volumes
    except Exception as e:
        print(f"CoinGecko fetch failed for {coin_name}: {e}")
        return None, None

def detect_manipulation(coin_name, reddit_posts, hours_window=6):
    prices, volumes = get_historical_data(coin_name)
    if not prices or not volumes:
        return False, "Insufficient historical data from CoinGecko"

    current_price = prices[-1][1]
    current_volume = volumes[-1][1]
    if len(volumes) < 7:
        return False, "Not enough volume history (<7 hours)"
    prev_volumes = [v[1] for v in volumes[-7:-1]]
    avg_volume = sum(prev_volumes) / len(prev_volumes)
    volume_spike = current_volume > avg_volume * 2.0

    price_24h_ago = prices[0][1]
    price_change_pct = abs((current_price - price_24h_ago) / price_24h_ago * 100)
    price_stable = price_change_pct < 1.0

    if not reddit_posts:
        return False, "No Reddit data"
    now = datetime.now().timestamp()
    cutoff = now - hours_window * 3600
    recent_posts = [p for p in reddit_posts if p.get("created_utc", 0) > cutoff]
    if len(recent_posts) < 3:
        return False, f"Insufficient Reddit posts in last {hours_window}h (got {len(recent_posts)})"

    all_sentiments = [p["sentiment_compound"] for p in recent_posts]
    human_sentiments = [p["sentiment_compound"] for p in recent_posts if p.get("bot_score", 0.5) < 0.3]
    if not human_sentiments:
        return False, "No human-only posts in recent window"
    raw_avg = sum(all_sentiments) / len(all_sentiments)
    human_avg = sum(human_sentiments) / len(human_sentiments)
    divergence = abs(raw_avg - human_avg)

    if volume_spike and price_stable and divergence > 0.2:
        reason = (f"Volume spike ({current_volume:.0f} vs avg {avg_volume:.0f}), price stable ({price_change_pct:.1f}%), "
                  f"sentiment divergence {divergence:.2f} (raw={raw_avg:.2f}, human={human_avg:.2f})")
        return True, reason
    return False, f"Conditions not met: spike={volume_spike}, stable={price_stable}, divergence={divergence:.2f}"

def scrape_reddit_posts(subreddit_name="CryptoCurrency", limit=20):
    url = f"https://old.reddit.com/r/{subreddit_name}/new/"
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://old.reddit.com/",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0"
    }
    session.headers.update(headers)
    time.sleep(random.uniform(0.5, 2.5))
    posts = []
    try:
        response = session.get(url, timeout=15)
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
            time_tag = thing.find("time")
            created_utc = 0
            if time_tag and time_tag.get("datetime"):
                try:
                    dt = datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00"))
                    created_utc = int(dt.timestamp())
                except:
                    created_utc = 0
            posts.append({
                "id": fullname[3:],
                "title": title_tag.text.strip() if title_tag else "",
                "author": author_tag.text.strip() if author_tag else None,
                "body": "",
                "created_utc": created_utc
            })
    except Exception as e:
        print(f"SCRAPER FAILED: {e}")
        return []
    time.sleep(random.uniform(1, 2))
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
                await asyncio.sleep(1.5)
            bot_score = bot_cache[author]
        processed.append({
            'id': post['id'],
            'title': post['title'],
            'sentiment_compound': sentiment['compound'],
            'sentiment_pos': sentiment['pos'],
            'sentiment_neg': sentiment['neg'],
            'sentiment_neu': sentiment['neu'],
            'sentiment_label': sentiment['label'],
            'bot_score': bot_score,
            'created_utc': post['created_utc']
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

    manipulation_alerts = []
    for coin_name in ["ethereum", "kaspa"]:
        flag, reason = detect_manipulation(coin_name, reddit_posts)
        if flag:
            manipulation_alerts.append(f"🚨 **Potential manipulation detected for {coin_name.upper()}** – {reason}")
        else:
            print(f"{coin_name}: {reason}")

    cached_data.update({
        "blockchain": blockchain,
        "reddit_posts": reddit_posts,
        "tweets": tweets,
        "market": market_data,
        "market_alerts": alerts,
        "manipulation_alerts": manipulation_alerts,
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
                resp = requests.post("http://localhost:8000/refresh", timeout=600)
                if resp.status_code == 200:
                    st.success("Data refreshed!")
                else:
                    st.error(f"Refresh failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

try:
    data = requests.get("http://localhost:8000/latest").json()

    manip_alerts = data.get("manipulation_alerts")
    if manip_alerts:
        st.error("⚠️ **Market Manipulation Detected!**")
        for alert in manip_alerts:
            st.warning(alert)
    else:
        st.success("✅ No market manipulation detected based on current data.")

    if data.get("market_alerts"):
        with st.expander("🔍 Volume‑only alerts (for reference)"):
            for alert in data["market_alerts"]:
                st.info(alert)

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
        st.subheader("📊 Market Data")
        for coin, info in market.items():
            if info:
                price = info.get('price', None)
                price_display = f"${price:,.2f}" if isinstance(price, (int, float)) else "N/A"
                volume = info.get('volume_24h', 'N/A')
                volume_display = f"{volume:,.0f}" if isinstance(volume, (int, float)) else volume
                spike = info.get('volume_spike', False)
                spike_display = "Present" if spike else "Not Present"

                st.markdown(f"**{coin.upper()}**")
                st.markdown(f"- Price: {price_display}")
                st.markdown(f"- Market Trading 24h Volume ($): {volume_display}")
                st.markdown(f"- Volume Spike: {spike_display}")
                st.markdown("")

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
