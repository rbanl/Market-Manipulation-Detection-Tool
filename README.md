# Market-Manipulation-Detection-Tool
Built a market manipulation detection tool deployed as a web app with a user‑initiated refresh architecture.  
It monitors cryptocurrency markets for potential manipulation by cross‑referencing on‑chain metrics, Reddit sentiment, and Twitter/X sentiment.  

NB: This is a working single‑user prototype. Data is refreshed on demand with a single button click – a deliberate choice that lets you check market conditions right before making a trade, while keeping resource usage low and respecting free API limits.


**What it Detects**  
- **On‑chain volume & price** – Daily on‑chain transaction volume (estimated for Ethereum and Bitcoin, exchange‑traded for Kaspa) is used to compute the NVT ratio. A volume spike with a stable price often signals artificial activity.  
- **Reddit & Twitter sentiment** – Posts from r/CryptoCurrency and recent tweets are analyzed with CryptoBERT, a transformer model fine‑tuned on crypto social media. The system also detects bot accounts on Reddit by scraping user profiles (age, karma, posting patterns) and flags likely bots.  

Tiered Manipulation Alerts

The system detects potential coordinated manipulation by combining three signals within a configurable time window (1 day, 14 days, 30 days):

Volume spike – A significant increase in on‑chain transaction volume (or exchange volume for Kaspa) compared to the historical average for the same window, while the price remains relatively stable (a classic pump‑and‑dump or wash‑trading signature).

Elevated bot activity on Reddit – More than 50% of Reddit posts about a coin within the window originate from accounts that the bot‑detection model scores as highly likely to be automated (bot score > 0.85).

Elevated bot activity on Twitter – Similarly, more than 50% of tweets about the same coin are classified as bots.

These signals are then mapped to three confidence tiers:

Low Confidence – A volume spike is detected and at least one platform (Reddit or Twitter) shows elevated bot activity. This indicates a possible attempt to manipulate sentiment, but the evidence is isolated to a single channel.

Medium Confidence – The same criteria as Low, but both Reddit and Twitter simultaneously exhibit bot‑driven activity during the volume spike. This cross‑platform coordination strongly suggests an organised campaign.

High Confidence – Meets the Medium criteria plus the earliest bot‑driven post on Reddit and the earliest bot‑driven tweet occur within 2.5 hours of each other. The tight timing alignment implies that the same actor (or a closely synchronised group) is orchestrating the activity on both platforms, dramatically increasing the likelihood of a coordinated manipulation attempt.

When a High‑confidence alert fires, the dashboard displays a prominent red warning; Medium‑confidence alerts appear as orange warnings; and Low‑confidence alerts are shown as blue informational notices. This graduated system allows the user to quickly assess the threat level without digging through raw data.


**Architecture (high‑level)**  
The system uses a FastAPI backend with two main endpoints (`/refresh` and `/latest`) and a WebSocket endpoint for live updates. The frontend is a Streamlit dashboard that displays real‑time blockchain metrics price, NVT ratio, $ worth amount of crypto  exchanged within a 24‑hour volume), Reddit and Twitter sentiment tables, and alerts. Data sources include CoinMarketCap for prices, Etherscan V2 for Ethereum on‑chain volume, Blockchair for Bitcoin on‑chain volume, the Kaspa REST API/Explorer for Kaspa stats, and CoinGecko (with file caching) for historical market data. A persistent file cache (`historical_cache.json`) limits API calls to avoid rate limits, while an in‑memory cache stores bot scores and latest data for fast refreshes.  

NB:Network Value to Transactions (NVT) ratio compares a cryptocurrency's total market capitalization to its daily on-chain transaction volume, providing insight into whether the asset is overvalued or undervalued relative to its network usage.

**NLP & Bot Detection**  
Sentiment is powered by CryptoBERT via Hugging Face transformers. Reddit bot scoring uses headless browser scraping of user profiles and caches results. Twitter data is fetched via the Tweepy client and analyzed with the same sentiment pipeline.  

**Key recent additions**  
- Three‑tier bot‑coordination confidence system combining Reddit and Twitter bot activity with volume spikes.  
- On‑chain metrics (NVT ratio, estimated 24‑hour amount moved) for Ethereum, Bitcoin, and Kaspa.  
- Live price display for all three coins sourced from CoinMarketCap.  
- Rate‑limit handling with pacing, retry logic, and file‑based caching for historical data.  
- Extended Twitter sentiment to include Bitcoin alongside Ethereum and Kaspa.


Please see my code below:
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import json
import time
import random
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv("k.env")
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket
import uvicorn
from web3 import Web3
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from coinpaprika.client import Client
from twitter_sentiment import run_twitter_pipeline
import time
import random
import os
import sqlite3
from dotenv import load_dotenv
load_dotenv("k.env")
import asyncio
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket
import uvicorn
from web3 import Web3
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from coinpaprika.client import Client
from twitter_sentiment import run_twitter_pipeline
from datetime import datetime
CMC_API_KEY = os.getenv("CMC_API_KEY")
if not CMC_API_KEY:
    raise ValueError("CMC_API_KEY not set in .env file")

# ---------- Environment & Clients ----------
ETH_RPC_URL = os.getenv("ETH_RPC")
eth_w3 = Web3(Web3.HTTPProvider(ETH_RPC_URL))
BOTOMETER_KEY = os.getenv("BotX_key")
if not ETH_RPC_URL:
    raise ValueError("ETH_RPC not set in .env file")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
BTC_RPC_URL = os.getenv("BTC_RPC")
if not BTC_RPC_URL:
    raise ValueError("BTC_RPC not present")

cp_client = Client()

COINGECKO_IDS = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "kaspa": "kaspa"
}

# ---------- CryptoBERT Sentiment Analyzer ----------
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

# ---------- Database Setup (for Reddit posts) ----------
def init_reddit_db():
    conn = sqlite3.connect('crypto_sentiment.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS reddit_posts
                 (id TEXT PRIMARY KEY,
                  title TEXT,
                  sentiment_compound REAL,
                  bot_score REAL,
                  created_utc INTEGER)''')
    conn.commit()
    conn.close()

def store_reddit_posts(posts):
    conn = sqlite3.connect('crypto_sentiment.db')
    c = conn.cursor()
    for p in posts:
        try:
            c.execute('''INSERT OR REPLACE INTO reddit_posts
                         (id, title, sentiment_compound, bot_score, created_utc)
                         VALUES (?, ?, ?, ?, ?)''',
                      (p['id'], p['title'], p['sentiment_compound'], p['bot_score'], p['created_utc']))
        except Exception as e:
            print(f"DB insert error: {e}")
    conn.commit()
    conn.close()

def get_reddit_posts_since(days_ago):
    """Retrieve reddit posts from the last `days_ago` days from database."""
    cutoff = datetime.now().timestamp() - days_ago * 86400
    conn = sqlite3.connect('crypto_sentiment.db')
    c = conn.cursor()
    c.execute('''SELECT sentiment_compound, bot_score, created_utc FROM reddit_posts WHERE created_utc > ?''', (cutoff,))
    rows = c.fetchall()
    conn.close()
    return [{'sentiment_compound': r[0], 'bot_score': r[1], 'created_utc': r[2]} for r in rows]

# ---------- Global cache ----------
cached_data = {
    "blockchain": None,
    "reddit_posts": [],
    "tweets": [],
    "market": {},
    "market_alerts": [],
    "manipulation_alerts": {},      # existing sentiment divergence alerts
    "bot_coordination_alerts": {}, # NEW: tiered alerts from bot activity
    "last_refresh": None
}
# ---------- Global constants (place near other API URLs) ----------
ETHERSCAN_KEY = os.getenv("ETHERSCAN_API_KEY")   # required
KASPA_API_BASE = "https://api.kaspa.org"

# ---------- NVT helper functions ----------
def fetch_ethereum_nvt(eth_price, eth_market_cap):
    """Returns (nvt_ratio, daily_volume_usd) or (None, None) on failure."""
    try:
        latest_block = eth_w3.eth.get_block('latest')['number']
        num_samples = 30
        start_block = latest_block - num_samples + 1
        total_wei = 0
        for blk_num in range(start_block, latest_block + 1):
            payload = {
                "chainid": 1,
                "module": "proxy",
                "action": "eth_getBlockByNumber",
                "tag": hex(blk_num),
                "boolean": "true",
                "apikey": ETHERSCAN_KEY
            }
            r = requests.get("https://api.etherscan.io/v2/api", params=payload, timeout=10)
            time.sleep(0.3)
            if r.status_code != 200:
                continue
            block = r.json().get("result", {})
            if block and "transactions" in block:
                total_wei += sum(int(tx["value"], 16) for tx in block["transactions"])

        if total_wei == 0:
            return None, None

        avg_wei_per_block = total_wei / num_samples
        daily_blocks = 5760
        daily_eth_transferred = (avg_wei_per_block * daily_blocks) / 1e18
        daily_volume_usd = daily_eth_transferred * eth_price

        if daily_volume_usd > 0:
            nvt = round(eth_market_cap / daily_volume_usd, 2)
            return nvt, daily_volume_usd
        return None, None
    except Exception as e:
        print(f"Etherscan NVT error: {e}")
        return None, None

def fetch_kaspa_nvt(kas_price, kas_market_cap):
    """Returns (nvt_ratio, daily_volume_usd) or (None, None) on failure."""
    try:
        resp = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={
                "ids": "kaspa",
                "vs_currencies": "usd",
                "include_24hr_vol": "true",
            },
            timeout=10
        )
        if resp.status_code != 200:
            return None, None
        data = resp.json().get("kaspa", {})
        daily_volume_usd = data.get("usd_24h_vol")
        if not daily_volume_usd or not kas_market_cap:
            return None, None
        nvt = round(kas_market_cap / daily_volume_usd, 2)
        return nvt, daily_volume_usd
    except Exception as e:
        print(f"Kaspa NVT error: {e}")
        return None, None

def fetch_bitcoin_nvt(btc_price, btc_market_cap):
    """Returns (nvt_ratio, daily_volume_usd) or (None, None) using Blockchair."""
    try:
        resp = requests.get("https://api.blockchair.com/bitcoin/stats", timeout=10)
        if resp.status_code != 200:
            print(f"Blockchair error: {resp.status_code}")
            return None, None

        data = resp.json().get("data", {})
        volume_24h_sats = data.get("volume_24h", 0)
        if volume_24h_sats <= 0:
            print("Blockchair error: volume_24h missing or zero")  # was silent before
            return None, None

        daily_volume_btc = volume_24h_sats / 1e8
        daily_volume_usd = daily_volume_btc * btc_price

        if daily_volume_usd > 0:
            nvt = round(btc_market_cap / daily_volume_usd, 2)
            return nvt, daily_volume_usd
        return None, None
    except Exception as e:
        print(f"Bitcoin error: {e}")
        return None, None

        
def _fetch_cmc_quote(symbol):
    """Fetch current price and market cap from CMC."""
    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"symbol": symbol, "convert": "USD"}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        raise Exception(f"CMC latest quote error for {symbol}: {resp.status_code}")
    data = resp.json()
    coin_array = data["data"][symbol]
    if not coin_array:
        raise Exception(f"No data returned for {symbol}")
    quote = coin_array[0]["quote"]["USD"]
    return {"price": quote["price"], "market_cap": quote["market_cap"]}


# ---------- Main blockchain data fetcher – simplified to NVT & amount moved only ----------
def fetch_blockchain():
    result = {
        "timestamp": datetime.now().isoformat(),
        "ethereum": {
            "price": None,          # <-- added
            "nvt_ratio": None,
            "amount_moved": None
        },
        "kaspa": {
            "price": None,          # <-- added
            "nvt_ratio": None,
            "amount_moved": None
        },
        "bitcoin": {
            "price": None,          # <-- added
            "nvt_ratio": None,
            "amount_moved": None
        }
    }

    # ---------- ETHEREUM ----------
    try:
        eth_data = _fetch_cmc_quote("ETH")
        eth_price = eth_data["price"]
        eth_market_cap = eth_data["market_cap"]
        result["ethereum"]["price"] = round(eth_price, 2)   # <-- added
        nvt, volume = fetch_ethereum_nvt(eth_price, eth_market_cap)
        result["ethereum"]["nvt_ratio"] = nvt
        result["ethereum"]["amount_moved"] = round(volume, 2) if volume else None
    except Exception as e:
        print(f"Ethereum error: {e}")

    # ---------- KASPA ----------
    try:
        kas_data = _fetch_cmc_quote("KAS")
        kas_price = kas_data["price"]
        kas_market_cap = kas_data["market_cap"]
        result["kaspa"]["price"] = round(kas_price, 5)       # <-- added (more decimals: KAS is sub-cent)
        nvt, volume = fetch_kaspa_nvt(kas_price, kas_market_cap)
        result["kaspa"]["nvt_ratio"] = nvt
        result["kaspa"]["amount_moved"] = round(volume, 2) if volume else None
    except Exception as e:
        print(f"Kaspa error: {e}")

    # ---------- BITCOIN ----------
    try:
        btc_data = _fetch_cmc_quote("BTC")
        btc_price = btc_data["price"]
        btc_market_cap = btc_data["market_cap"]
        result["bitcoin"]["price"] = round(btc_price, 2)     # <-- added
        nvt, volume = fetch_bitcoin_nvt(btc_price, btc_market_cap)
        result["bitcoin"]["nvt_ratio"] = nvt
        result["bitcoin"]["amount_moved"] = round(volume, 2) if volume else None
    except Exception as e:
        print(f"Bitcoin error: {e}")

    return result
    
# ---------- Historical data (NOW USING COINCAP – free, no key) ----------
# CoinCap asset IDs (no API key needed)
COINCAP_IDS = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "kaspa": "kaspa"
}

# Per‑refresh cache – CLEAR THIS at the start of every /refresh
_historical_cache = {}

CMC_BASE = "https://pro-api.coinmarketcap.com"
COIN_MAP = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "kaspa": "KAS"
}

HISTORICAL_CACHE_FILE = "historical_cache.json"
CACHE_TTL = 3600  # seconds (1 hour)

def load_hist_cache():
    if os.path.exists(HISTORICAL_CACHE_FILE):
        try:
            with open(HISTORICAL_CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}



def save_hist_cache(cache):
    with open(HISTORICAL_CACHE_FILE, "w") as f:
        json.dump(cache, f)
def get_historical_data(coin_name, days=1):
    """
    Fetch daily price & volume from CoinGecko (free, no key).
    Cached to a JSON file to stay under the 10‑30 req/min limit.
    Returns (prices, volumes) as lists of [timestamp_ms, value] or (None, None).
    """
    gecko_id = COINGECKO_IDS.get(coin_name)
    if not gecko_id:
        return None, None

    cache_key = f"{gecko_id}_{days}"
    cache = load_hist_cache()

    # Return cached data if it's still fresh
    if cache_key in cache:
        entry = cache[cache_key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["prices"], entry["volumes"]

    # Fetch from CoinGecko
    url = f"https://api.coingecko.com/api/v3/coins/{gecko_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily" if days > 1 else "hourly"
    }
    try:
        time.sleep(1.5)   # gentle rate limiting – keeps you under ~30/min
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"CoinGecko error for {coin_name}: {resp.status_code}")
            return None, None
        data = resp.json()
        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])
        if len(prices) < 2 or len(volumes) < 2:
            return None, None

        # Store in cache
        cache[cache_key] = {
            "timestamp": time.time(),
            "prices": prices,
            "volumes": volumes
        }
        save_hist_cache(cache)
        return prices, volumes

    except Exception as e:
        print(f"CoinGecko fetch failed for {coin_name}: {e}")
        return None, None
        
def has_volume_spike(coin_name, window_days):
    """Return True if current volume > 2x average of previous days in the window."""
    prices, volumes = get_historical_data(coin_name, days=window_days)
    if not prices or not volumes or len(volumes) < 3:
        return False
    current_volume = volumes[-1][1]
    prev_volumes = [v[1] for v in volumes[:-1]]
    avg_volume = sum(prev_volumes) / len(prev_volumes)
    return current_volume > avg_volume * 2.0

# ---------- NEW: Bot coordination tiered detection ----------
def compute_bot_coordination(coin_name, window_days, reddit_posts, tweets):
    """
    Returns a dict with 'tier' (None, 'Low', 'Medium', 'High') and 'reason'.
    Uses:
      - volume spike (mandatory)
      - elevated bot activity on Reddit ( >50% of posts with bot_score > 0.85 )
      - elevated bot activity on Twitter ( >50% of tweets with bot_score > 0.85 )
      - timing proximity (if both spikes, earliest timestamps <= 2.5h apart)
    """
    # 1. Volume spike check
    if not has_volume_spike(coin_name, window_days):
        return {"tier": None, "reason": "No volume spike"}

    # 2. Filter posts and tweets within the window
    now = datetime.now().timestamp()
    cutoff = now - window_days * 86400

    reddit_window = [p for p in reddit_posts if p.get('created_utc', 0) > cutoff]
    tweet_window = [t for t in tweets if t.get('created_at')]
    # Convert tweet timestamps (ISO) to Unix
    tweet_window_filtered = []
    for t in tweet_window:
        try:
            ts = datetime.fromisoformat(t['created_at'].replace('Z', '+00:00')).timestamp()
            if ts > cutoff:
                t['_timestamp'] = ts
                tweet_window_filtered.append(t)
        except:
            pass

    # 3. Elevated bot activity on Reddit
    reddit_high_bot = [p for p in reddit_window if p.get('bot_score', 0.5) > 0.85]
    reddit_spike = len(reddit_high_bot) > 0.5 * len(reddit_window) if len(reddit_window) > 0 else False

    # 4. Elevated bot activity on Twitter
    twitter_high_bot = [t for t in tweet_window_filtered if t.get('bot_score', 0.5) > 0.85]
    twitter_spike = len(twitter_high_bot) > 0.5 * len(tweet_window_filtered) if len(tweet_window_filtered) > 0 else False

    # 5. Determine tier
    if not (reddit_spike or twitter_spike):
        return {"tier": None, "reason": "No elevated bot activity on Reddit or Twitter"}

    # Volume spike already true, so we have at least one spike
    tier = "Low"
    reason_parts = []
    if reddit_spike:
        reason_parts.append("Reddit bot spike (>50% posts with bot_score>0.85)")
    if twitter_spike:
        reason_parts.append("Twitter bot spike (>50% tweets with bot_score>0.85)")

    if reddit_spike and twitter_spike:
        tier = "Medium"
        reason_parts.append("both platforms")

        # Check timing proximity
        earliest_reddit = min((p['created_utc'] for p in reddit_high_bot), default=None)
        earliest_twitter = min((t['_timestamp'] for t in twitter_high_bot), default=None)
        if earliest_reddit and earliest_twitter:
            diff_seconds = abs(earliest_reddit - earliest_twitter)
            if diff_seconds <= 2.5 * 3600:  # 2.5 hours
                tier = "High"
                reason_parts.append("timing within 2.5h")

    reason = f"Volume spike + {' + '.join(reason_parts)}"
    return {"tier": tier, "reason": reason}

# ---------- Reddit scraper (unchanged) ----------
def scrape_reddit_rss(subreddit="CryptoCurrency", limit=20):
    """Fetch latest posts via Reddit's public RSS feed."""
    url = f"https://www.reddit.com/r/{subreddit}/new/.rss?limit={limit}"
    headers = {"User-Agent": "RSS Reader/1.0 (CryptoMonitor)"}  # descriptive, benign
    posts = []

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"RSS error: {resp.status_code}")
            return posts

        root = ET.fromstring(resp.content)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall('atom:entry', ns):
            # Title
            title_elem = entry.find('atom:title', ns)
            title = title_elem.text if title_elem is not None else ""

            # Author (name only)
            author_elem = entry.find('atom:author/atom:name', ns)
            author = author_elem.text if author_elem is not None else None

            # Post ID – extract from the <id> tag, which is a URL like
            # https://www.reddit.com/r/CryptoCurrency/comments/abc123/...
            id_elem = entry.find('atom:id', ns)
            post_id = None
            if id_elem is not None:
                # Example: "https://www.reddit.com/r/CryptoCurrency/comments/1d35abc/some_title/"
                parts = id_elem.text.split('/')
                # The ID is after 'comments'
                if 'comments' in parts:
                    idx = parts.index('comments')
                    if idx + 1 < len(parts):
                        post_id = parts[idx + 1]

            # Timestamp
            updated = entry.find('atom:updated', ns).text
            created_utc = 0
            if updated:
                try:
                    dt = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    created_utc = int(dt.timestamp())
                except:
                    pass

            if len(posts) < limit:
                posts.append({
                    'id': post_id,
                    'title': title,
                    'author': author,
                    'created_utc': created_utc,
                    'body': ''  # not provided by RSS, kept for compatibility
                })
    except Exception as e:
        print(f"RSS scraper error: {e}")

    return posts
# ---------- Reddit processing with bot scoring ----------
from reddit_bot_detector import get_reddit_bot_score

sentiment_analyzer = CryptoBERTSentimentAnalyzer()
bot_cache = {}

async def fetch_reddit_posts(limit=20):
    raw_posts = scrape_reddit_rss(limit=limit)
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

# ---------- FastAPI app ----------
app = FastAPI()

@app.on_event("startup")
def startup_event():
    init_reddit_db()

    
def fetch_market_data(coin_ids=["btc-bitcoin", "eth-ethereum", "kas-kaspa"]):
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

def detect_manipulation(coin_name, window_days, reddit_posts_list=None):
    prices, volumes = get_historical_data(coin_name, days=window_days)
    if not prices or not volumes:
        return False, f"Insufficient historical data for {window_days}d"

    current_price = prices[-1][1]
    current_volume = volumes[-1][1]
    if len(volumes) < 3:
        return False, "Not enough volume history"
    prev_volumes = [v[1] for v in volumes[:-1]]
    avg_volume = sum(prev_volumes) / len(prev_volumes)
    volume_spike = current_volume > avg_volume * 2.0

    start_price = prices[0][1]
    price_change_pct = abs((current_price - start_price) / start_price * 100)
    stable_threshold = 1.0 if window_days <= 1 else 5.0
    price_stable = price_change_pct < stable_threshold

    if reddit_posts_list is None:
        posts = get_reddit_posts_since(window_days)
    else:
        posts = reddit_posts_list
    if len(posts) < 3:
        return False, f"Insufficient Reddit posts in last {window_days}d"

    all_sentiments = [p['sentiment_compound'] for p in posts]
    human_sentiments = [p['sentiment_compound'] for p in posts if p.get('bot_score', 0.5) < 0.3]
    if not human_sentiments:
        return False, "No human-only posts in window"

    raw_avg = sum(all_sentiments) / len(all_sentiments)
    human_avg = sum(human_sentiments) / len(human_sentiments)
    divergence = abs(raw_avg - human_avg)

    if volume_spike and price_stable and divergence > 0.2:
        reason = (f"Volume spike ({current_volume:.0f} vs avg {avg_volume:.0f}), price stable ({price_change_pct:.1f}% over {window_days}d), "
                  f"sentiment divergence {divergence:.2f} (raw={raw_avg:.2f}, human={human_avg:.2f})")
        return True, reason
    return False, f"Conditions not met: spike={volume_spike}, stable={price_stable}, divergence={divergence:.2f}"

@app.post("/refresh")
async def refresh_data():
    _historical_cache.clear()
    # 1. Blockchain
    blockchain = fetch_blockchain()
    # 2. Reddit (fresh scrape + bot scores)
    reddit_posts = await fetch_reddit_posts()
    # Store in DB for future long-window queries
    store_reddit_posts(reddit_posts)
    # 3. Market data (display only)
    market_data, alerts = fetch_market_data()
    # 4. Twitter
    tweets = []
    if TWITTER_BEARER_TOKEN:
        try:
            df = run_twitter_pipeline(max_tweets_per_coin=10, store_in_db=False)
            if not df.empty:
                tweets = df.to_dict(orient='records')
            print(f"Fetched {len(tweets)} tweets from X")
        except Exception as e:
            print(f"Twitter/X error: {e}")
    else:
        print("No TWITTER_BEARER_TOKEN found – skipping X data")

    # 5. Existing sentiment divergence alerts (unchanged)
    windows = [1, 14, 30]   # days
    manipulation_alerts = {}
    for window in windows:
        alerts_for_window = []
        for coin_name in ["bitcoin", "ethereum", "kaspa"]:
            flag, reason = detect_manipulation(coin_name, window)
            if flag:
                alerts_for_window.append(f"🚨 **Potential manipulation detected for {coin_name.upper()} ({window}d window)** – {reason}")
            else:
                print(f"{coin_name} ({window}d): {reason}")
        manipulation_alerts[f"{window}d"] = alerts_for_window

    # 6. NEW: Bot coordination tiered alerts
    bot_coordination = {}
    for window in windows:
        window_key = f"{window}d"
        bot_coordination[window_key] = {}
        for coin_name in ["bitcoin", "ethereum", "kaspa"]:
            # Need reddit_posts (current) and tweets (current) - but for longer windows we might want to use DB? 
            # For simplicity, we use the posts and tweets fetched in this refresh (which only contain recent items).
            # To be more accurate, we could query DB for longer windows, but that would add complexity.
            # We'll use the in-memory lists; for 14d/30d windows, they may not have enough data, but it's a start.
            result = compute_bot_coordination(coin_name, window, reddit_posts, tweets)
            if result["tier"] is not None:
                bot_coordination[window_key][coin_name] = result
    # Update global cache
    cached_data.update({
        "blockchain": blockchain,
        "reddit_posts": reddit_posts,
        "tweets": tweets,
        "market": market_data,
        "market_alerts": alerts,
        "manipulation_alerts": manipulation_alerts,
        "bot_coordination_alerts": bot_coordination,
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


#FRONTEND CODE:
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Crypto Market Monitor", layout="wide")
st.title("📡 Crypto Market Manipulation Monitor")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh Data (may take 1-3 min on first run)"):
        with st.spinner("Fetching fresh crypto data from Reddit and X"):
            try:
                resp = requests.post("http://localhost:8000/refresh", timeout=900)
                if resp.status_code == 200:
                    st.success("Data refreshed!")
                else:
                    st.error(f"Refresh failed: {resp.status_code}")
            except Exception as e:
                st.error(f"Connection error: {e}")

try:
    data = requests.get("http://localhost:8000/latest").json()

    # ===== BOT COORDINATION TIERED ALERTS =====
    bot_alerts = data.get("bot_coordination_alerts", {})
    if bot_alerts:
        st.subheader("🤖 Bot Coordination Confidence Alerts")
        for window_key, coins_dict in bot_alerts.items():
            window_label = {"1d": "1 day", "14d": "14 days", "30d": "30 days"}.get(window_key, window_key)
            for coin_name, alert_info in coins_dict.items():
                tier = alert_info.get("tier")
                reason = alert_info.get("reason", "")
                if tier == "Low":
                    st.info(f"**{coin_name.upper()} ({window_label})** – 🟡 **Low confidence**: {reason}")
                elif tier == "Medium":
                    st.warning(f"**{coin_name.upper()} ({window_label})** – 🟠 **Medium confidence**: {reason}")
                elif tier == "High":
                    st.error(f"**{coin_name.upper()} ({window_label})** – 🔴 **High confidence**: {reason}")
    else:
        st.info("No bot coordination alerts at this time.")

    # ===== SENTIMENT DIVERGENCE ALERTS =====
    manip_alerts_dict = data.get("manipulation_alerts", {})
    coins = ["bitcoin", "ethereum", "kaspa"]
    windows = ["1d", "14d", "30d"]
    window_labels = {"1d": "1 day", "14d": "14 days", "30d": "30 days"}

    any_alert = False
    for coin in coins:
        for window in windows:
            alert_list = manip_alerts_dict.get(window, [])
            if any(coin.upper() in a for a in alert_list):
                any_alert = True
                break
        if any_alert:
            break

    if any_alert:
        st.subheader("📉 Sentiment Divergence Alerts")
        st.error("⚠️ **Market Manipulation Detected (Sentiment Divergence)**")
        for window in windows:
            alert_list = manip_alerts_dict.get(window, [])
            if alert_list:
                st.subheader(f"Time window: {window_labels[window]}")
                for alert in alert_list:
                    st.warning(alert)
    else:
        all_windows_str = ", ".join(window_labels[w] for w in windows)
        all_coins_str = ", ".join(c.capitalize() for c in coins)
        st.success(f"✅ No sentiment divergence alerts for {all_coins_str} across the {all_windows_str} time windows.")

    st.subheader("🔍 Detailed Check – By Coin & Time Window (Sentiment Divergence)")
    for coin in coins:
        with st.expander(f"**{coin.upper()}**", expanded=False):
            for window in windows:
                alert_list = manip_alerts_dict.get(window, [])
                coin_alert = next((a for a in alert_list if coin.upper() in a), None)
                if coin_alert:
                    reason = coin_alert.split("–", 1)[-1].strip()
                    st.warning(f"**{window_labels[window]}** → ⚠️ Manipulation detected: {reason}")
                else:
                    st.success(f"**{window_labels[window]}** → ✅ No manipulation detected")
# ===== ON-CHAIN METRICS =====
    blockchain = data.get("blockchain")
    if blockchain:
        st.subheader("⛓️ Key On-Chain Metrics")

        def format_metric(value, decimals=2, default="N/A"):
            """Safely format a numeric metric for display."""
            if value is None or value == "N/A":
                return default
            try:
                v = float(value)
                return f"{v:,.{decimals}f}"
            except (ValueError, TypeError):
                return str(value)

        # ---------- ETHEREUM ----------
        st.markdown("### 🔹 Ethereum (ETH)")
        eth = blockchain.get("ethereum") or {}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price (USD)", format_metric(eth.get("price"), decimals=2))
        with col2:
            st.metric("NVT Ratio", format_metric(eth.get("nvt_ratio"), decimals=2))
        with col3:
            st.metric("Amount Moved (USD)", format_metric(eth.get("amount_moved"), decimals=2))

        # ---------- KASPA ----------
        st.markdown("### 🔹 Kaspa (KAS)")
        kas = blockchain.get("kaspa") or {}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price (USD)", format_metric(kas.get("price"), decimals=5))
        with col2:
            st.metric("NVT Ratio", format_metric(kas.get("nvt_ratio"), decimals=2))
        with col3:
            st.metric("Amount Moved (USD)", format_metric(kas.get("amount_moved"), decimals=2))

        # ---------- BITCOIN ----------
        st.markdown("### 🔹 Bitcoin (BTC)")
        btc = blockchain.get("bitcoin") or {}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Price (USD)", format_metric(btc.get("price"), decimals=2))
        with col2:
            st.metric("NVT Ratio", format_metric(btc.get("nvt_ratio"), decimals=2))
        with col3:
            st.metric("Amount Moved (USD)", format_metric(btc.get("amount_moved"), decimals=2))
    else:
        st.warning("Blockchain data not available")



    # ----- Tweets -----
    tweets = data.get("tweets")
    if tweets:
        st.subheader("🐦 Twitter Sentiment")
        df_tweets = pd.DataFrame(tweets)
        if "text" in df_tweets.columns:
            df_tweet_display = df_tweets[["text", "sentiment_compound", "coin"]].rename(columns={
                "text": "Tweet Text",
                "sentiment_compound": "Confidence of Sentiment Score",
                "coin": "Coin"
            })
            st.dataframe(df_tweet_display)
            st.caption("💡 **Confidence of Sentiment Score** (0–1): Confidence in the sentiment classification.")
        else:
            st.dataframe(df_tweets)

    # ----- Volume‑only alerts (reference) -----
    if data.get("market_alerts"):
        with st.expander("🔍 Volume‑only alerts (for reference)"):
            for alert in data["market_alerts"]:
                st.info(alert)

    # ----- Market data (CoinPaprika) -----
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

    # ----- Reddit posts -----
reddit_posts = data.get("reddit_posts")
if reddit_posts:
    st.subheader("💬 Reddit Sentiment")
    df = pd.DataFrame(reddit_posts)
    column_rename = {
        "title": "Post Title",
        "author": "Author",
        "sentiment_compound": "Confidence of Sentiment Score",
        "bot_score": "Bot Likelihood",
        "sentiment_label": "Sentiment"
    }
    df_display = df.rename(columns=column_rename)
    display_cols = [col for col in ["Post Title", "Author", "Confidence of Sentiment Score", "Bot Likelihood", "Sentiment"] if col in df_display.columns]
    st.dataframe(df_display[display_cols])
    st.caption("💡 **Confidence of Sentiment Score** (0–1): How confident the model is about its prediction. "
               "**Bot Likelihood** (0–1): Probability the account is a bot (0 = human, 1 = bot). "
               "**Sentiment**: Bullish, Neutral, or Bearish.")
else:
    st.info("No Reddit posts fetched (scraper may have failed).")
