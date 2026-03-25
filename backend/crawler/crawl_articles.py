import os
import re
import sys
import json
import time
import logging
from typing import Optional

import requests
from bs4 import BeautifulSoup

# optional deps
try:
    from newspaper import Article as NpArticle
    NEWSPAPER_OK = True
except ImportError:
    NEWSPAPER_OK = False

try:
    from langdetect import detect as lang_detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False


# project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")


# crawl params (edit here)
MIN_WORDS = 500
TARGET_PER_TOPIC = 4
MAX_PER_TOPIC = 6
REQUEST_DELAY = 1.5
TIMEOUT = 15
RETRIES = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}


# basic logger (keep it simple)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# fetch html with retry
def fetch(url: str) -> Optional[BeautifulSoup]:
    for i in range(RETRIES):
        try:
            r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            return BeautifulSoup(r.text, "lxml")
        except Exception as e:
            log.warning(f"retry {i+1}/{RETRIES} failed: {url}")
            time.sleep(2 ** i)
    return None


# quick Vietnamese filter (not perfect but good enough)
VIET_REGEX = re.compile(r"[àáạảãâăđêôơư]", re.I)
COMMON_WORDS = ["và", "của", "trong", "là", "có"]


def is_vietnamese(text: str) -> bool:
    if len(text) < 100:
        return False

    if len(VIET_REGEX.findall(text)) < 20:
        return False

    if sum(w in text.lower() for w in COMMON_WORDS) < 2:
        return False

    if LANGDETECT_OK:
        try:
            return lang_detect(text[:1000]) == "vi"
        except:
            pass

    return True


# remove junk html -> plain text
def clean_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    texts = []
    for el in soup.find_all(["h1", "h2", "p", "li"]):
        txt = el.get_text(strip=True)
        if len(txt) > 20:
            texts.append(txt)

    return "\n".join(texts)


def word_count(text: str) -> int:
    return len(text.split())


# extract article content
def extract(url: str) -> Optional[dict]:
    title, content = "", ""

    # try newspaper first
    if NEWSPAPER_OK:
        try:
            art = NpArticle(url, language="vi")
            art.download()
            art.parse()
            title = art.title
            content = art.text
        except Exception:
            log.debug(f"newspaper failed: {url}")

    # fallback to bs4
    if not content or word_count(content) < 200:
        soup = fetch(url)
        if not soup:
            log.warning(f"failed to fetch: {url}")
            return None

        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else ""

        content = clean_text(soup)

    wc = word_count(content)

    if wc < MIN_WORDS:
        log.info(f"too short ({wc} words): {url}")
        return None

    if not is_vietnamese(content):
        log.info(f"not vietnamese: {url}")
        return None

    log.info(f"ok ({wc} words): {title[:60]}")
    return {
        "title": title,
        "url": url,
        "content": content
    }


# save article
def save(article, topic, idx):
    path = os.path.join(DATASET_DIR, topic)
    os.makedirs(path, exist_ok=True)

    file = os.path.join(path, f"{idx:02d}.json")
    with open(file, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)


# crawl one topic
def crawl(topic, urls, seen):
    results = []

    log.info(f"\n--- topic: {topic} ---")

    for url in urls:
        if len(results) >= MAX_PER_TOPIC:
            break

        if url in seen:
            continue

        seen.add(url)
        time.sleep(REQUEST_DELAY)

        art = extract(url)
        if art:
            art["topic"] = topic
            results.append(art)

    log.info(f"collected {len(results)} articles for {topic}")
    return results


# entry point
def main():
    log.info("start crawl")

    SEED_URLS = {
        "du_lich": [
            "https://vnexpress.net/du-lich-vung-tau-nhung-dieu-can-biet-truoc-khi-den-4560123.html",
        ]
    }

    seen = set()
    all_data = []

    for topic, urls in SEED_URLS.items():
        articles = crawl(topic, urls, seen)

        for i, art in enumerate(articles, 1):
            save(art, topic, i)

        all_data.extend(articles)

    # save merged file
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(os.path.join(DATASET_DIR, "all.json"), "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    log.info(f"done: {len(all_data)} articles")


if __name__ == "__main__":
    main()