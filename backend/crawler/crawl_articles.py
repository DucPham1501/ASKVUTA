"""
backend/crawler/crawl_articles.py
----------------------------------
Thu thập ~30 bài viết tiếng Việt về Vũng Tàu, phân loại theo chủ đề.

Chiến lược:
  1. Mỗi chủ đề có danh sách URL bài viết được kiểm duyệt thủ công
     từ các trang báo/du lịch uy tín tiếng Việt.
  2. Bổ sung bằng DuckDuckGo Lite search (không cần API key).
  3. Dùng newspaper3k để extract nội dung; fallback BeautifulSoup.
  4. Lọc bài viết tiếng Việt, đủ dài (≥500 từ).
  5. Lưu data/dataset/topic/01.json … data/dataset/topic/NN.json

Chạy (từ project root):
    pip install requests beautifulsoup4 lxml newspaper3k langdetect
    python backend/crawler/crawl_articles.py

Output:
    data/dataset/du_lich/01.json
    data/dataset/dac_san/01.json
    ... (tổng ~30 file)
"""

import json
import logging
import os
import re
import sys
import time
from typing import Optional
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# newspaper3k (optional)
try:
    from newspaper import Article as NpArticle
    NEWSPAPER_OK = True
except ImportError:
    NEWSPAPER_OK = False

# langdetect (optional – dùng để xác nhận tiếng Việt)
try:
    from langdetect import detect as lang_detect
    LANGDETECT_OK = True
except ImportError:
    LANGDETECT_OK = False

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))        # backend/crawler/
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))     # project root

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(_SCRIPT_DIR, "crawl_articles.log"), encoding="utf-8"
        ),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
DATASET_DIR     = os.path.join(_PROJECT_ROOT, "data", "dataset")
MIN_WORDS       = 500        # bỏ bài quá ngắn
TARGET_PER_TOPIC = 4         # số bài tối thiểu mỗi chủ đề
MAX_PER_TOPIC    = 6         # số bài tối đa mỗi chủ đề
REQUEST_DELAY    = 1.5       # giây giữa các request
REQUEST_TIMEOUT  = 15
MAX_RETRIES      = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": "https://www.google.com/",
}

# ---------------------------------------------------------------------------
# Danh sách URL nguồn đã được kiểm duyệt cho từng chủ đề
# Ưu tiên: bài dài, nội dung thuần Việt, không cần đăng nhập
# ---------------------------------------------------------------------------
SEED_URLS: dict[str, list[str]] = {

    "du_lich": [
        "https://vnexpress.net/du-lich-vung-tau-nhung-dieu-can-biet-truoc-khi-den-4560123.html",
        "https://vntrip.vn/cam-nang/du-lich-vung-tau",
        "https://mytour.vn/location/114-vung-tau.html",
        "https://ivivu.com/blog/du-lich/kinh-nghiem-du-lich-vung-tau/",
        "https://dulichvietnam.com.vn/du-lich-vung-tau.html",
        "https://www.baobariavungtau.com.vn/du-lich/",
        "https://vnexpress.net/hanh-trinh/kham-pha/vung-tau",
        "https://dantri.com.vn/du-lich/vung-tau.htm",
    ],

    "dia_diem_du_lich": [
        "https://vntrip.vn/cam-nang/dia-diem-du-lich-vung-tau",
        "https://ivivu.com/blog/du-lich/dia-diem-du-lich-vung-tau/",
        "https://mytour.vn/16898-nhung-dia-diem-du-lich-vung-tau-dep-nhat.html",
        "https://dulichvietnam.com.vn/dia-diem-du-lich-vung-tau.html",
        "https://traveloka.com/vi-vn/explore/destination/vung-tau/39",
        "https://www.tripadvisor.com.vn/Tourism-g303944-Vung_Tau_Ba_Ria_Vung_Tau_Province-Vacations.html",
        "https://klook.com/vi/blog/dia-diem-du-lich-vung-tau/",
    ],

    "dac_san": [
        "https://vntrip.vn/cam-nang/dac-san-vung-tau",
        "https://ivivu.com/blog/am-thuc/dac-san-vung-tau/",
        "https://mytour.vn/17200-dac-san-vung-tau.html",
        "https://dulichvietnam.com.vn/am-thuc-vung-tau.html",
        "https://foody.vn/vung-tau",
        "https://vnexpress.net/du-lich/am-thuc/vung-tau",
        "https://reviewvilla.vn/dac-san-vung-tau/",
        "https://www.baobariavungtau.com.vn/am-thuc/",
    ],

    "bai_bien": [
        "https://vntrip.vn/cam-nang/bai-bien-vung-tau",
        "https://ivivu.com/blog/du-lich/bai-bien-vung-tau/",
        "https://mytour.vn/16900-bai-bien-vung-tau.html",
        "https://dulichvietnam.com.vn/bai-bien-vung-tau.html",
        "https://vnexpress.net/du-lich/kham-pha/bai-bien-vung-tau",
        "https://reviewvilla.vn/bai-bien-dep-o-vung-tau/",
        "https://baothanhhoa.vn/du-lich/bai-tam-vung-tau",
    ],

    "danh_lam_thang_canh": [
        "https://vntrip.vn/cam-nang/danh-lam-thang-canh-vung-tau",
        "https://ivivu.com/blog/du-lich/danh-lam-thang-canh-vung-tau/",
        "https://mytour.vn/17202-canh-dep-vung-tau.html",
        "https://dulichvietnam.com.vn/canh-quan-vung-tau.html",
        "https://www.baobariavungtau.com.vn/van-hoa/di-tich-lich-su/",
        "https://dantri.com.vn/du-lich/danh-lam-thang-canh-vung-tau.htm",
    ],

    "lich_su": [
        "https://vi.wikipedia.org/wiki/V%C5%A9ng_T%C3%A0u",
        "https://vi.wikipedia.org/wiki/T%E1%BB%89nh_B%C3%A0_R%E1%BB%8Ba_%E2%80%93_V%C5%A9ng_T%C3%A0u",
        "https://www.baobariavungtau.com.vn/lich-su-van-hoa/",
        "https://vnexpress.net/du-lich/kham-pha/lich-su-vung-tau",
        "https://thethaovanhoa.vn/xa-hoi/lich-su-vung-tau.htm",
        "https://dantri.com.vn/su-kien/lich-su-vung-tau.htm",
        "https://baobariavungtau.com.vn/chinh-tri/vung-tau-qua-cac-thoi-ky-lich-su",
    ],

    "van_hoa_le_hoi": [
        "https://vntrip.vn/cam-nang/le-hoi-vung-tau",
        "https://ivivu.com/blog/du-lich/le-hoi-vung-tau/",
        "https://mytour.vn/17201-le-hoi-vung-tau.html",
        "https://www.baobariavungtau.com.vn/van-hoa/le-hoi/",
        "https://dulichvietnam.com.vn/van-hoa-le-hoi-vung-tau.html",
        "https://vnexpress.net/du-lich/van-hoa/le-hoi-vung-tau",
        "https://thethaovanhoa.vn/van-hoa/le-hoi-vung-tau.htm",
    ],

    "kinh_te": [
        "https://vi.wikipedia.org/wiki/V%C5%A9ng_T%C3%A0u#Kinh_t%E1%BA%BF",
        "https://www.baobariavungtau.com.vn/kinh-te/",
        "https://vnexpress.net/kinh-doanh/vung-tau",
        "https://dantri.com.vn/kinh-doanh/vung-tau.htm",
        "https://cafef.vn/vung-tau.chn",
        "https://vneconomy.vn/vung-tau.htm",
        "https://baobariavungtau.com.vn/kinh-te/can-tho-tp-ho-chi-minh-vung-tau",
    ],

    "kinh_nghiem_du_lich": [
        "https://vntrip.vn/cam-nang/kinh-nghiem-du-lich-vung-tau",
        "https://ivivu.com/blog/du-lich/kinh-nghiem-du-lich-vung-tau-tu-a-z/",
        "https://mytour.vn/16897-kinh-nghiem-du-lich-vung-tau.html",
        "https://dulichvietnam.com.vn/kinh-nghiem-du-lich-vung-tau.html",
        "https://reviewvilla.vn/kinh-nghiem-du-lich-vung-tau/",
        "https://vnexpress.net/du-lich/kham-pha/kinh-nghiem-du-lich-vung-tau",
        "https://blogdulich.net/kinh-nghiem-du-lich-vung-tau",
        "https://traveloka.com/vi-vn/explore/destination/kinh-nghiem-du-lich-vung-tau/",
    ],
}

# Câu truy vấn DuckDuckGo bổ sung nếu chưa đủ bài
FALLBACK_QUERIES: dict[str, list[str]] = {
    "du_lich":               ["du lịch Vũng Tàu tất tần tật", "kinh nghiệm đi Vũng Tàu"],
    "dia_diem_du_lich":      ["địa điểm du lịch Vũng Tàu nổi tiếng", "địa điểm check-in Vũng Tàu"],
    "dac_san":               ["đặc sản Vũng Tàu ngon", "món ăn nổi tiếng Vũng Tàu"],
    "bai_bien":              ["bãi biển Vũng Tàu đẹp", "bãi tắm ở Vũng Tàu"],
    "danh_lam_thang_canh":   ["danh lam thắng cảnh Vũng Tàu", "cảnh đẹp Vũng Tàu"],
    "lich_su":               ["lịch sử Vũng Tàu", "lịch sử Bà Rịa Vũng Tàu"],
    "van_hoa_le_hoi":        ["lễ hội Vũng Tàu", "văn hóa Vũng Tàu truyền thống"],
    "kinh_te":               ["kinh tế Vũng Tàu phát triển", "cảng biển Vũng Tàu kinh tế"],
    "kinh_nghiem_du_lich":   ["kinh nghiệm du lịch Vũng Tàu từ A đến Z", "cẩm nang Vũng Tàu"],
}

# ---------------------------------------------------------------------------
# HTTP session với retry
# ---------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s

SESSION = make_session()


def fetch(url: str, retries: int = MAX_RETRIES) -> Optional[BeautifulSoup]:
    """Tải trang HTML, trả về BeautifulSoup hoặc None."""
    for attempt in range(1, retries + 1):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.HTTPError as e:
            log.warning(f"HTTP {e.response.status_code} – {url}")
            return None
        except requests.exceptions.ConnectionError as e:
            log.warning(f"Kết nối thất bại (lần {attempt}/{retries}) – {url}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            log.warning(f"Timeout (lần {attempt}/{retries}) – {url}")
            if attempt < retries:
                time.sleep(2)
        except Exception as e:
            log.warning(f"Lỗi khác – {url}: {e}")
            return None
    return None

# ---------------------------------------------------------------------------
# Phát hiện tiếng Việt
# ---------------------------------------------------------------------------
VIET_CHARS = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)
VIET_WORDS = ["và", "của", "các", "những", "trong", "với", "được", "là", "có", "đến", "tại", "cho", "này", "một"]


def is_vietnamese(text: str) -> bool:
    """Kiểm tra văn bản có phải tiếng Việt không."""
    if not text or len(text) < 100:
        return False
    # Đếm ký tự có dấu đặc trưng tiếng Việt
    viet_count = len(VIET_CHARS.findall(text[:2000]))
    if viet_count < 20:
        return False
    # Kiểm tra từ phổ biến
    lower = text[:500].lower()
    word_hits = sum(1 for w in VIET_WORDS if f" {w} " in lower)
    if word_hits < 2:
        return False
    # Xác nhận bằng langdetect nếu có
    if LANGDETECT_OK:
        try:
            return lang_detect(text[:1000]) == "vi"
        except Exception:
            pass
    return True

# ---------------------------------------------------------------------------
# Làm sạch văn bản
# ---------------------------------------------------------------------------
NOISE_SELECTORS = [
    "nav", "footer", "header", "aside", "script", "style", "noscript",
    "form", "button", "input", "iframe", "figure > figcaption",
    "[class*='menu']", "[class*='navbar']", "[class*='sidebar']",
    "[class*='advertisement']", "[class*='ads']", "[class*='banner']",
    "[class*='popup']", "[class*='modal']", "[class*='related']",
    "[class*='recommend']", "[class*='social']", "[class*='share']",
    "[class*='comment']", "[class*='breadcrumb']", "[class*='pagination']",
    "[class*='widget']", "[class*='cookie']", "[class*='subscribe']",
    "[class*='newsletter']", "[class*='tag-list']", "[class*='author-bio']",
    "[id*='comment']", "[id*='sidebar']", "[id*='footer']", "[id*='header']",
    "[id*='menu']", "[id*='ads']",
]

CONTENT_SELECTORS = [
    "article .entry-content",
    "article .post-content",
    "article .article-content",
    "article .article-body",
    ".entry-content",
    ".post-content",
    ".article-content",
    ".article-body",
    ".content-detail",
    ".detail-content",
    ".news-content",
    ".post-body",
    ".single-content",
    ".story-body",
    ".cms-body",
    "article",
    "main",
]


def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    """Xoá các phần tử nhiễu khỏi soup."""
    for selector in NOISE_SELECTORS:
        try:
            for el in soup.select(selector):
                el.decompose()
        except Exception:
            pass
    return soup


def soup_to_text(soup: BeautifulSoup) -> str:
    """
    Chuyển soup thành văn bản thuần, giữ cấu trúc đoạn và tiêu đề.
    """
    soup = clean_soup(soup)

    # Tìm container bài viết
    container = None
    for sel in CONTENT_SELECTORS:
        try:
            container = soup.select_one(sel)
            if container and len(container.get_text(strip=True)) > 300:
                break
        except Exception:
            continue
    if not container:
        container = soup.body or soup

    lines = []
    seen_lines: set[str] = set()

    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        if not isinstance(el, Tag):
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if not text or len(text) < 15:
            continue
        # Loại bỏ đoạn trùng lặp
        normalized = re.sub(r"\s+", " ", text.lower())
        if normalized in seen_lines:
            continue
        seen_lines.add(normalized)

        if el.name in ("h1", "h2"):
            lines.append(f"\n## {text}\n")
        elif el.name in ("h3", "h4"):
            lines.append(f"\n### {text}\n")
        elif el.name == "blockquote":
            lines.append(f"> {text}")
        elif el.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return "\n".join(lines).strip()


def count_words(text: str) -> int:
    return len(text.split())

# ---------------------------------------------------------------------------
# Trích xuất bài viết: newspaper3k → BeautifulSoup fallback
# ---------------------------------------------------------------------------

def extract_article(url: str) -> Optional[dict]:
    """
    Trích xuất title + content từ URL.
    Trả về dict hoặc None nếu thất bại / không đủ chất lượng.
    """
    title = ""
    content = ""

    # --- Thử newspaper3k trước ---
    if NEWSPAPER_OK:
        try:
            art = NpArticle(url, language="vi", fetch_images=False, request_timeout=REQUEST_TIMEOUT)
            art.download()
            art.parse()
            title   = (art.title or "").strip()
            content = (art.text  or "").strip()
        except Exception as e:
            log.debug(f"newspaper3k thất bại cho {url}: {e}")

    # --- Fallback BeautifulSoup ---
    if not content or count_words(content) < MIN_WORDS // 2:
        soup = fetch(url)
        if not soup:
            return None
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else ""
        bs_content = soup_to_text(soup)
        # Dùng nội dung dài hơn
        if count_words(bs_content) > count_words(content):
            content = bs_content

    if not content:
        log.debug(f"Không lấy được nội dung: {url}")
        return None

    # Kiểm tra tiếng Việt
    if not is_vietnamese(content):
        log.debug(f"Không phải tiếng Việt: {url}")
        return None

    # Kiểm tra độ dài
    word_count = count_words(content)
    if word_count < MIN_WORDS:
        log.debug(f"Bài quá ngắn ({word_count} từ): {url}")
        return None

    log.info(f"  ✓ '{title[:60]}' ({word_count} từ)")
    return {"title": title, "url": url, "content": content}

# ---------------------------------------------------------------------------
# DuckDuckGo Lite search
# ---------------------------------------------------------------------------

def duckduckgo_search(query: str, max_results: int = 10) -> list[str]:
    """
    Tìm kiếm qua DuckDuckGo Lite HTML (không cần API key).
    Trả về danh sách URL kết quả.
    """
    urls = []
    try:
        search_url = "https://lite.duckduckgo.com/lite/"
        data = {"q": query, "kl": "vn-vi"}
        resp = SESSION.post(search_url, data=data, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.select("a.result-link, a[href*='//']"):
            href = a.get("href", "")
            if not href or href.startswith("//duckduckgo") or "duckduckgo.com" in href:
                continue
            if href.startswith("http"):
                urls.append(href)
            if len(urls) >= max_results:
                break
    except Exception as e:
        log.warning(f"DuckDuckGo search thất bại '{query}': {e}")
    return urls


# ---------------------------------------------------------------------------
# Lưu JSON
# ---------------------------------------------------------------------------

def save_article(article: dict, topic: str, index: int) -> str:
    """Lưu bài viết vào dataset/topic/NN.json. Trả về đường dẫn file."""
    topic_dir = os.path.join(DATASET_DIR, topic)
    os.makedirs(topic_dir, exist_ok=True)
    filename = f"{index:02d}.json"
    filepath = os.path.join(topic_dir, filename)
    payload = {
        "title":   article["title"],
        "url":     article["url"],
        "topic":   topic,
        "content": article["content"],
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return filepath

# ---------------------------------------------------------------------------
# Main crawler
# ---------------------------------------------------------------------------

def crawl_topic(topic: str, seen_urls: set[str]) -> list[dict]:
    """
    Crawl một chủ đề: seed URLs → DuckDuckGo fallback nếu chưa đủ.
    Trả về danh sách bài viết đã thu thập.
    """
    collected: list[dict] = []
    candidate_urls: list[str] = list(SEED_URLS.get(topic, []))

    log.info(f"\n{'='*55}")
    log.info(f"  Chủ đề: {topic}  (mục tiêu: {TARGET_PER_TOPIC}–{MAX_PER_TOPIC} bài)")
    log.info(f"{'='*55}")

    def try_url(url: str) -> bool:
        """Thử crawl một URL, trả về True nếu thành công."""
        if url in seen_urls:
            return False
        seen_urls.add(url)
        time.sleep(REQUEST_DELAY)
        art = extract_article(url)
        if art:
            art["topic"] = topic
            collected.append(art)
            return True
        return False

    # --- Bước 1: Seed URLs ---
    for url in candidate_urls:
        if len(collected) >= MAX_PER_TOPIC:
            break
        try_url(url)

    # --- Bước 2: DuckDuckGo nếu chưa đủ ---
    if len(collected) < TARGET_PER_TOPIC:
        queries = FALLBACK_QUERIES.get(topic, [])
        for query in queries:
            if len(collected) >= TARGET_PER_TOPIC:
                break
            log.info(f"  Tìm kiếm DuckDuckGo: '{query}'")
            time.sleep(REQUEST_DELAY)
            search_results = duckduckgo_search(query, max_results=8)
            for url in search_results:
                if len(collected) >= MAX_PER_TOPIC:
                    break
                try_url(url)

    log.info(f"  → Thu được {len(collected)} bài cho '{topic}'")
    return collected


def main():
    log.info("=" * 55)
    log.info("  Bắt đầu thu thập dataset Vũng Tàu")
    log.info("=" * 55)

    if not NEWSPAPER_OK:
        log.warning("newspaper3k chưa cài → dùng BeautifulSoup. Cài bằng: pip install newspaper3k")
    if not LANGDETECT_OK:
        log.warning("langdetect chưa cài → phát hiện tiếng Việt bằng heuristic. Cài: pip install langdetect")

    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    topics = list(SEED_URLS.keys())
    for topic in topics:
        articles = crawl_topic(topic, seen_urls)
        # Lưu từng bài
        for i, art in enumerate(articles, start=1):
            path = save_article(art, topic, i)
            log.info(f"    Đã lưu: {path}")
        all_articles.extend(articles)

    # Tổng hợp thống kê
    log.info("\n" + "=" * 55)
    log.info("  THỐNG KÊ KẾT QUẢ")
    log.info("=" * 55)
    from collections import Counter
    counts = Counter(a["topic"] for a in all_articles)
    total = 0
    for topic in topics:
        n = counts.get(topic, 0)
        status = "✓" if n >= TARGET_PER_TOPIC else f"⚠ (mục tiêu {TARGET_PER_TOPIC})"
        log.info(f"  {status:25s} {topic}: {n} bài")
        total += n
    log.info(f"\n  Tổng cộng: {total} bài")

    # Lưu file tổng hợp
    summary_path = os.path.join(DATASET_DIR, "all_articles.json")
    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, ensure_ascii=False, indent=2)
    log.info(f"\n  File tổng hợp: {summary_path}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
