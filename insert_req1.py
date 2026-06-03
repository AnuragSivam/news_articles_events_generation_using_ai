import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
from datetime import datetime
import time
import random

BASE_URL = "https://www.mzamin.com"
START_DATE = datetime(2026, 5, 10)
END_DATE = datetime(2026, 5, 14)
TARGET_ARTICLES = 1600

SEED_URLS = [
    "https://www.mzamin.com/politics",
    "https://www.mzamin.com/sports",
    "https://www.mzamin.com/economy",
    "https://www.mzamin.com/world",
    "https://www.mzamin.com/newstoday",
    "https://www.mzamin.com/category.php?cat=1",
    "https://www.mzamin.com/category.php?cat=2",
    "https://www.mzamin.com/category.php?cat=3",
    "https://www.mzamin.com/category.php?cat=4",
    "https://www.mzamin.com/category.php?cat=5"
]

BLACKLIST = [
    "entertainment", "binodon", "বিনোদোন", "movie", "cinema", 
    "drama", "music", "gossip", "showbiz", "celebrity"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
}

def get_article_date(soup):
    meta_date = soup.find("meta", property="article:published_time")
    if meta_date:
        return meta_date["content"][:10]
    return None

def is_allowed(url):
    parsed = urlparse(url)
    if "mzamin.com" not in parsed.netloc:
        return False
    
    url_lower = url.lower()
    if any(word in url_lower for word in BLACKLIST):
        return False
        
    if url_lower.endswith(('.jpg', '.png', '.pdf', '.mp4', '.zip', '.jpeg', '.gif')):
        return False
    return True

visited = set()
article_links = set()
queue = deque(SEED_URLS)

try:
    while queue and len(article_links) < TARGET_ARTICLES:
        current_url = queue.popleft()
        
        if current_url in visited:
            continue
        
        visited.add(current_url)
        
        try:
            response = requests.get(current_url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            pub_date_str = get_article_date(soup)
            if pub_date_str:
                try:
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                    if START_DATE <= pub_date <= END_DATE:
                        if current_url not in article_links:
                            article_links.add(current_url)
                            with open("mz_selected_genres.txt", "a", encoding="utf-8") as f:
                                f.write(f"{current_url}\n")
                            print(f"[{len(article_links)}] {pub_date_str} | {current_url}")
                except ValueError:
                    pass

            for a in soup.find_all("a", href=True):
                full_link = urljoin(BASE_URL, a["href"]).split("#")[0].split("?")[0]
                
                if is_allowed(full_link) and full_link not in visited:
                    queue.append(full_link)

            time.sleep(0.05)

        except Exception:
            continue

except KeyboardInterrupt:
    pass

print(f"Finished: {len(article_links)} links saved.")