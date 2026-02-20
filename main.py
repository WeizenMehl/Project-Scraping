import requests
import laptop
from bs4 import BeautifulSoup

laptop_links = []
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://geizhals.at/",
    "Connection": "keep-alive",
}
session = requests.Session()
session.headers.update(headers)
session.max_redirects = 3

for i in range(1, 9):
    url = f"https://geizhals.at/?cat=nb&pg={i}#productlist"
    response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    product_tags = soup.find_all("a", class_="productlist__link")

    for tag in product_tags:
        href = tag.get("href")
        if href:
            laptop_links.append(href)
