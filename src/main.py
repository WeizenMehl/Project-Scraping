import requests
import csv

# import numpy as np
from bs4 import BeautifulSoup

laptop_links = []
laptops = []
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

for i in range(1, 100):
    url = f"https://geizhals.at/?cat=nb&pg={i}#productlist"
    response = session.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    product_tags = soup.find_all("a", class_="productlist__link")

    for tag in product_tags:
        href = tag.get("href")
        href = f"https://geizhals.at/{href}"
        if href:
            laptop_links.append(href)
for link in laptop_links:
    response = session.get(link)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        specs_grid = soup.find("dl", class_="specs-grid")

        if specs_grid:
            specs = {}

            for spec in specs_grid.find_all("div", class_="specs-grid__item"):
                type_ = spec.find("dt")
                stat = spec.find("dd")

                if type_ and stat:
                    specs[type_.text.strip()] = stat.text.strip()

            name = (
                soup.find("h1", class_="variant__header__headline").get_text().strip()
            )
            span_range = soup.find("span", id="pricerange-min-max")
            max_price = ""
            min_price = ""
            if span_range:
                min_strong = span_range.find("strong", id="pricerange-min")
                max_strong = span_range.find("strong", id="pricerange-max")

                if min_strong and min_strong.find("span", class_="gh_price"):
                    price_min = (
                        min_strong.find("span", class_="gh_price").get_text().strip()
                    )

                if max_strong and max_strong.find("span", class_="gh_price"):
                    price_max = (
                        max_strong.find("span", class_="gh_price").get_text().strip()
                    )
            if not specs["Betriebssystem"]:
                print(link)
            laptops.append(
                {
                    "name": name,
                    "resolution": specs["Display"],
                    "hz": specs["Display"],
                    "cpu": specs["CPU"],
                    "gpu": specs["Grafik"],
                    "ram": specs["RAM"],
                    "os": specs["Betriebssystem"],
                    "price_min": price_min,
                    "price_max": price_max,
                    "url": link,
                }
            )
with open("laptops.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)

    writer.writerow(
        [
            "name",
            "resolution",
            "hz",
            "cpu",
            "gpu",
            "ram",
            "os",
            "price_min",
            "price_max",
            "url",
        ]
    )

    for lap in laptops:
        writer.writerow(
            [
                lap.get("name", ""),
                lap.get("resolution", ""),
                lap.get("hz", ""),
                lap.get("cpu", ""),
                lap.get("gpu", ""),
                lap.get("ram", ""),
                lap.get("os", ""),
                lap.get("price_min", ""),
                lap.get("price_max", ""),
                lap.get("url", ""),
            ]
        )
