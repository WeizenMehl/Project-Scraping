import requests
from laptop import Laptop
import numpy as np
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

for i in range(1, 9):
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

        specs_grid = soup.find('dl', class_='specs-grid')

        if specs_grid:
            specs = {}

            for spec in specs_grid.find_all('div', class_='specs-grid__item'):
                type_ = spec.find('dt')
                stat = spec.find('dd')

                if type_ and stat:
                    specs[type_.text.strip()] = stat.text.strip()

            # Extract the required data to initialize the class
            resolution = specs["Display"].split(",")[1].strip()  # 2560x1664
            hz = specs["Display"].split(",")[3].strip()  # 60Hz
            cpu = specs["CPU"]
            gpu = specs["Grafik"]
            ram = specs["RAM"].split()[0]  # 16GB
            os = specs["Betriebssystem"]
            # Assuming you have the price as well (manually adding a price here for now)
            price = "€ 895,46"
            # Initialize the Laptop class
            laptops.append(Laptop(resolution, hz, cpu, gpu, ram, os, price))
