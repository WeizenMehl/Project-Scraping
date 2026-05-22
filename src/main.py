import requests
import csv
import re
import time
from typing import List, Optional
from bs4 import BeautifulSoup
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DisplaySpecs:
    """Data class for structured display specifications."""

    dimension: str = ""
    resolution: str = ""
    aspect_ratio: str = ""
    ppi: str = ""
    hz: str = ""


@dataclass
class LaptopData:
    """Data class for complete laptop specifications."""

    name: str
    display: DisplaySpecs
    cpu: str
    gpu: str
    ram: str
    os: str
    price_min: str
    price_max: str
    url: str


class GeizhalsLaptopScraper:
    """
    Web scraper for extracting laptop specifications from Geizhals.at.

    Handles pagination, product parsing, display spec extraction, and CSV export.
    Includes robust error handling and rate limiting.
    """

    BASE_URL = "https://geizhals.at"
    SEARCH_URL = "https://geizhals.at/?cat=nb&pg={page}#productlist"

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://geizhals.at/",
        "Connection": "keep-alive",
    }

    def __init__(self, max_pages: int = 50, delay: float = 1.0):
        """
        Initialize the scraper.

        Args:
            max_pages: Maximum number of pages to scrape
            delay: Delay between requests in seconds (rate limiting)
        """
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.session.max_redirects = 3
        self.max_pages = max_pages
        self.delay = delay
        self.laptops: List[LaptopData] = []
        self.no_os_count = 0

    def clean_resolution(self, res_str: str) -> str:
        """
        Extract numeric resolution (e.g., '1920x1080') from formatted string.

        Args:
            res_str: Input string like '1920x1080 (Full HD)'

        Returns:
            Clean resolution string or empty string if invalid
        """
        if not res_str:
            return ""

        # Remove parenthetical content and extra whitespace
        numeric_part = re.sub(r"\s*\([^)]*\)", "", res_str).strip()
        return numeric_part

    def parse_display_specs(self, text: str) -> DisplaySpecs:
        """
        Parse display specifications from comma-separated string.

        Args:
            text: Raw display spec string like "15.6\",1920x1080 (Full HD),16:9,141,144 Hz"

        Returns:
            Structured DisplaySpecs object
        """
        if not text:
            return DisplaySpecs()

        parts = [p.strip() for p in text.split(",")]
        return DisplaySpecs(
            dimension=parts[0] if len(parts) > 0 else "",
            resolution=self.clean_resolution(parts[1]) if len(parts) > 1 else "",
            aspect_ratio=parts[2] if len(parts) > 2 else "",
            ppi=parts[3] if len(parts) > 3 else "",
            hz=parts[4] if len(parts) > 4 else "",
        )

    def get_product_links(self) -> List[str]:
        """Extract product links from all search result pages."""
        links = []

        for page in range(1, self.max_pages + 1):
            print(f"📄 Scraping page {page}/{self.max_pages}")

            url = self.SEARCH_URL.format(page=page)
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, "html.parser")
                product_tags = soup.find_all("a", class_="productlist__link")

                for tag in product_tags:
                    href = tag.get("href")
                    if href:
                        full_url = f"{self.BASE_URL}/{href.lstrip('/')}"
                        links.append(full_url)
                        print(f"   → Found: {full_url.split('/')[-1]}")

                # Rate limiting
                time.sleep(self.delay)

            except requests.RequestException as e:
                print(f"❌ Error on page {page}: {e}")
                continue

        print(f"✅ Found {len(links)} total product links")
        return links

    def extract_price_range(self, soup: BeautifulSoup) -> tuple[str, str]:
        """
        Extract minimum and maximum price from price range element.

        Returns:
            Tuple of (price_min, price_max)
        """
        span_range = soup.find("span", id="pricerange-min-max")
        if not span_range:
            return "", ""

        price_min = ""
        price_max = ""

        min_strong = span_range.find("strong", id="pricerange-min")
        max_strong = span_range.find("strong", id="pricerange-max")

        if min_strong:
            price_min_elem = min_strong.find("span", class_="gh_price")
            price_min = price_min_elem.get_text().strip() if price_min_elem else ""

        if max_strong:
            price_max_elem = max_strong.find("span", class_="gh_price")
            price_max = price_max_elem.get_text().strip() if price_max_elem else ""

        return price_min, price_max

    def scrape_product(self, url: str) -> Optional[LaptopData]:
        """
        Scrape detailed specifications from individual product page.

        Args:
            url: Product URL

        Returns:
            LaptopData object or None if parsing failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Extract product name
            name_tag = soup.find("h1", class_="variant__header__headline")
            name = (
                name_tag.get_text().strip().split(",")[0].strip()
                if name_tag
                else "Unknown"
            )

            # Extract specs grid
            specs_grid = soup.find("dl", class_="specs-grid")
            if not specs_grid:
                print(f"⚠️  No specs grid found for {name}")
                return None

            specs = {}
            for spec in specs_grid.find_all("div", class_="specs-grid__item"):
                dt = spec.find("dt")
                dd = spec.find("dd")
                if dt and dd:
                    specs[dt.text.strip()] = dd.text.strip()

            # Handle missing OS gracefully
            os_value = specs.get("Betriebssystem", "Not specified")
            if os_value == "Not specified":
                self.no_os_count += 1
                print(f"ℹ️  No OS found for {name} ({self.no_os_count} total)")

            # Parse display specs
            display_specs = self.parse_display_specs(specs.get("Display", ""))

            # Extract prices
            price_min, price_max = self.extract_price_range(soup)

            return LaptopData(
                name=name,
                display=display_specs,
                cpu=specs.get("CPU", ""),
                gpu=specs.get("Grafik", ""),
                ram=specs.get("RAM", ""),
                os=os_value,
                price_min=price_min,
                price_max=price_max,
                url=url,
            )

        except requests.RequestException as e:
            print(f"❌ Network error for {url}: {e}")
            return None
        except Exception as e:
            print(f"❌ Parse error for {url}: {e}")
            return None

    def scrape_all(self) -> None:
        """Main scraping workflow."""
        print("🚀 Starting Geizhals laptop scraper...")

        # Get all product links
        links = self.get_product_links()

        # Scrape individual products
        print(f"\n📊 Scraping details for {len(links)} products...")
        for i, link in enumerate(links, 1):
            print(f"[{i}/{len(links)}] Processing: {link.split('/')[-1]}")

            laptop = self.scrape_product(link)
            if laptop:
                self.laptops.append(laptop)
                print(f"   ✅ Added: {laptop.name}")
            else:
                print("   ❌ Skipped")

            time.sleep(self.delay)  # Rate limiting

        print("\n🎉 Scraping complete!")
        print(f"   Total laptops: {len(self.laptops)}")
        print(f"   No OS found: {self.no_os_count}")

    def save_to_csv(self, filename: str = "laptops.csv") -> None:
        """Export scraped data to CSV file."""
        fieldnames = [
            "name",
            "dimension",
            "resolution",
            "aspect_ratio",
            "ppi",
            "hz",
            "cpu",
            "gpu",
            "ram",
            "os",
            "price_min",
            "price_max",
            "url",
        ]

        filepath = Path(filename)
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for laptop in self.laptops:
                row = {
                    "name": laptop.name,
                    "dimension": laptop.display.dimension,
                    "resolution": laptop.display.resolution,
                    "aspect_ratio": laptop.display.aspect_ratio,
                    "ppi": laptop.display.ppi,
                    "hz": laptop.display.hz,
                    "cpu": laptop.cpu,
                    "gpu": laptop.gpu,
                    "ram": laptop.ram,
                    "os": laptop.os,
                    "price_min": laptop.price_min,
                    "price_max": laptop.price_max,
                    "url": laptop.url,
                }
                writer.writerow(row)

        print(f"💾 Saved {len(self.laptops)} laptops to {filename}")


def main():
    """Main execution function."""
    scraper = GeizhalsLaptopScraper(max_pages=50, delay=1.0)
    scraper.scrape_all()
    scraper.save_to_csv("laptops.csv")


if __name__ == "__main__":
    main()
