"""Web crawler module using Firecrawl API for collecting RAG data.

Supports multiple crawl modes:
- single: Scrape only the given URL
- links: Scrape URL + all links found on that page  
- search: Use map API to find URLs by topic, then scrape
- domain: Use crawl API to crawl entire domain
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from tqdm import tqdm

from src.config import DATA_CRAWLED_DIR
from src.utils.common import remove_diacritics

load_dotenv()

DELAY_SECONDS = 6


class WebCrawler:
    """Web crawler with multiple crawl modes."""
    
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY", "")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY required")
        self.app = FirecrawlApp(api_key=self.api_key)
        self._last_request = 0.0
    
    def _wait_rate_limit(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request
        if elapsed < DELAY_SECONDS:
            time.sleep(DELAY_SECONDS - elapsed)
        self._last_request = time.time()
    
    def _scrape_page(self, url: str, get_links: bool = False) -> dict | None:
        """Scrape a single page."""
        self._wait_rate_limit()
        
        formats = ["markdown", "summary"]
        if get_links:
            formats.append("links")
        
        try:
            result = self.app.scrape(url, formats=formats, only_main_content=True)
            if hasattr(result, 'markdown') and result.markdown:
                meta = result.metadata if hasattr(result, 'metadata') else None
                return {
                    "url": url,
                    "title": getattr(meta, 'title', '') if meta else "",
                    "content": result.markdown,
                    "summary": getattr(result, 'summary', '') or "",
                    "description": getattr(meta, 'description', '') if meta else "",
                    "keywords": getattr(meta, 'keywords', '') if meta else "",
                    "links": result.links if get_links and hasattr(result, 'links') else [],
                }
        except Exception as e:
            print(f"[Crawler] Error scraping {url}: {e}")
        return None
    
    def crawl_single(self, url: str) -> list[dict]:
        """Mode: single - Crawl only the given URL."""
        print(f"[Crawler] Mode: single | URL: {url}")
        result = self._scrape_page(url)
        if result:
            result.pop("links", None)
            return [result]
        return []
    
    def crawl_links(self, url: str, max_pages: int = 10, topic: str = "") -> list[dict]:
        """Mode: links - Crawl URL + links containing topic keywords."""
        if not topic:
            raise ValueError("Topic required for links mode. Use comma to separate keywords.")
        
        keywords = [k.strip() for k in topic.split(",") if k.strip()]
        keywords_normalized = [remove_diacritics(k).replace("_", " ").replace("-", " ") for k in keywords]
        
        print(f"[Crawler] Mode: links | URL: {url}")
        print(f"[Crawler] Keywords: {keywords}")
        print(f"[Crawler] Scraping main page...")
        
        documents = []
        main_result = self._scrape_page(url, get_links=True)
        if not main_result:
            return []
        
        links = main_result.pop("links", [])
        documents.append(main_result)
        print(f"[Crawler] Found {len(links)} total links")
        
        if not links or max_pages <= 1:
            return documents
        
        urls_to_scrape = []
        source_domain = urlparse(url).netloc
        
        skip_patterns = [
            '.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '/Special:', '/Đặc_biệt:', '/Tập_tin:', '/File:',
            '/Thể_loại:', '/Category:', '/Help:', '/Wikipedia:',
            'action=edit', 'action=history', 'oldid=', 'diff=',
            '/w/index.php', '/wiki/index.php',
            '#cite', '#ref', '#mw-',
        ]
        
        for link in links:
            if hasattr(link, 'url'):
                link_url = link.url
            elif isinstance(link, str):
                link_url = link
            elif isinstance(link, dict):
                link_url = link.get('url', '')
            else:
                continue
            
            if not link_url:
                continue
            
            link_domain = urlparse(link_url).netloc
            if link_domain and link_domain != source_domain:
                continue
            
            decoded_url = unquote(link_url)
            if any(p.lower() in decoded_url.lower() for p in skip_patterns):
                continue
            
            normalized_url = remove_diacritics(decoded_url).replace("_", " ").replace("-", " ")
            
            for i, kw_norm in enumerate(keywords_normalized):
                if kw_norm in normalized_url:
                    print(f"        Match: '{keywords[i]}' in {decoded_url}")
                    urls_to_scrape.append(link_url)
                    break
        
        print(f"[Crawler] Matched {len(urls_to_scrape)} links")
        
        if not urls_to_scrape:
            return documents
        
        scrape_count = min(len(urls_to_scrape), max_pages - 1)
        print(f"[Crawler] Scraping {scrape_count} pages...")
        
        for page_url in tqdm(urls_to_scrape[:scrape_count], desc="Scraping"):
            if page_url == url:
                continue
            result = self._scrape_page(page_url)
            if result:
                result.pop("links", None)
                documents.append(result)
        
        return documents
    
    def crawl_search(self, url: str, topic: str, max_pages: int = 10) -> list[dict]:
        """Mode: search - Use map API to find URLs by topic, then scrape."""
        print(f"[Crawler] Mode: search | URL: {url} | Topic: {topic}")
        documents = []
        
        try:
            map_result = self.app.map(url=url, search=topic, limit=max_pages)
            
            urls_to_scrape = []
            if hasattr(map_result, 'links'):
                for link in map_result.links:
                    if hasattr(link, 'url'):
                        urls_to_scrape.append(link.url)
                    elif isinstance(link, str):
                        urls_to_scrape.append(link)
            elif isinstance(map_result, list):
                for link in map_result:
                    if isinstance(link, str):
                        urls_to_scrape.append(link)
                        
        except Exception as e:
            print(f"[Crawler] Error: Map API failed - {e}")
            return []
        
        print(f"[Crawler] Found {len(urls_to_scrape)} URLs")
        
        for page_url in tqdm(urls_to_scrape, desc="Scraping"):
            result = self._scrape_page(page_url)
            if result:
                result.pop("links", None)
                documents.append(result)
        
        return documents
    
    def crawl_domain(self, url: str, max_pages: int = 10, topic: str | None = None) -> list[dict]:
        """Mode: domain - Use crawl API to crawl domain."""
        print(f"[Crawler] Mode: domain | URL: {url}")
        documents = []
        
        crawl_options = {
            "limit": max_pages,
            "scrape_options": {
                "formats": ["markdown"],
                "only_main_content": True,
            }
        }
        
        if topic:
            crawl_options["include_paths"] = [f"*{topic}*"]
        
        try:
            result = self.app.crawl(url=url, **crawl_options)
            
            pages = []
            if hasattr(result, 'data'):
                pages = result.data or []
            elif isinstance(result, dict) and 'data' in result:
                pages = result['data'] or []
            
            for page in pages:
                if hasattr(page, 'markdown') and page.markdown:
                    meta = page.metadata if hasattr(page, 'metadata') else None
                    documents.append({
                        "url": getattr(meta, 'sourceURL', '') if meta else "",
                        "title": getattr(meta, 'title', '') if meta else "",
                        "content": page.markdown,
                        "description": getattr(meta, 'description', '') if meta else "",
                        "keywords": getattr(meta, 'keywords', '') if meta else "",
                    })
                    
        except Exception as e:
            print(f"[Crawler] Error: Crawl API failed - {e}")
        
        print(f"[Crawler] Crawled {len(documents)} pages")
        return documents


def crawl_website(
    url: str,
    mode: str = "links",
    topic: str | None = None,
    max_pages: int = 10,
    api_key: str | None = None,
) -> dict:
    """Main crawl function with mode selection.
    
    Args:
        url: URL to crawl
        mode: Crawl mode - 'single', 'links', 'search', 'domain'
        topic: Optional topic filter (required for 'search' mode)
        max_pages: Maximum pages to crawl
        api_key: Firecrawl API key
    """
    crawler = WebCrawler(api_key)
    
    if mode == "single":
        documents = crawler.crawl_single(url)
    elif mode == "links":
        documents = crawler.crawl_links(url, max_pages, topic)
    elif mode == "search":
        if not topic:
            raise ValueError("Topic required for search mode")
        documents = crawler.crawl_search(url, topic, max_pages)
    elif mode == "domain":
        documents = crawler.crawl_domain(url, max_pages, topic)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use: single, links, search, domain")
    
    print(f"[Crawler] Total: {len(documents)} documents")
    
    return {
        "source": url,
        "domain": urlparse(url).netloc,
        "mode": mode,
        "topic": topic,
        "crawled_at": datetime.now().isoformat(),
        "total_pages": len(documents),
        "documents": documents,
    }


def save_crawled_data(data: dict, output_dir: str | Path | None = None, filename: str | None = None) -> Path:
    """Save crawled data to JSON."""
    if output_dir is None:
        output_path = DATA_CRAWLED_DIR
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not filename:
        domain = data.get("domain", "unknown").replace(".", "_")
        filename = f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    path = output_path / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"[Crawler] Saved: {path}")
    return path
