import asyncio
from crawl4ai import AsyncWebCrawler
from typing import List, Dict

class ContentFetcher:
    """
    A utility class to fetch content from a list of URLs using Crawl4AI's AsyncWebCrawler.
    """

    async def fetch_content(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Asynchronously fetch webpage content from a list of URLs.

        Parameters:
            urls (List[str]): List of webpage URLs.

        Returns:
            List[Dict[str, str]]: A list of dictionaries with 'url' and 'content'.
        """

        content_list = []
        
        async with AsyncWebCrawler() as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url)
                    content_list.append({"url": url, "content": result.markdown})
                except Exception as e:
                    print(f"Failed to fetch content for {url}: {e}")
        
        return content_list