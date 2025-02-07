from duckduckgo_search import DDGS
from typing import List, Dict, Optional

class WebSearch:
    """
    A reusable class for performing web searches using DuckDuckGo.
    """

    def __init__(self, max_results: int = 5, region: str = "wt-wt", safe_search: str = "moderate"):
        """
        Initialize the WebSearch client with configurable options.
        """
        self.max_results = max_results
        self.region = region
        self.safe_search = safe_search
        self.ddg_client = DDGS()  

    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo.

        Parameters:
            query (str): The search query.
            max_results (int, optional): How many results to retrieve. If None, defaults to self.max_results.

        Returns:
            List[Dict[str, str]]: List of structured search results.
        """
        used_max_results = max_results if max_results is not None else self.max_results
        try:
            results = self.ddg_client.text(
                query,
                max_results=used_max_results,
                region=self.region,
                safesearch=self.safe_search
            )

            if not results:
                print("No results found.")
                return []

            structured_results = [
                {
                    "title": result.get("title", "No Title"),
                    "url": result.get("href", ""),
                    "snippet": result.get("body", "No snippet available.")
                }
                for result in results
            ]

            return structured_results

        except Exception as e:
            print(f"Search request failed: {e}")
            return []
