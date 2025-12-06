"""Web Search Service: Performs quick online searches."""
import requests
from typing import Dict, List, Optional
import time


class WebSearchService:
    """Service to perform quick web searches."""
    
    def __init__(self, api_key: Optional[str] = None, search_engine: str = "duckduckgo"):
        """
        Initialize web search service.
        
        Args:
            api_key: Optional API key for search services (if needed)
            search_engine: Search engine to use ('duckduckgo' or 'google')
        """
        self.api_key = api_key
        self.search_engine = search_engine
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Perform a quick web search.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
        
        Returns:
            List of search results with 'title', 'snippet', and 'url'
        """
        try:
            if self.search_engine == "duckduckgo":
                return self._search_duckduckgo(query, max_results)
            elif self.search_engine == "google":
                return self._search_google(query, max_results)
            else:
                print(f"[WebSearch] Unknown search engine: {self.search_engine}, using DuckDuckGo")
                return self._search_duckduckgo(query, max_results)
        except Exception as e:
            print(f"[WebSearch] Error searching: {e}")
            return []
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using DuckDuckGo (no API key needed)."""
        try:
            # Try to import BeautifulSoup
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                print("[WebSearch] beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
                return []
            
            # Use DuckDuckGo HTML search (simple, no API key)
            url = "https://html.duckduckgo.com/html/"
            params = {
                'q': query,
                'kl': 'us-en'
            }
            
            response = self.session.get(url, params=params, timeout=5)
            response.raise_for_status()
            
            # Parse HTML results (simple extraction)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            # Find result links
            result_links = soup.find_all('a', class_='result__a', limit=max_results)
            
            for link in result_links:
                title = link.get_text(strip=True)
                url = link.get('href', '')
                
                # Find snippet (next sibling or nearby)
                snippet_elem = link.find_next('a', class_='result__snippet')
                if not snippet_elem:
                    snippet_elem = link.find_next('div', class_='result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                
                if title and url:
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'url': url
                    })
            
            return results[:max_results]
        except Exception as e:
            print(f"[WebSearch] DuckDuckGo search error: {e}")
            # Fallback: return empty results
            return []
    
    def _search_google(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Search using Google (requires API key or scraping)."""
        # For now, fallback to DuckDuckGo
        # Can be extended with Google Custom Search API if API key is provided
        print("[WebSearch] Google search not implemented, using DuckDuckGo")
        return self._search_duckduckgo(query, max_results)
    
    def get_search_context(self, query: str, max_results: int = 3) -> str:
        """
        Get search results as formatted context text.
        
        Args:
            query: Search query
            max_results: Maximum results to include
        
        Returns:
            Formatted context string with search results
        """
        results = self.search(query, max_results)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"{i}. {result['title']}\n"
                f"   {result['snippet']}\n"
                f"   Source: {result['url']}"
            )
        
        return "\n\n".join(context_parts)

