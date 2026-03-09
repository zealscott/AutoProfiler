"""Web search and digest functions replacing AgentScope's built-in services."""

import os
from typing import Optional

import requests
from bs4 import BeautifulSoup

from core.toolkit import ServiceExecStatus, ServiceResponse


def bing_search(query: str, api_key: str, num_results: int = 10) -> ServiceResponse:
    """
    Search the web using Bing Search API.
    Args:
        query (`str`):
            The search query string.
        api_key (`str`):
            The Bing Search API key.
        num_results (`int`, defaults to `10`):
            The number of results to return.
    Returns:
        `ServiceResponse`: Search results with status and content.
    """
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {"q": query, "count": num_results, "textDecorations": True, "textFormat": "HTML"}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        return ServiceResponse(ServiceExecStatus.SUCCESS, results)
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))


def google_search(
    query: str,
    api_key: str,
    cse_id: str,
    num_results: int = 10,
) -> ServiceResponse:
    """
    Search the web using Google Custom Search API.
    Args:
        query (`str`):
            The search query string.
        api_key (`str`):
            The Google API key.
        cse_id (`str`):
            The Google Custom Search Engine ID.
        num_results (`int`, defaults to `10`):
            The number of results to return.
    Returns:
        `ServiceResponse`: Search results with status and content.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": cse_id, "q": query, "num": min(num_results, 10)}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        return ServiceResponse(ServiceExecStatus.SUCCESS, results)
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))


def digest_webpage(url: str) -> ServiceResponse:
    """
    Fetch and extract the main text content from a webpage.
    Args:
        url (`str`):
            The URL of the webpage to digest.
    Returns:
        `ServiceResponse`: The extracted text content with status.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate if too long
        if len(text) > 8000:
            text = text[:8000] + "\n...[truncated]"

        return ServiceResponse(ServiceExecStatus.SUCCESS, text)
    except Exception as e:
        return ServiceResponse(ServiceExecStatus.ERROR, str(e))
