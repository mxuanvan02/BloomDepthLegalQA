#!/usr/bin/env python3
"""
9Router (NINEROUTER) search/fetch client.

Provides precise web search via Tavily/Exa providers through the local 9Router gateway,
avoiding LLM hallucination by returning real URLs with metadata.

Usage:
    from ninerouter_search import NineRouterSearch
    client = NineRouterSearch()
    results = client.search("giáo trình luật dân sự pdf", provider="tavily", limit=10)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


ENV_FILES = (
    Path("/opt/data/.env"),
    Path("/Users/van/AI/hermes-stack/config/.env"),
    Path("/Users/van/AI/openclaw/.env"),
)


def _load_env() -> None:
    """Load NINEROUTER credentials from known .env locations."""
    for path in ENV_FILES:
        if not path.exists():
            continue
        try:
            for raw in path.read_text(errors="replace").splitlines():
                line = raw.strip()
                if line.startswith("NINEROUTER_API_KEY=") and "NINEROUTER_API_KEY" not in os.environ:
                    os.environ["NINEROUTER_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
                if line.startswith("NINEROUTER_ENDPOINT=") and "NINEROUTER_ENDPOINT" not in os.environ:
                    os.environ["NINEROUTER_ENDPOINT"] = line.split("=", 1)[1].strip().strip('"').strip("'")
        except Exception as e:
            logger.debug(f"Could not read env file {path}: {e}")


def _base_url() -> str:
    """Resolve 9Router base URL, adjusting for non-docker host."""
    value = os.environ.get("NINEROUTER_ENDPOINT", "http://host.docker.internal:20128/v1").rstrip("/")
    if not Path("/.dockerenv").exists():
        value = value.replace("host.docker.internal", "127.0.0.1")
    return value


class NineRouterSearch:
    """Client for 9Router search and fetch endpoints."""

    def __init__(self, timeout: int = 45):
        _load_env()
        self.api_key = os.environ.get("NINEROUTER_API_KEY", "")
        self.base_url = _base_url()
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError("NINEROUTER_API_KEY is not set. Cannot use 9Router search.")

    def _post(self, path: str, payload: dict) -> dict:
        """POST request to 9Router endpoint."""
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return {
                    "ok": True,
                    "status": response.status,
                    "body": json.loads(response.read().decode("utf-8", "replace")),
                }
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = {"raw": body[:500]}
            return {"ok": False, "status": exc.code, "body": parsed}
        except URLError as exc:
            return {"ok": False, "status": None, "body": {"error": str(exc)}}

    def search(
        self,
        query: str,
        provider: str = "tavily",
        limit: int = 10,
    ) -> List[Dict]:
        """
        Search the web via 9Router.

        Returns normalized list of:
            {url, title, snippet, score, published_at, provider}
        """
        result = self._post("/search", {
            "provider": provider,
            "query": query,
            "limit": limit,
        })

        if not result["ok"]:
            logger.warning(f"Search failed for '{query}' via {provider}: {result['body']}")
            return []

        body = result["body"]
        raw_results = body.get("results", [])

        normalized = []
        for r in raw_results:
            normalized.append({
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": r.get("score", 0),
                "published_at": r.get("published_at"),
                "provider": provider,
            })

        return normalized

    def search_multi_provider(
        self,
        query: str,
        providers: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Search across multiple providers and merge/dedupe by URL."""
        if providers is None:
            providers = ["tavily", "exa"]

        all_results = []
        seen_urls = set()

        for provider in providers:
            results = self.search(query, provider=provider, limit=limit)
            for r in results:
                if r["url"] and r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)

        return all_results

    def fetch(self, url: str, provider: str = "jina-reader") -> Optional[str]:
        """Fetch URL content via 9Router fetch endpoint."""
        result = self._post("/web/fetch", {
            "provider": provider,
            "url": url,
        })

        if not result["ok"]:
            logger.warning(f"Fetch failed for '{url}' via {provider}: {result['body']}")
            return None

        body = result["body"]
        return body.get("content") or body.get("text") or ""


if __name__ == "__main__":
    # Quick self-test
    logging.basicConfig(level=logging.INFO)
    client = NineRouterSearch()
    print("Testing 9Router search...")
    results = client.search("giáo trình luật dân sự Việt Nam pdf", provider="tavily", limit=5)
    print(f"\nFound {len(results)} results:")
    for r in results:
        print(f"  [{r['score']:.2f}] {r['title']}")
        print(f"        {r['url']}")
