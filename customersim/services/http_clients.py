#!/usr/bin/env python3

import asyncio
import random

import aiohttp
import logging
from typing import List
from customersim.services.constants import Document
logger = logging.getLogger(__name__)

class LookupServiceClient:
    """HTTP client for the consolidated Lookup Service"""

    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.base_url = base_url
        self.session = None
        self._connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

    async def _get_session(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout
            )
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
        if self._connector and not self._connector.closed:
            await self._connector.close()

    async def search_products(self, query: str, k: int = 4, product_category: str = None,
                              max_retries: int = 3, base_delay: float = 0.5, max_delay: float = 8.0):
        """Search for products via HTTP API with exponential backoff."""
        session = await self._get_session()
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    f"{self.base_url}/products/search",
                    json={"query": query, "k": k, "product_category": product_category},
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [Document(item["page_content"], item["metadata"]) for item in data]

                    error_text = await response.text()

                    if 400 <= response.status < 500 and response.status != 429:
                        raise RuntimeError(
                            f"Product search failed with non-retryable status {response.status}: {error_text}")

                    logger.warning(f"Product search retryable error {response.status} "
                                   f"(attempt {attempt+1}/{max_retries+1}): {error_text}")
                    last_exception = RuntimeError(
                        f"Product search failed with status {response.status}: {error_text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Product search request failed "
                               f"(attempt {attempt+1}/{max_retries+1}): {e}")
                last_exception = e

            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 0.5), max_delay)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Product search failed after {max_retries+1} attempts") from last_exception

    async def search_buying_guides(self, query: str, k: int = 4, product_category: str = None):
        """Search for buying guides via HTTP API"""
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/guides/search",
                json={"query": query, "k": k, "product_category": product_category},
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    # Convert back to Document-like objects for compatibility
                    return [Document(item["page_content"], item["metadata"]) for item in data]
                else:
                    error_text = await response.text()
                    logger.error(f"Buying guide search error {response.status}: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search buying guides: {e}")
            return []

# Legacy classes for backward compatibility
class ProductLookupClient:
    """Legacy wrapper for product lookups"""
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.client = LookupServiceClient(base_url)

    async def top_docs(self, query: str, k: int = 4, product_category: str = None):
        return await self.client.search_products(query, k, product_category)

    async def close(self):
        await self.client.close()

class BuyingGuideClient:
    """Legacy wrapper for buying guide lookups"""
    def __init__(self, base_url: str = "http://127.0.0.1:8003"):
        self.client = LookupServiceClient(base_url)

    async def top_docs(self, query: str, k: int = 4, product_category: str = None):
        return await self.client.search_buying_guides(query, k, product_category)

    async def close(self):
        await self.client.close()