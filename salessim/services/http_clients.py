#!/usr/bin/env python3

import aiohttp
import logging
from typing import List
from salessim.services.constants import Document
logger = logging.getLogger(__name__)

class LookupServiceClient:
    """HTTP client for the consolidated Lookup Service"""

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
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

    async def search_products(self, query: str, k: int = 4):
        """Search for products via HTTP API"""
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/products/search",
                json={"query": query, "k": k}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return [Document(item["page_content"], item["metadata"]) for item in data]
                else:
                    error_text = await response.text()
                    logger.error(f"Product search error {response.status}: {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            return []

    async def search_buying_guides(self, query: str, k: int = 4):
        """Search for buying guides via HTTP API"""
        session = await self._get_session()

        try:
            async with session.post(
                f"{self.base_url}/guides/search",
                json={"query": query, "k": k},
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

    async def find_recommended_items_in_response(self, candidates: List, response: str, sim_threshold: float = 0.70):
        """Find recommended items in response via HTTP API"""
        session = await self._get_session()

        # Convert candidates to serializable format
        candidates_data = []
        for candidate in candidates:
            candidates_data.append({
                "page_content": candidate.page_content,
                "metadata": candidate.metadata
            })

        try:
            async with session.post(
                f"{self.base_url}/sales/find_recommended_items",
                json={
                    "candidates": candidates_data,
                    "response": response,
                    "sim_threshold": sim_threshold
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response_obj:
                if response_obj.status == 200:
                    data = await response_obj.json()
                    # Convert back to Document-like objects for compatibility
                    return [Document(item["page_content"], item["metadata"]) for item in data]
                else:
                    error_text = await response_obj.text()
                    logger.error(f"Find recommended items error {response_obj.status}: {error_text}")
                    return []
        except Exception as e:
            breakpoint()
            logger.error(f"Failed to find recommended items: {e}")
            return []

# Legacy classes for backward compatibility
class ProductLookupClient:
    """Legacy wrapper for product lookups"""
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.client = LookupServiceClient(base_url)

    async def top_docs(self, query: str, k: int = 4):
        return await self.client.search_products(query, k)

    async def find_recommended_items_in_response(self, candidates: List, response: str, sim_threshold: float = 0.70):
        return await self.client.find_recommended_items_in_response(candidates, response, sim_threshold)

    async def close(self):
        await self.client.close()

class BuyingGuideClient:
    """Legacy wrapper for buying guide lookups"""
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.client = LookupServiceClient(base_url)

    async def top_docs(self, query: str, k: int = 4):
        return await self.client.search_buying_guides(query, k)

    async def close(self):
        await self.client.close()