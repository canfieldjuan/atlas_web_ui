"""
Anti-detection HTTP client for B2B review scraping.

Wraps curl_cffi to produce Chrome/Firefox-identical TLS handshakes,
combined with proxy rotation, browser profile consistency, and
per-domain rate limiting.
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Any

from curl_cffi.requests import AsyncSession, Response

from .profiles import BrowserProfileManager
from .proxy import ProxyConfig, ProxyManager
from .rate_limiter import DomainRateLimiter

logger = logging.getLogger("atlas.services.scraping.client")


class AntiDetectionClient:
    """HTTP client with TLS fingerprint spoofing and anti-detection measures."""

    def __init__(
        self,
        *,
        proxy_manager: ProxyManager,
        profile_manager: BrowserProfileManager,
        rate_limiter: DomainRateLimiter,
        min_delay: float = 2.0,
        max_delay: float = 8.0,
        max_retries: int = 2,
    ) -> None:
        self._proxy = proxy_manager
        self._profiles = profile_manager
        self._rate_limiter = rate_limiter
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._max_retries = max_retries

    async def get(
        self,
        url: str,
        *,
        domain: str,
        referer: str | None = None,
        sticky_session: bool = False,
        prefer_residential: bool = False,
    ) -> Response:
        """Fetch a URL with anti-detection measures.

        Args:
            url: Target URL.
            domain: Domain name for rate limiting and proxy selection.
            referer: Referer header (e.g. Google search URL or previous page).
            sticky_session: Reuse same proxy across calls for this domain.
            prefer_residential: Use residential proxy (for Cloudflare sites).

        Returns:
            curl_cffi Response object.
        """
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                # 1. Rate limit
                await self._rate_limiter.acquire(domain)

                # 2. Select browser profile
                profile = self._profiles.get_profile()

                # 3. Get proxy
                proxy = self._proxy.get_proxy(
                    domain=domain,
                    sticky=sticky_session,
                    prefer_residential=prefer_residential,
                )

                # 4. Build headers
                headers = profile.build_headers(
                    referer=referer,
                    proxy_geo=proxy.geo if proxy else None,
                )

                # 5. Random delay (human-like)
                delay = random.uniform(self._min_delay, self._max_delay)
                if attempt > 0:
                    # Exponential backoff on retries
                    delay *= (2 ** attempt)
                await asyncio.sleep(delay)

                # 6. Execute via curl_cffi with matching TLS fingerprint
                async with AsyncSession(impersonate=profile.impersonate) as session:
                    resp = await session.get(
                        url,
                        headers=headers,
                        proxy=proxy.url if proxy else None,
                        timeout=30,
                    )

                # Log non-200 responses
                if resp.status_code == 403:
                    logger.warning(
                        "Blocked (403) on %s attempt %d/%d (proxy=%s, profile=%s)",
                        domain, attempt + 1, self._max_retries + 1,
                        proxy.type if proxy else "none", profile.impersonate,
                    )
                    if attempt < self._max_retries:
                        # Clear sticky session on block so next attempt uses different proxy
                        self._proxy.clear_sticky(domain)
                        continue
                elif resp.status_code == 429:
                    logger.warning("Rate limited (429) on %s, backing off", domain)
                    await asyncio.sleep(30 + random.uniform(5, 15))
                    if attempt < self._max_retries:
                        continue

                return resp

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Request failed for %s attempt %d/%d: %s",
                    url, attempt + 1, self._max_retries + 1, exc,
                )
                if attempt < self._max_retries:
                    self._proxy.clear_sticky(domain)
                    await asyncio.sleep(random.uniform(2, 5))

        raise last_exc or RuntimeError(f"All retries exhausted for {url}")


# ---------------------------------------------------------------------------
# Module-level singleton (lazy init)
# ---------------------------------------------------------------------------

_client: AntiDetectionClient | None = None


def get_scrape_client() -> AntiDetectionClient:
    """Get or create the module-level scrape client singleton."""
    global _client
    if _client is None:
        from ...config import settings

        cfg = settings.b2b_scrape
        _client = AntiDetectionClient(
            proxy_manager=ProxyManager.from_config(cfg),
            profile_manager=BrowserProfileManager(),
            rate_limiter=DomainRateLimiter.from_config(cfg),
            min_delay=cfg.min_delay_seconds,
            max_delay=cfg.max_delay_seconds,
            max_retries=cfg.max_retries,
        )
    return _client
