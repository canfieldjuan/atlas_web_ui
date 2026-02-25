"""
TrustRadius parser for B2B review scraping.

URL pattern: trustradius.com/products/{slug}/reviews?page={n}
Uses BeautifulSoup for HTML parsing. Rating scale is 1-10.
Residential proxy recommended.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..client import AntiDetectionClient
from . import ScrapeResult, ScrapeTarget, register_parser

logger = logging.getLogger("atlas.services.scraping.parsers.trustradius")

_DOMAIN = "trustradius.com"
_BASE_URL = "https://www.trustradius.com/products"


class TrustRadiusParser:
    """Parse TrustRadius review pages."""

    source_name = "trustradius"
    prefer_residential = True

    async def scrape(self, target: ScrapeTarget, client: AntiDetectionClient) -> ScrapeResult:
        """Scrape TrustRadius reviews for the given product."""
        reviews: list[dict] = []
        errors: list[str] = []
        pages_scraped = 0
        seen_ids: set[str] = set()

        for page in range(1, target.max_pages + 1):
            url = f"{_BASE_URL}/{target.product_slug}/reviews?page={page}"
            referer = (
                f"https://www.google.com/search?q={quote_plus(target.vendor_name)}+trustradius+reviews"
                if page == 1
                else f"{_BASE_URL}/{target.product_slug}/reviews?page={page - 1}"
            )

            try:
                resp = await client.get(
                    url,
                    domain=_DOMAIN,
                    referer=referer,
                    sticky_session=True,
                    prefer_residential=True,
                )
                pages_scraped += 1

                if resp.status_code == 403:
                    errors.append(f"Page {page}: blocked (403)")
                    break
                if resp.status_code != 200:
                    errors.append(f"Page {page}: HTTP {resp.status_code}")
                    continue

                page_reviews = _parse_page(resp.text, target, seen_ids)
                if not page_reviews:
                    break  # No more reviews

                reviews.extend(page_reviews)

            except Exception as exc:
                errors.append(f"Page {page}: {exc}")
                logger.warning("TrustRadius page %d failed for %s: %s", page, target.product_slug, exc)
                break

        logger.info(
            "TrustRadius scrape for %s: %d reviews from %d pages",
            target.vendor_name, len(reviews), pages_scraped,
        )

        return ScrapeResult(reviews=reviews, pages_scraped=pages_scraped, errors=errors)


def _parse_page(
    html: str, target: ScrapeTarget, seen_ids: set[str]
) -> list[dict]:
    """Parse a single TrustRadius review page."""
    soup = BeautifulSoup(html, "html.parser")
    reviews: list[dict] = []

    # TrustRadius review cards are in div elements with review data
    review_cards = soup.select('[data-testid="review-card"], .review-card, [class*="ReviewCard"]')

    # Fallback: look for common review container patterns
    if not review_cards:
        review_cards = soup.select('div[id^="review-"], article[class*="review"]')

    for card in review_cards:
        try:
            review = _parse_review_card(card, target)
            if review and review.get("source_review_id") not in seen_ids:
                seen_ids.add(review["source_review_id"])
                reviews.append(review)
        except Exception:
            logger.debug("Failed to parse TrustRadius review card", exc_info=True)

    return reviews


def _parse_review_card(card, target: ScrapeTarget) -> dict | None:
    """Extract review data from a single TrustRadius review card."""
    # Extract review ID
    review_id = card.get("id", "") or card.get("data-review-id", "")
    if not review_id:
        # Try to find any unique identifier
        link = card.select_one("a[href*='/reviews/']")
        if link:
            href = link.get("href", "")
            parts = href.rstrip("/").split("/")
            review_id = parts[-1] if parts else ""
    if not review_id:
        return None

    # Extract rating (TrustRadius uses 1-10 scale)
    rating = None
    rating_el = card.select_one('[class*="rating"], [class*="score"], [data-rating]')
    if rating_el:
        rating_text = rating_el.get("data-rating") or rating_el.get_text(strip=True)
        rating_match = re.search(r"(\d+(?:\.\d+)?)", rating_text or "")
        if rating_match:
            rating = float(rating_match.group(1))

    # Extract review text
    review_text = ""
    text_el = card.select_one('[class*="review-text"], [class*="ReviewText"], .review-body')
    if text_el:
        review_text = text_el.get_text(strip=True)

    # Extract pros/cons if available
    pros = _extract_section(card, "like", "pros", "best")
    cons = _extract_section(card, "dislike", "cons", "worst")

    # Combine for review_text if main text is empty
    if not review_text:
        parts = []
        if pros:
            parts.append(f"Pros: {pros}")
        if cons:
            parts.append(f"Cons: {cons}")
        review_text = "\n".join(parts)

    if not review_text or len(review_text) < 20:
        return None

    # Extract reviewer info
    reviewer_name = ""
    reviewer_el = card.select_one('[class*="reviewer"], [class*="author"], [class*="Reviewer"]')
    if reviewer_el:
        reviewer_name = reviewer_el.get_text(strip=True)

    reviewer_title = ""
    title_el = card.select_one('[class*="title"], [class*="role"], [class*="Title"]')
    if title_el and title_el != reviewer_el:
        reviewer_title = title_el.get_text(strip=True)

    reviewer_company = ""
    company_el = card.select_one('[class*="company"], [class*="org"], [class*="Company"]')
    if company_el:
        reviewer_company = company_el.get_text(strip=True)

    company_size = ""
    size_el = card.select_one('[class*="size"], [class*="employees"]')
    if size_el:
        company_size = size_el.get_text(strip=True)

    # Extract date
    reviewed_at = None
    date_el = card.select_one("time, [class*='date'], [class*='Date']")
    if date_el:
        date_str = date_el.get("datetime") or date_el.get_text(strip=True)
        reviewed_at = date_str

    # Build source URL
    source_url = f"https://www.trustradius.com/products/{target.product_slug}/reviews#{review_id}"

    return {
        "source": "trustradius",
        "source_url": source_url,
        "source_review_id": review_id,
        "vendor_name": target.vendor_name,
        "product_name": target.product_name,
        "product_category": target.product_category,
        "rating": rating,
        "rating_max": 10,  # TrustRadius uses 10-point scale
        "summary": None,
        "review_text": review_text[:10000],
        "pros": pros,
        "cons": cons,
        "reviewer_name": reviewer_name or None,
        "reviewer_title": reviewer_title or None,
        "reviewer_company": reviewer_company or None,
        "company_size_raw": company_size or None,
        "reviewer_industry": None,
        "reviewed_at": reviewed_at,
        "raw_metadata": {},
    }


def _extract_section(card, *keywords: str) -> str | None:
    """Extract a pros/cons section by looking for keyword matches in class names."""
    for kw in keywords:
        el = card.select_one(f'[class*="{kw}"]')
        if el:
            text = el.get_text(strip=True)
            if text and len(text) > 5:
                return text[:5000]
    return None


# Auto-register
register_parser(TrustRadiusParser())
