"""
Unit tests for competitive intelligence normalization and scoring.

Tests _compute_vulnerability_score, _normalize_feature_requests (pure functions)
and _normalize_competitors (async, uses a mock pool for brand lookups).
"""

import pytest

from atlas_brain.autonomous.tasks.competitive_intelligence import (
    _compute_vulnerability_score,
    _normalize_competitors,
    _normalize_feature_requests,
)


# ------------------------------------------------------------------
# Mock pool for _normalize_competitors brand lookup
# ------------------------------------------------------------------

class _MockPool:
    """Minimal mock that returns canned brand rows for the DISTINCT brand query."""

    def __init__(self, brands: list[str]):
        self._brands = brands

    async def fetch(self, query, *args):
        return [{"brand": b} for b in self._brands]


# ------------------------------------------------------------------
# _compute_vulnerability_score
# ------------------------------------------------------------------

class TestVulnerabilityScore:
    def test_worst_case(self):
        """No repurchase, max pain, lowest rating -> near max vulnerability."""
        score = _compute_vulnerability_score({
            "repurchase_yes": 0,
            "repurchase_no": 100,
            "avg_pain_score": 10.0,
            "avg_rating": 1.0,
        })
        assert score == 96.0

    def test_best_case(self):
        """All repurchase, no pain, perfect rating -> 0 vulnerability."""
        score = _compute_vulnerability_score({
            "repurchase_yes": 100,
            "repurchase_no": 0,
            "avg_pain_score": 0.0,
            "avg_rating": 5.0,
        })
        assert score == 0.0

    def test_no_signal_defaults(self):
        """No repurchase data uses 0.5 defaults -> midrange score."""
        score = _compute_vulnerability_score({
            "repurchase_yes": 0,
            "repurchase_no": 0,
            "avg_pain_score": 5.0,
            "avg_rating": 3.0,
        })
        # (1-0.5)*35 + 5/10*35 + (5-3)/5*20 + 0.5*10 = 17.5 + 17.5 + 8 + 5 = 48
        assert score == 48.0

    def test_missing_keys_use_defaults(self):
        """Empty dict uses all defaults (pain=5, rating=3, signal=0.5)."""
        score = _compute_vulnerability_score({})
        assert score == 48.0

    def test_clamped_to_range(self):
        """Score is always between 0 and 100."""
        score = _compute_vulnerability_score({
            "repurchase_yes": 1000,
            "repurchase_no": 0,
            "avg_pain_score": 0.0,
            "avg_rating": 5.0,
        })
        assert 0.0 <= score <= 100.0

    def test_typical_complaint_data(self):
        """Typical complaint skew: low repurchase, high pain, low rating."""
        score = _compute_vulnerability_score({
            "repurchase_yes": 5,
            "repurchase_no": 20,
            "avg_pain_score": 7.5,
            "avg_rating": 2.0,
        })
        # (1-0.2)*35 + 7.5/10*35 + (5-2)/5*20 + 0.8*10 = 28+26.25+12+8 = 74.25
        assert score == 74.25

    def test_higher_pain_means_higher_score(self):
        """Increasing pain should increase vulnerability."""
        base = {"repurchase_yes": 10, "repurchase_no": 10, "avg_rating": 3.0}
        low_pain = _compute_vulnerability_score({**base, "avg_pain_score": 2.0})
        high_pain = _compute_vulnerability_score({**base, "avg_pain_score": 8.0})
        assert high_pain > low_pain

    def test_lower_rating_means_higher_score(self):
        """Lower ratings should increase vulnerability."""
        base = {"repurchase_yes": 10, "repurchase_no": 10, "avg_pain_score": 5.0}
        good_rating = _compute_vulnerability_score({**base, "avg_rating": 4.5})
        bad_rating = _compute_vulnerability_score({**base, "avg_rating": 1.5})
        assert bad_rating > good_rating


# ------------------------------------------------------------------
# _normalize_feature_requests
# ------------------------------------------------------------------

class TestNormalizeFeatureRequests:
    def test_case_dedup(self):
        """Same feature in different cases should merge."""
        gaps = [
            {"feature": "USB-C port", "category": "peripherals", "mentions": 5, "avg_pain_score": 6.0},
            {"feature": "usb-c port", "category": "peripherals", "mentions": 3, "avg_pain_score": 4.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert len(result) == 1
        assert result[0]["mentions"] == 8
        # Keeps higher pain score
        assert result[0]["avg_pain_score"] == 6.0

    def test_whitespace_collapse(self):
        """Extra whitespace should not create separate entries."""
        gaps = [
            {"feature": "better  build quality", "category": "tools", "mentions": 3, "avg_pain_score": 5.0},
            {"feature": "better build quality", "category": "tools", "mentions": 2, "avg_pain_score": 7.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert len(result) == 1
        assert result[0]["mentions"] == 5
        assert result[0]["avg_pain_score"] == 7.0

    def test_different_categories_stay_separate(self):
        """Same feature in different categories should not merge."""
        gaps = [
            {"feature": "longer warranty", "category": "electronics", "mentions": 4, "avg_pain_score": 5.0},
            {"feature": "longer warranty", "category": "appliances", "mentions": 3, "avg_pain_score": 6.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert len(result) == 2

    def test_empty_features_skipped(self):
        """Entries with empty or None features are dropped."""
        gaps = [
            {"feature": "", "category": "x", "mentions": 5, "avg_pain_score": 5.0},
            {"feature": None, "category": "x", "mentions": 3, "avg_pain_score": 5.0},
            {"feature": "real feature", "category": "x", "mentions": 2, "avg_pain_score": 5.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert len(result) == 1
        assert result[0]["feature"] == "real feature"

    def test_sorted_by_mentions_desc(self):
        """Results should be sorted by mentions descending."""
        gaps = [
            {"feature": "feature A", "category": "x", "mentions": 2, "avg_pain_score": 5.0},
            {"feature": "feature B", "category": "x", "mentions": 10, "avg_pain_score": 5.0},
            {"feature": "feature C", "category": "x", "mentions": 5, "avg_pain_score": 5.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert [r["mentions"] for r in result] == [10, 5, 2]

    def test_empty_input(self):
        """Empty list returns empty list."""
        assert _normalize_feature_requests([]) == []

    def test_preserves_first_casing_for_display(self):
        """The display name should keep the casing of the first occurrence."""
        gaps = [
            {"feature": "USB-C Port", "category": "x", "mentions": 3, "avg_pain_score": 5.0},
            {"feature": "usb-c port", "category": "x", "mentions": 2, "avg_pain_score": 5.0},
        ]
        result = _normalize_feature_requests(gaps)
        assert result[0]["feature"] == "USB-C Port"


# ------------------------------------------------------------------
# _normalize_competitors
# ------------------------------------------------------------------

class TestNormalizeCompetitors:
    @pytest.mark.asyncio
    async def test_noise_filtered(self):
        """Generic noise words are removed."""
        pool = _MockPool(["Logitech"])
        flows = [
            {"source_brand": "A", "competitor": "iTunes", "direction": "switched_to", "mentions": 5},
            {"source_brand": "A", "competitor": "competitor", "direction": "switched_to", "mentions": 3},
            {"source_brand": "A", "competitor": "Amazon", "direction": "switched_to", "mentions": 4},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_short_names_filtered(self):
        """Names under 3 chars are filtered unless they are a known brand."""
        pool = _MockPool(["LG", "Sony"])
        flows = [
            {"source_brand": "A", "competitor": "AB", "direction": "switched_to", "mentions": 5},
            {"source_brand": "A", "competitor": "XY", "direction": "switched_to", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_known_brand_short_name_kept(self):
        """2-char brand 'LG' is kept because it matches known brands."""
        pool = _MockPool(["LG"])
        flows = [
            {"source_brand": "Sony", "competitor": "LG", "direction": "switched_to", "mentions": 5},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 1
        assert result[0]["competitor"] == "LG"

    @pytest.mark.asyncio
    async def test_brand_aware_collapse(self):
        """Long model strings collapse to just the brand."""
        pool = _MockPool(["Kingston"])
        flows = [
            {"source_brand": "SanDisk", "competitor": "Kingston 16GB SDHC Class 4", "direction": "switched_to", "mentions": 3},
            {"source_brand": "SanDisk", "competitor": "Kingston 2GB", "direction": "switched_to", "mentions": 2},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 1
        assert result[0]["competitor"] == "Kingston"
        assert result[0]["mentions"] == 5

    @pytest.mark.asyncio
    async def test_case_dedup(self):
        """'Harmony One' and 'Harmony ONE' collapse to the same entry."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "Logitech", "competitor": "Harmony One", "direction": "considered", "mentions": 4},
            {"source_brand": "Logitech", "competitor": "Harmony ONE", "direction": "considered", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 1
        assert result[0]["competitor"] == "Harmony One"
        assert result[0]["mentions"] == 7

    @pytest.mark.asyncio
    async def test_preserves_original_brand_casing(self):
        """Brand casing comes from DB, not .title()."""
        pool = _MockPool(["LG"])
        flows = [
            {"source_brand": "Sony", "competitor": "lg flatron", "direction": "switched_to", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert result[0]["competitor"] == "LG"

    @pytest.mark.asyncio
    async def test_numeric_strings_filtered(self):
        """Purely numeric competitor names are dropped."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "A", "competitor": "12345", "direction": "switched_to", "mentions": 5},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_empty_competitor_skipped(self):
        """Empty or None competitor names are skipped."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "A", "competitor": "", "direction": "switched_to", "mentions": 5},
            {"source_brand": "A", "competitor": None, "direction": "switched_to", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_different_directions_stay_separate(self):
        """Same competitor with different directions should not merge."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "A", "competitor": "Brand X", "direction": "switched_to", "mentions": 5},
            {"source_brand": "A", "competitor": "Brand X", "direction": "considered", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_sorted_by_mentions_desc(self):
        """Results sorted by mentions descending."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "A", "competitor": "Brand X", "direction": "switched_to", "mentions": 2},
            {"source_brand": "A", "competitor": "Brand Y", "direction": "switched_to", "mentions": 10},
            {"source_brand": "A", "competitor": "Brand Z", "direction": "switched_to", "mentions": 5},
        ]
        result = await _normalize_competitors(flows, pool)
        assert [r["mentions"] for r in result] == [10, 5, 2]

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty list returns empty list."""
        pool = _MockPool([])
        result = await _normalize_competitors([], pool)
        assert result == []

    @pytest.mark.asyncio
    async def test_different_source_brands_stay_separate(self):
        """Same competitor from different sources should not merge."""
        pool = _MockPool([])
        flows = [
            {"source_brand": "A", "competitor": "Brand X", "direction": "switched_to", "mentions": 5},
            {"source_brand": "B", "competitor": "Brand X", "direction": "switched_to", "mentions": 3},
        ]
        result = await _normalize_competitors(flows, pool)
        assert len(result) == 2
