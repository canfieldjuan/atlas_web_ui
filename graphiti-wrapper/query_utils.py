"""
Query processing utilities for enhanced search.

Provides query expansion, classification, and temporal detection.
No external dependencies - pure Python pattern matching.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes"):
        return True
    if val in ("false", "0", "no"):
        return False
    return default


def get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


@dataclass
class QueryClassification:
    """Result of query classification."""
    is_math: bool = False
    is_datetime: bool = False
    is_web_search: bool = False
    is_tool_specific: bool = False
    should_skip_search: bool = False
    reason: Optional[str] = None
    detected_pattern: Optional[str] = None


@dataclass
class ExpandedQuery:
    """Result of query expansion."""
    original: str
    variants: list = field(default_factory=list)
    transformations_applied: list = field(default_factory=list)


@dataclass
class TemporalIntent:
    """Detected temporal intent from query."""
    is_historical: Optional[bool] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    requires_latest: bool = False
    data_source_type: Optional[str] = None


# ============================================================================
# Query Classification
# ============================================================================

def classify_query(query: str) -> QueryClassification:
    """
    Classify query to determine if GraphRAG search should be skipped.

    Math, datetime, and web search queries are better handled by tools.
    """
    if not query or not isinstance(query, str):
        return QueryClassification(reason="Invalid query")

    q = query.lower().strip()

    skip_math = get_env_bool("GRAPHRAG_SKIP_MATH_QUERIES", True)
    skip_datetime = get_env_bool("GRAPHRAG_SKIP_DATETIME_QUERIES", True)
    skip_web = get_env_bool("GRAPHRAG_SKIP_WEBSEARCH_QUERIES", True)

    is_math = skip_math and _detect_math_query(q)
    is_datetime = skip_datetime and _detect_datetime_query(q)
    is_web_search = skip_web and _detect_web_search_query(q)

    is_tool_specific = is_math or is_datetime or is_web_search

    return QueryClassification(
        is_math=is_math,
        is_datetime=is_datetime,
        is_web_search=is_web_search,
        is_tool_specific=is_tool_specific,
        should_skip_search=is_tool_specific,
        reason=_get_skip_reason(is_math, is_datetime, is_web_search),
        detected_pattern=_get_detected_pattern(q, is_math, is_datetime, is_web_search),
    )


def _detect_math_query(query: str) -> bool:
    """Detect if query is a pure mathematical calculation."""
    knowledge_pattern = re.compile(
        r"\b(tell me about|what is .{10,}|explain|describe|about|"
        r"information|details|how does|why|who|where)\b",
        re.IGNORECASE
    )
    if knowledge_pattern.search(query):
        text_only = re.sub(r"[\d\s\+\-\*\/\%\(\)\.]+", "", query).strip()
        if len(text_only) > 20:
            return False

    if re.match(r"^[\d\s\+\-\*\/\%\(\)\.]+$", query):
        return True

    if re.match(
        r"^(what is|how much is|calculate|compute)\s+[\d\s\+\-\*\/\%\(\)\.]+\??$",
        query,
        re.IGNORECASE
    ):
        return True

    if re.match(r"^\d+\s*%\s*of\s*\d+\??$", query, re.IGNORECASE):
        return True

    return False


def _detect_datetime_query(query: str) -> bool:
    """Detect if query is about date/time."""
    if re.match(
        r"^(what time|what's the time|current time|what.*time is it)",
        query,
        re.IGNORECASE
    ):
        return True

    if re.match(
        r"^(what.*date|what's the date|current date|today's date)",
        query,
        re.IGNORECASE
    ):
        return True

    if re.search(
        r"(what time in|time in.*timezone|convert.*time)",
        query,
        re.IGNORECASE
    ):
        return True

    return False


def _detect_web_search_query(query: str) -> bool:
    """Detect if query is a web search request."""
    if re.match(
        r"^(search for|search the web|look up|find on web|google)",
        query,
        re.IGNORECASE
    ):
        return True

    if re.search(
        r"(latest|current|recent|breaking).*news",
        query,
        re.IGNORECASE
    ):
        return True

    return False


def _get_skip_reason(is_math: bool, is_datetime: bool, is_web_search: bool) -> Optional[str]:
    """Generate human-readable reason for skipping search."""
    if is_math:
        return "Math calculation - calculator tool appropriate"
    if is_datetime:
        return "DateTime query - datetime tool appropriate"
    if is_web_search:
        return "Web search query - web_search tool appropriate"
    return None


def _get_detected_pattern(
    query: str,
    is_math: bool,
    is_datetime: bool,
    is_web_search: bool
) -> Optional[str]:
    """Get the detected pattern for logging."""
    if is_math:
        if re.search(r"\d+\s*%\s*of\s*\d+", query, re.IGNORECASE):
            return "percentage_calculation"
        if re.search(r"\d+\s*[\+\-\*\/\%]\s*\d+", query):
            return "math_operator"
        if re.match(r"^(calculate|compute)", query, re.IGNORECASE):
            return "calculate_keyword"
        return "arithmetic_expression"
    if is_datetime:
        return "datetime_query"
    if is_web_search:
        return "web_search_request"
    return None


# ============================================================================
# Query Expansion
# ============================================================================

def expand_query(query: str) -> ExpandedQuery:
    """
    Expand a user query into multiple search variants.

    Strips command prefixes while preserving search intent.
    """
    if not query or not isinstance(query, str):
        return ExpandedQuery(original=query or "", variants=[], transformations_applied=[])

    trimmed = query.strip()
    variants = [trimmed]
    transformations = []

    match = re.match(r"^search\s+(for\s+|the\s+web\s+for\s+)?(.+)$", trimmed, re.IGNORECASE)
    if match and match.group(2):
        without_prefix = match.group(2).strip()
        if without_prefix and without_prefix not in variants:
            variants.append(without_prefix)
            transformations.append("remove_search_prefix")

    match = re.match(r"^(.+?)\s+in\s+my\s+documents?\s*$", trimmed, re.IGNORECASE)
    if match and match.group(1):
        without_suffix = match.group(1).strip()
        if without_suffix and without_suffix not in variants:
            variants.append(without_suffix)
            transformations.append("remove_document_suffix")

    match = re.match(r"^look\s+up\s+(.+)$", trimmed, re.IGNORECASE)
    if match and match.group(1):
        without_prefix = match.group(1).strip()
        if without_prefix and without_prefix not in variants:
            variants.append(without_prefix)
            transformations.append("remove_lookup_prefix")

    match = re.match(
        r"^find\s+(?:me\s+)?(?:information\s+(?:about|on)\s+)?(.+)$",
        trimmed,
        re.IGNORECASE
    )
    if match and match.group(1):
        without_prefix = match.group(1).strip()
        if len(without_prefix) > 3 and without_prefix not in variants:
            variants.append(without_prefix)
            transformations.append("remove_find_prefix")

    match = re.match(r"^tell\s+me\s+(?:about\s+)?(.+)$", trimmed, re.IGNORECASE)
    if match and match.group(1):
        without_prefix = match.group(1).strip()
        if without_prefix and without_prefix not in variants:
            variants.append(without_prefix)
            transformations.append("remove_tell_me_prefix")

    match = re.match(r"^what\s+(?:is|are)\s+(?:the\s+)?(.+?)(?:\?)?$", trimmed, re.IGNORECASE)
    if match and match.group(1):
        without_prefix = match.group(1).strip()
        if len(without_prefix) > 2 and without_prefix not in variants:
            variants.append(without_prefix)
            transformations.append("remove_what_is_prefix")

    quoted_match = re.search(r'"([^"]+)"', trimmed)
    if quoted_match and quoted_match.group(1):
        quoted_term = quoted_match.group(1).strip()
        if quoted_term and quoted_term not in variants:
            variants.append(quoted_term)
            transformations.append("extract_quoted_term")

    return ExpandedQuery(
        original=trimmed,
        variants=variants,
        transformations_applied=transformations,
    )


def get_best_variant(expanded: ExpandedQuery) -> str:
    """Get the best variant for searching."""
    if not expanded.variants:
        return expanded.original

    if expanded.transformations_applied and len(expanded.variants) > 1:
        return expanded.variants[-1]

    return expanded.variants[0]


def should_expand_query(query: str) -> bool:
    """Check if a query would benefit from expansion."""
    if not query or not isinstance(query, str):
        return False

    patterns = [
        r"^search\s+",
        r"^look\s+up\s+",
        r"^find\s+",
        r"^tell\s+me\s+",
        r"^what\s+(?:is|are)\s+",
        r"^when\s+(?:was|were|did)\s+",
        r"in\s+my\s+documents?\s*$",
        r'"[^"]+"',
    ]

    trimmed = query.strip()
    for pattern in patterns:
        if re.search(pattern, trimmed, re.IGNORECASE):
            return True

    return False


# ============================================================================
# Temporal Classification
# ============================================================================

LATEST_PATTERNS = [
    re.compile(r"\b(latest|newest|recent|current|today|now)\b", re.IGNORECASE),
    re.compile(r"\b(this week|this month|this year)\b", re.IGNORECASE),
    re.compile(r"\b(up to date|up-to-date)\b", re.IGNORECASE),
]

HISTORICAL_PATTERNS = [
    re.compile(
        r"\b(history|historical|past|previous|old|older|archive|archived)\b",
        re.IGNORECASE
    ),
    re.compile(r"\b(used to|formerly|originally)\b", re.IGNORECASE),
    re.compile(r"\b(back in|back when|years ago|months ago)\b", re.IGNORECASE),
]

RELATIVE_DATE_PATTERNS = [
    (re.compile(r"\blast week\b", re.IGNORECASE), 7),
    (re.compile(r"\blast month\b", re.IGNORECASE), 30),
    (re.compile(r"\blast year\b", re.IGNORECASE), 365),
    (re.compile(r"\byesterday\b", re.IGNORECASE), 1),
]

DYNAMIC_DAYS_PATTERN = re.compile(r"\blast (\d+) days?\b", re.IGNORECASE)
EXPLICIT_DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def detect_temporal_intent(query: str) -> TemporalIntent:
    """
    Detect temporal intent from query string.

    Returns filters for is_historical, date_from, date_to.
    """
    intent = TemporalIntent()
    lower_query = query.lower()

    default_latest_days = get_env_int("GRAPHRAG_TEMPORAL_LATEST_DAYS", 30)

    for pattern in LATEST_PATTERNS:
        if pattern.search(lower_query):
            intent.requires_latest = True
            intent.date_from = _get_date_days_ago(default_latest_days)
            break

    for pattern in HISTORICAL_PATTERNS:
        if pattern.search(lower_query):
            intent.is_historical = True
            break

    for pattern, days_delta in RELATIVE_DATE_PATTERNS:
        if pattern.search(lower_query):
            intent.date_from = _get_date_days_ago(days_delta)
            break

    dynamic_match = DYNAMIC_DAYS_PATTERN.search(lower_query)
    if dynamic_match:
        days = int(dynamic_match.group(1))
        intent.date_from = _get_date_days_ago(days)

    explicit_match = EXPLICIT_DATE_PATTERN.search(query)
    if explicit_match:
        date_str = explicit_match.group(0)
        try:
            parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
            if parsed_date < datetime.now():
                intent.date_from = date_str
        except ValueError:
            pass

    return intent


def _get_date_days_ago(days: int) -> str:
    """Get ISO date string for N days ago."""
    date = datetime.now() - timedelta(days=days)
    return date.strftime("%Y-%m-%d")


# ============================================================================
# Query Decomposition
# ============================================================================

@dataclass
class SubQuery:
    """A sub-query from decomposition."""
    query: str
    priority: int
    requires_graph: bool = True
    query_type: str = "original"


@dataclass
class DecomposedQuery:
    """Result of query decomposition."""
    original: str
    is_complex: bool
    sub_queries: list
    complexity_reason: Optional[str] = None


COMPARISON_PATTERNS = [
    re.compile(r"\b(vs\.?|versus|compare|compared to|difference between)\b", re.IGNORECASE),
    re.compile(r"\b(better than|worse than|similar to)\b", re.IGNORECASE),
    re.compile(r"\b(pros and cons|advantages and disadvantages)\b", re.IGNORECASE),
]

MULTI_PART_PATTERNS = [
    re.compile(r"\b(and also|additionally|furthermore|moreover)\b", re.IGNORECASE),
    re.compile(r"\b(first|second|third|finally)\b", re.IGNORECASE),
]

QUESTION_CHAIN_PATTERN = re.compile(r"\?.*\?")
CONJUNCTION_SPLIT_PATTERN = re.compile(r"\s+and\s+(?=\w)", re.IGNORECASE)


class QueryDecomposer:
    """
    Breaks complex queries into simpler sub-queries for better retrieval.

    Handles:
    - Comparison queries: "A vs B", "compare X and Y"
    - Question chains: "What is X? How does Y work?"
    - Conjunction queries: "Tell me about X and Y"
    """

    def decompose(self, query: str) -> DecomposedQuery:
        """Decompose a query into sub-queries if complex."""
        has_comparison = self._matches_any(query, COMPARISON_PATTERNS)
        has_multi_parts = self._matches_any(query, MULTI_PART_PATTERNS)
        has_question_chain = bool(QUESTION_CHAIN_PATTERN.search(query))
        has_conjunction = bool(CONJUNCTION_SPLIT_PATTERN.search(query))

        is_complex = has_comparison or has_multi_parts or has_question_chain or has_conjunction

        if not is_complex:
            return DecomposedQuery(
                original=query,
                is_complex=False,
                sub_queries=[SubQuery(query=query, priority=1, query_type="original")],
            )

        complexity_reason = ""
        if has_comparison:
            complexity_reason = "comparison"
        elif has_question_chain:
            complexity_reason = "multiple_questions"
        elif has_multi_parts:
            complexity_reason = "multi_part"
        elif has_conjunction:
            complexity_reason = "conjunction"

        sub_queries = self._create_sub_queries(query, complexity_reason)

        return DecomposedQuery(
            original=query,
            is_complex=True,
            sub_queries=sub_queries,
            complexity_reason=complexity_reason,
        )

    def _create_sub_queries(self, query: str, reason: str) -> list:
        """Create sub-queries based on complexity type."""
        if reason == "comparison":
            return self._decompose_comparison(query)
        elif reason == "multiple_questions":
            return self._decompose_question_chain(query)
        elif reason in ("multi_part", "conjunction"):
            return self._decompose_conjunction(query)
        return [SubQuery(query=query, priority=1, query_type="original")]

    def _decompose_comparison(self, query: str) -> list:
        """Decompose comparison queries (e.g., 'A vs B')."""
        entities = self._extract_comparison_entities(query)

        if len(entities) < 2:
            return [SubQuery(query=query, priority=1, query_type="original")]

        return [
            SubQuery(
                query=f"Information about {entity}",
                priority=idx + 1,
                query_type="comparison",
            )
            for idx, entity in enumerate(entities)
        ]

    def _decompose_question_chain(self, query: str) -> list:
        """Decompose question chains (multiple questions)."""
        questions = [q.strip() for q in query.split("?") if len(q.strip()) > 10]

        if len(questions) <= 1:
            return [SubQuery(query=query, priority=1, query_type="original")]

        return [
            SubQuery(
                query=q if q.endswith("?") else f"{q}?",
                priority=idx + 1,
                query_type="part",
            )
            for idx, q in enumerate(questions)
        ]

    def _decompose_conjunction(self, query: str) -> list:
        """Decompose conjunction queries (A and B)."""
        parts = CONJUNCTION_SPLIT_PATTERN.split(query)

        if len(parts) <= 1:
            return [SubQuery(query=query, priority=1, query_type="original")]

        return [
            SubQuery(query=part.strip(), priority=idx + 1, query_type="part")
            for idx, part in enumerate(parts)
            if part.strip()
        ]

    def _extract_comparison_entities(self, query: str) -> list:
        """Extract entities from comparison query."""
        vs_match = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)", query, re.IGNORECASE)
        if vs_match:
            return [vs_match.group(1).strip(), vs_match.group(2).strip()]

        compare_match = re.search(
            r"compare\s+(.+?)\s+(?:and|to|with)\s+(.+?)(?:\?|$)",
            query,
            re.IGNORECASE
        )
        if compare_match:
            return [compare_match.group(1).strip(), compare_match.group(2).strip()]

        diff_match = re.search(
            r"difference\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
            query,
            re.IGNORECASE
        )
        if diff_match:
            return [diff_match.group(1).strip(), diff_match.group(2).strip()]

        entities = self._extract_capitalized_entities(query)
        return entities[:2]

    def _extract_capitalized_entities(self, query: str) -> list:
        """Extract capitalized words as potential entities."""
        matches = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", query)
        return list(dict.fromkeys(matches))

    def _matches_any(self, query: str, patterns: list) -> bool:
        """Check if query matches any pattern."""
        return any(pattern.search(query) for pattern in patterns)

    def should_decompose(self, query: str) -> bool:
        """Estimate if decomposition is worth the overhead."""
        if len(query) < 30:
            return False

        word_count = len(query.split())
        if word_count < 5:
            return False

        decomposed = self.decompose(query)
        return decomposed.is_complex and len(decomposed.sub_queries) > 1


query_decomposer = QueryDecomposer()
