"""Query analysis for the codebase intelligence system."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured representation of an analyzed user query."""

    original_query: str
    normalized_query: str
    intent: str
    symbols: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)


class QueryAnalyzer:
    """Normalize queries and extract intent, symbols, and paths."""

    _HISTORICAL_PATTERN = re.compile(
        r"\b(?:previous|earlier|old|history|historical|"
        r"changed|change|changes|commit|commits|"
        r"before|after|introduced|removed|deprecated)\b",
        re.IGNORECASE,
    )

    _CONCEPTUAL_PATTERNS = (
        re.compile(r"\bhow\b", re.IGNORECASE),
        re.compile(r"\bwhy\b", re.IGNORECASE),
        re.compile(r"\barchitecture\b", re.IGNORECASE),
        re.compile(r"\bdesign\b", re.IGNORECASE),
        re.compile(r"\bflow\b", re.IGNORECASE),
        re.compile(r"\bworks?\b", re.IGNORECASE),
        re.compile(r"\brelationship\b", re.IGNORECASE),
        re.compile(r"\bpurpose\b", re.IGNORECASE),
    )

    _SYMBOL_CALL_PATTERN = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)"
    )

    _CLASS_SYMBOL_PATTERN = re.compile(
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _FUNCTION_SYMBOL_PATTERN = re.compile(
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _METHOD_SYMBOL_PATTERN = re.compile(
        r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _NAMED_CLASS_PATTERN = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+class\b",
        re.IGNORECASE,
    )

    _NAMED_FUNCTION_PATTERN = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+function\b",
        re.IGNORECASE,
    )

    _NAMED_METHOD_PATTERN = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+method\b",
        re.IGNORECASE,
    )

    _SYMBOL_STOPWORDS = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
    }

    _PATH_PATTERN = re.compile(
        r"(?<![\w.-])"
        r"(?:[A-Za-z0-9_.-]+/)+"
        r"[A-Za-z0-9_.-]+"
    )

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze one user query deterministically."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        normalized_query = self._normalize(query)

        if not normalized_query:
            raise ValueError("Query cannot be empty.")

        symbols = self._extract_symbols(normalized_query)
        paths = self._extract_paths(normalized_query)

        intent = self._detect_intent(
            normalized_query,
            symbols,
            paths,
        )

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            symbols=symbols,
            paths=paths,
        )

    def _normalize(self, query: str) -> str:
        """Normalize whitespace without changing query meaning."""

        return " ".join(query.strip().split())

    def _detect_intent(
        self,
        query: str,
        symbols: list[str],
        paths: list[str],
    ) -> str:
        """Determine the primary query intent."""

        if symbols:
            return "exact_symbol"

        if paths:
            return "path"

        if self._HISTORICAL_PATTERN.search(query):
            return "historical"

        if self._looks_conceptual(query):
            return "conceptual"

        return "general"

    def _looks_conceptual(self, query: str) -> bool:
        """Identify queries that ask how or why something works."""

        return any(
            pattern.search(query)
            for pattern in self._CONCEPTUAL_PATTERNS
        )

    def _extract_symbols(self, query: str) -> list[str]:
        """Extract explicitly referenced code symbols."""

        symbols: list[str] = []

        patterns = (
            self._SYMBOL_CALL_PATTERN,
            self._CLASS_SYMBOL_PATTERN,
            self._FUNCTION_SYMBOL_PATTERN,
            self._METHOD_SYMBOL_PATTERN,
            self._NAMED_CLASS_PATTERN,
            self._NAMED_FUNCTION_PATTERN,
            self._NAMED_METHOD_PATTERN,
        )

        for pattern in patterns:
            for match in pattern.finditer(query):
                symbol = match.group(1)

                if symbol.lower() in self._SYMBOL_STOPWORDS:
                    continue

                if symbol not in symbols:
                    symbols.append(symbol)

        return symbols

    def _extract_paths(self, query: str) -> list[str]:
        """Extract repository-style paths from a query."""

        paths: list[str] = []

        for match in self._PATH_PATTERN.finditer(query):
            path = match.group(0).rstrip(".,;:!?")

            if path not in paths:
                paths.append(path)

        return paths