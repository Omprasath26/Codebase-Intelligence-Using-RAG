"""Query analysis for the codebase intelligence system."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    """Structured representation of an analyzed user query."""

    original_query: str
    normalized_query: str
    intent: str

    symbols: list[str] = Field(default_factory=list)
    paths: list[str] = Field(default_factory=list)
    error_messages: list[str] = Field(default_factory=list)
    issue_numbers: list[str] = Field(default_factory=list)
    pr_numbers: list[str] = Field(default_factory=list)


class QueryAnalyzer:
    """Normalize queries and extract retrieval-relevant query entities."""

    _HISTORICAL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b(?:previous|earlier|old|history|historical|"
        r"changed|change|changes|commit|commits|"
        r"before|after|introduced|removed|deprecated|"
        r"originally|rationale)\b",
        re.IGNORECASE,
    )

    _LOCATION_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\b(?:where|location|located)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bimplemented\b",
            re.IGNORECASE,
        ),
    )

    _DEPENDENCY_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bdepend(?:s|ency|encies)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bimports?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdepends\s+on\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bused\s+by\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\buses?\b",
            re.IGNORECASE,
        ),
    )

    _DEBUGGING_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\b(?:bug|buggy|debug|debugging)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:error|exception|traceback)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fails?|failure|failing)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhy\s+(?:does|doesn't|isn't|is|did)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:fix|fixing|resolve|resolving)\b",
            re.IGNORECASE,
        ),
    )

    _CONFIGURATION_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bconfig(?:uration)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bsettings?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bconfigure\b",
            re.IGNORECASE,
        ),
    )

    _TESTING_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\btests?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btesting\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpytest\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\btest\s+case\b",
            re.IGNORECASE,
        ),
    )

    _COMPARISON_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bcompare\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bcomparison\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdifference(?:s)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bversus\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bvs\.?\b",
            re.IGNORECASE,
        ),
    )

    _IMPACT_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bimpact\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\baffect(?:s|ed|ing)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bconsequence(?:s)?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+breaks\b",
            re.IGNORECASE,
        ),
    )

    _EXPLANATION_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bexplain\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdescribe\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+does\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhat\s+is\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bwhy\s+(?:was|were|is|are|does|do)\b",
            re.IGNORECASE,
        ),
    )

    _CONCEPTUAL_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\barchitecture\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdesign\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brelationship\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bpurpose\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+does\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bhow\s+do\b",
            re.IGNORECASE,
        ),
    )

    _FLOW_PATTERNS: ClassVar[
        tuple[re.Pattern[str], ...]
    ] = (
        re.compile(
            r"\bflow\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bexecution\s+flow\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brequest\s+flow\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bprocess(?:ing)?\s+flow\b",
            re.IGNORECASE,
        ),
    )

    _SYMBOL_CALL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\(\)"
    )

    _CLASS_SYMBOL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _FUNCTION_SYMBOL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _METHOD_SYMBOL_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\bmethod\s+([A-Za-z_][A-Za-z0-9_]*)\b",
        re.IGNORECASE,
    )

    _NAMED_CLASS_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+class\b",
        re.IGNORECASE,
    )

    _NAMED_FUNCTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+function\b",
        re.IGNORECASE,
    )

    _NAMED_METHOD_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s+method\b",
        re.IGNORECASE,
    )

    _SYMBOL_STOPWORDS: ClassVar[set[str]] = {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
    }

    _PATH_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![\w.-])"
        r"(?:[A-Za-z0-9_.-]+/)+"
        r"[A-Za-z0-9_.-]+"
    )

    _ISSUE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![\w#])"
        r"(?:issue|issues)\s*#?\s*(\d+)",
        re.IGNORECASE,
    )

    _PR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?<![\w#])"
        r"(?:pr|pull\s+request|pull-request)"
        r"\s*#?\s*(\d+)",
        re.IGNORECASE,
    )

    _QUOTED_ERROR_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"""["']([^"']*(?:Error|Exception|Traceback)[^"']*)["']""",
        re.IGNORECASE,
    )

    _EXCEPTION_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\b("
        r"[A-Za-z_][A-Za-z0-9_.]*"
        r"(?:Error|Exception)"
        r")\b"
        r"(?::\s*"
        r"[^\n,.;!?\"']+"
        r")?",
        re.IGNORECASE,
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
        error_messages = self._extract_error_messages(
            normalized_query
        )
        issue_numbers = self._extract_issue_numbers(
            normalized_query
        )
        pr_numbers = self._extract_pr_numbers(
            normalized_query
        )

        intent = self._detect_intent(
            normalized_query,
            symbols,
            paths,
            error_messages,
            issue_numbers,
            pr_numbers,
        )

        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            symbols=symbols,
            paths=paths,
            error_messages=error_messages,
            issue_numbers=issue_numbers,
            pr_numbers=pr_numbers,
        )

    def _normalize(self, query: str) -> str:
        """Normalize whitespace without changing identifiers."""

        return " ".join(query.strip().split())

    def _detect_intent(self,query: str,symbols: list[str],paths: list[str],error_messages: list[str],issue_numbers: list[str],pr_numbers: list[str]) -> str:
        """Determine the primary query intent."""

        # Error/exception evidence takes highest priority.
        if error_messages:
            return "debugging"

        # Explicit comparison and impact questions are unambiguous.
        if self._looks_comparison(query):
            return "comparison"

        if self._looks_impact(query):
            return "impact_analysis"

        # Explicit dependency/configuration/testing questions.
        if self._looks_dependency(query):
            return "dependency"

        if self._looks_configuration(query):
            return "configuration"

        if self._looks_testing(query):
            return "testing"

        # Flow is more specific than generic conceptual language.
        if self._looks_flow(query):
            return "flow"

        # Historical questions should be classified before generic
        # explanation language.
        if self._looks_historical(query):
            return "historical"

        # Explicit symbol/path location questions.
        if self._looks_location(query):
            if symbols or paths:
                return "location"

        # A directly referenced symbol is an exact-symbol query when
        # the question is specifically about that symbol.
        if symbols:
            if self._looks_symbol_explanation(query):
                return "exact_symbol"
            return "exact_symbol"

        # Explicit issue/PR references without a historical marker
        # can still represent an explanation request.
        if issue_numbers or pr_numbers:
            if self._looks_explanation(query):
                return "explanation"
            return "general"

        # Direct explanation language takes precedence over broad
        # conceptual language.
        if self._looks_explanation(query):
            return "explanation"

        if self._looks_conceptual(query):
            return "conceptual"

        if paths:
            return "path"

        return "general"

    def _looks_symbol_explanation(self,query: str) -> bool:
        """Identify direct explanations of explicitly named symbols."""

        return bool(
            re.search(
                r"\b(?:explain|describe)\b",
                query,
                re.IGNORECASE,
            )
        )

    def _looks_conceptual(self,query: str) -> bool:
        """Identify architecture and design-oriented questions."""

        return any(
            pattern.search(query)
            for pattern in self._CONCEPTUAL_PATTERNS
        )

    def _looks_location(self,query: str) -> bool:
        """Identify queries asking where something is implemented."""

        return any(
            pattern.search(query)
            for pattern in self._LOCATION_PATTERNS
        )

    def _looks_dependency(self,query: str) -> bool:
        """Identify dependency and relationship questions."""

        return any(
            pattern.search(query)
            for pattern in self._DEPENDENCY_PATTERNS
        )

    def _looks_debugging(self,query: str) -> bool:
        """Identify debugging-oriented queries."""

        return any(
            pattern.search(query)
            for pattern in self._DEBUGGING_PATTERNS
        )

    def _looks_configuration(self,query: str) -> bool:
        """Identify configuration-related queries."""

        return any(
            pattern.search(query)
            for pattern in self._CONFIGURATION_PATTERNS
        )

    def _looks_testing(self,query: str) -> bool:
        """Identify testing-related queries."""

        return any(
            pattern.search(query)
            for pattern in self._TESTING_PATTERNS
        )

    def _looks_comparison(self,query: str) -> bool:
        """Identify comparison questions."""

        return any(
            pattern.search(query)
            for pattern in self._COMPARISON_PATTERNS
        )

    def _looks_impact(self,query: str) -> bool:
        """Identify impact-analysis questions."""

        return any(
            pattern.search(query)
            for pattern in self._IMPACT_PATTERNS
        )

    def _looks_explanation(self,query: str) -> bool:
        """Identify direct explanation questions."""

        return any(
            pattern.search(query)
            for pattern in self._EXPLANATION_PATTERNS
        )

    def _looks_flow(self,query: str) -> bool:
        """Identify execution or processing flow questions."""

        return any(
            pattern.search(query)
            for pattern in self._FLOW_PATTERNS
        )

    def _looks_historical(self,query: str) -> bool:
        """Identify historical or change-oriented questions."""

        return bool(
            self._HISTORICAL_PATTERN.search(query)
        )

    def _extract_symbols(self,query: str) -> list[str]:
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

    def _extract_paths(self,query: str) -> list[str]:
        """Extract repository-style paths from a query."""

        paths: list[str] = []

        for match in self._PATH_PATTERN.finditer(query):
            path = match.group(0).rstrip(
                ".,;:!?"
            )

            if path not in paths:
                paths.append(path)

        return paths

    def _extract_error_messages(self,query: str) -> list[str]:
        """Extract explicit error or exception references."""

        errors: list[str] = []

        # First preserve explicitly quoted error messages exactly.
        for match in self._QUOTED_ERROR_PATTERN.finditer(query):
            error = match.group(1).strip()

            if error and error not in errors:
                errors.append(error)

        # Then extract standalone exception names or messages.
        for match in self._EXCEPTION_PATTERN.finditer(query):
            error = match.group(0).strip()

            if error and error not in errors:
                errors.append(error)

        return errors

    def _extract_issue_numbers(self,query: str) -> list[str]:
        """Extract explicitly referenced issue numbers."""

        issue_numbers: list[str] = []

        for match in self._ISSUE_PATTERN.finditer(query):
            number = match.group(1)

            if number and number not in issue_numbers:
                issue_numbers.append(number)

        return issue_numbers

    def _extract_pr_numbers(self,query: str) -> list[str]:
        """Extract explicitly referenced pull-request numbers."""

        pr_numbers: list[str] = []

        for match in self._PR_PATTERN.finditer(query):
            number = match.group(1)

            if number and number not in pr_numbers:
                pr_numbers.append(number)

        return pr_numbers