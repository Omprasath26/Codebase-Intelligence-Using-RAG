from src.query_analysis import QueryAnalyzer


def test_normalizes_query_whitespace() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "  how   does   the crawler   work?  "
    )

    assert result.original_query == (
        "  how   does   the crawler   work?  "
    )
    assert result.normalized_query == (
        "how does the crawler work?"
    )


def test_normalization_preserves_exact_identifier_case() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Where is HtmlResponse implemented?"
    )

    assert result.normalized_query == (
        "Where is HtmlResponse implemented?"
    )
    assert result.symbols == []


def test_detects_exact_symbol_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Where is parse_response() implemented?"
    )

    assert result.intent == "location"
    assert result.symbols == ["parse_response"]
    assert result.paths == []


def test_extracts_class_symbol() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Explain the Request class."
    )

    assert result.intent == "exact_symbol"
    assert result.symbols == ["Request"]
    assert result.paths == []


def test_extracts_function_symbol() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Explain the function parse_response."
    )

    assert result.intent == "exact_symbol"
    assert result.symbols == ["parse_response"]
    assert result.paths == []


def test_detects_path_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "What is implemented in scrapy/http/request.py?"
    )

    assert result.intent == "location"
    assert result.paths == [
        "scrapy/http/request.py"
    ]
    assert result.symbols == []


def test_detects_historical_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "When was this behavior introduced?"
    )

    assert result.intent == "historical"


def test_detects_conceptual_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "How does the request processing work?"
    )

    assert result.intent == "conceptual"


def test_detects_flow_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "How does the request processing flow work?"
    )

    assert result.intent == "flow"


def test_detects_general_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Tell me about Scrapy."
    )

    assert result.intent == "general"


def test_detects_explanation_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Explain how Request objects are created."
    )

    assert result.intent == "explanation"


def test_detects_dependency_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "What modules does Request depend on?"
    )

    assert result.intent == "dependency"


def test_detects_debugging_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        'Why am I getting "KeyError: response"?'
    )

    assert result.intent == "debugging"


def test_detects_configuration_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "How do I configure the crawler settings?"
    )

    assert result.intent == "configuration"


def test_detects_testing_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "What tests cover the request processing code?"
    )

    assert result.intent == "testing"


def test_detects_comparison_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Compare the Request and Response classes."
    )

    assert result.intent == "comparison"


def test_detects_impact_analysis_intent() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "What is the impact of changing Request?"
    )

    assert result.intent == "impact_analysis"


def test_extracts_error_message() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        'Why does "KeyError: response" happen in the crawler?'
    )

    assert result.error_messages == [
        "KeyError: response"
    ]
    assert result.intent == "debugging"


def test_extracts_exception_reference() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "How can I fix ValueError in request processing?"
    )

    assert result.error_messages == [
        "ValueError"
    ]
    assert result.intent == "debugging"


def test_extracts_issue_number() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "What changed in issue #1234?"
    )

    assert result.issue_numbers == ["1234"]
    assert result.pr_numbers == []
    assert result.intent == "historical"


def test_extracts_pull_request_number() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Why was PR #5678 merged?"
    )

    assert result.issue_numbers == []
    assert result.pr_numbers == ["5678"]
    assert result.intent == "explanation"


def test_extracts_issue_and_pr_numbers() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Compare issue #1234 with PR #5678."
    )

    assert result.issue_numbers == ["1234"]
    assert result.pr_numbers == ["5678"]
    assert result.intent == "comparison"


def test_preserves_multiple_entities() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Where is parse_response() in "
        "scrapy/http/request.py?"
    )

    assert result.symbols == ["parse_response"]
    assert result.paths == [
        "scrapy/http/request.py"
    ]
    assert result.intent == "location"


def test_rejects_empty_query() -> None:
    analyzer = QueryAnalyzer()

    try:
        analyzer.analyze("   ")
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
    else:
        raise AssertionError("Expected ValueError.")


def test_rejects_non_string_query() -> None:
    analyzer = QueryAnalyzer()

    try:
        analyzer.analyze(None)  # type: ignore[arg-type]
    except TypeError as exc:
        assert str(exc) == "Query must be a string."
    else:
        raise AssertionError("Expected TypeError.")