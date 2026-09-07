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


def test_detects_exact_symbol_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Where is parse_response() implemented?"
    )

    assert result.intent == "exact_symbol"
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

    assert result.intent == "path"
    assert result.paths == ["scrapy/http/request.py"]
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
        "How does the request processing flow work?"
    )

    assert result.intent == "conceptual"


def test_detects_general_query() -> None:
    analyzer = QueryAnalyzer()

    result = analyzer.analyze(
        "Tell me about Scrapy."
    )

    assert result.intent == "general"


def test_rejects_empty_query() -> None:
    analyzer = QueryAnalyzer()

    try:
        analyzer.analyze("   ")
    except ValueError as exc:
        assert str(exc) == "Query cannot be empty."
    else:
        raise AssertionError("Expected ValueError.")