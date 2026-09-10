"""Tests for conversation state management."""

from src.context_assembly import (ContextBlock,EvidencePackage)
from src.conversation import (ConversationState,ConversationTurn)
from src.generation import RAGResponse
from src.query_analysis import QueryAnalysis


def make_analysis(query: str = "Where is the Request class implemented?") -> QueryAnalysis:
    """Create a deterministic query analysis for testing."""
    return QueryAnalysis(
        original_query=query,
        normalized_query=query,
        intent="location",
        symbols=["Request"],
        paths=[],
        error_messages=[],
        issue_numbers=[],
        pr_numbers=[],
    )


def make_response(answer: str = "The Request class is implemented in the request module.",) -> RAGResponse:
    """Create a deterministic RAG response for testing."""
    return RAGResponse(
        answer=answer,
        sources=[],
        confidence=0.9,
        limitation=None,
        repository_revision="abc123",
        evidence_metadata={},
        validated=True,
    )


def make_evidence(repository: str = "scrapy/scrapy",commit_sha: str = "abc123") -> EvidencePackage:
    """Create a minimal evidence package."""
    from src.code_processing import KnowledgeChunk

    chunk = KnowledgeChunk(
        stable_id="chunk-request",
        artifact_id="artifact-request",
        repository=repository,
        artifact_type="code",
        content="class Request:\n    pass",
        source_path_or_object_id="scrapy/http/request/__init__.py",
        source_url=(
            "https://github.com/scrapy/scrapy/"
            "blob/abc123/scrapy/http/request/__init__.py"
        ),
        commit_sha=commit_sha,
        ref="master",
        language="python",
        start_line=1,
        end_line=2,
        module="scrapy.http.request",
        symbol="Request",
        parent_symbol=None,
        metadata={
            "symbol_type": "class",
        },
    )

    block = ContextBlock(
        chunk=chunk,
        score=1.0,
        retrieval_method="dense",
        dense_score=1.0,
        bm25_score=None,
        rerank_score=None,
        expansion_type="retrieved",
    )

    return EvidencePackage(
        query="Where is the Request class implemented?",
        context_blocks=[block],
        sufficient=True,
        limitation=None,
        repository_revision=commit_sha,
        total_characters=len(chunk.content),
    )


def test_new_conversation_has_unique_id():
    first = ConversationState()
    second = ConversationState()

    assert first.conversation_id
    assert second.conversation_id
    assert first.conversation_id != second.conversation_id


def test_new_conversation_has_empty_state():
    state = ConversationState()

    assert state.repository is None
    assert state.ref is None
    assert state.turns == ()
    assert state.latest_evidence is None
    assert state.last_turn is None
    assert state.previous_analysis is None
    assert state.previous_response is None


def test_repository_scope_can_be_set_explicitly():
    state = ConversationState()

    scoped = state.with_repository_scope(
        repository="scrapy/scrapy",
        ref="master",
    )

    assert scoped.repository == "scrapy/scrapy"
    assert scoped.ref == "master"
    assert state.repository is None


def test_add_turn_preserves_repository_scope():
    state = ConversationState().with_repository_scope(
        repository="scrapy/scrapy",
        ref="master",
    )

    analysis = make_analysis()

    updated = state.add_turn(
        query="Where is the Request class implemented?",
        analysis=analysis,
    )

    assert updated.repository == "scrapy/scrapy"
    assert updated.ref == "master"
    assert len(updated.turns) == 1


def test_add_turn_preserves_previous_turns():
    state = ConversationState()

    first = state.add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(
            "Where is Request implemented?"
        ),
    )

    second = first.add_turn(
        query="What does its constructor do?",
        analysis=make_analysis(
            "What does its constructor do?"
        ),
    )

    assert len(second.turns) == 2
    assert second.turns[0].query == (
        "Where is Request implemented?"
    )
    assert second.turns[1].query == (
        "What does its constructor do?"
    )


def test_last_turn_returns_latest_turn():
    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
    )

    assert state.last_turn is not None
    assert state.last_turn.query == (
        "Where is Request implemented?"
    )


def test_previous_analysis_returns_latest_analysis():
    analysis = make_analysis()

    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=analysis,
    )

    assert state.previous_analysis == analysis


def test_previous_response_returns_latest_response():
    response = make_response()

    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
        response=response,
    )

    assert state.previous_response == response


def test_response_can_be_added_after_turn_creation():
    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
    )

    response = make_response()

    updated = state.update_response(response)

    assert updated.previous_response == response
    assert state.previous_response is None


def test_update_response_requires_existing_turn():
    state = ConversationState()

    try:
        state.update_response(make_response())
    except ValueError as exc:
        assert str(exc) == (
            "Cannot update a response without a conversation turn."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty conversation."
        )


def test_follow_up_context_preserves_previous_analysis():
    analysis = make_analysis()

    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=analysis,
    )

    context = state.follow_up_context()

    assert context["previous_analysis"] == analysis
    assert context["repository"] is None
    assert context["ref"] is None


def test_follow_up_context_preserves_previous_response():
    response = make_response()

    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
        response=response,
    )

    context = state.follow_up_context()

    assert context["previous_response"] == response


def test_update_evidence_sets_repository_from_evidence():
    state = ConversationState()

    evidence = make_evidence()

    updated = state.update_evidence(evidence)

    assert updated.repository == "scrapy/scrapy"
    assert updated.latest_evidence == evidence


def test_new_evidence_replaces_old_evidence():
    first = make_evidence(
        repository="scrapy/scrapy",
        commit_sha="old123",
    )

    second = make_evidence(
        repository="scrapy/scrapy",
        commit_sha="new456",
    )

    state = ConversationState().update_evidence(first)
    updated = state.update_evidence(second)

    assert updated.latest_evidence == second
    assert updated.latest_evidence != first

    assert (
        updated.latest_evidence.context_blocks[0]
        .chunk.repository
        == "scrapy/scrapy"
    )

    assert (
        updated.latest_evidence.repository_revision
        == "new456"
    )


def test_latest_evidence_is_available_to_follow_up_context():
    evidence = make_evidence()

    state = ConversationState().update_evidence(
        evidence
    )

    context = state.follow_up_context()

    assert context["latest_evidence"] == evidence


def test_reset_creates_new_conversation():
    state = (
        ConversationState()
        .with_repository_scope(
            repository="scrapy/scrapy",
            ref="master",
        )
        .add_turn(
            query="Where is Request implemented?",
            analysis=make_analysis(),
            response=make_response(),
        )
        .update_evidence(make_evidence())
    )

    reset_state = state.reset()

    assert reset_state.conversation_id != (
        state.conversation_id
    )
    assert reset_state.repository is None
    assert reset_state.ref is None
    assert reset_state.turns == ()
    assert reset_state.latest_evidence is None


def test_reset_does_not_modify_original_state():
    state = ConversationState().add_turn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
    )

    reset_state = state.reset()

    assert len(state.turns) == 1
    assert len(reset_state.turns) == 0


def test_conversation_turn_is_immutable():
    turn = ConversationTurn(
        query="Where is Request implemented?",
        analysis=make_analysis(),
    )

    try:
        turn.query = "changed"
    except Exception:
        pass
    else:
        raise AssertionError(
            "ConversationTurn should be immutable."
        )


def test_conversation_state_is_immutable():
    state = ConversationState()

    try:
        state.repository = "scrapy/scrapy"
    except Exception:
        pass
    else:
        raise AssertionError(
            "ConversationState should be immutable."
        )


def test_invalid_query_is_rejected():
    state = ConversationState()

    try:
        state.add_turn(
            query="   ",
            analysis=make_analysis(),
        )
    except ValueError as exc:
        assert str(exc) == (
            "query must be a non-empty string."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty query."
        )


def test_invalid_analysis_is_rejected():
    state = ConversationState()

    try:
        state.add_turn(
            query="Where is Request implemented?",
            analysis="invalid",  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "analysis must be a QueryAnalysis."
        )
    else:
        raise AssertionError(
            "Expected TypeError for invalid analysis."
        )


def test_invalid_evidence_is_rejected():
    state = ConversationState()

    try:
        state.update_evidence(
            "invalid"  # type: ignore[arg-type]
        )
    except TypeError as exc:
        assert str(exc) == (
            "evidence_package must be an EvidencePackage."
        )
    else:
        raise AssertionError(
            "Expected TypeError for invalid evidence."
        )