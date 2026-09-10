"""First-release Streamlit chat UI for the codebase intelligence system."""

from __future__ import annotations
from typing import Any
import httpx
import streamlit as st
from src.generation import RAGResponse, SourceReference


DEFAULT_API_URL = "http://127.0.0.1:8000"


def _build_query_payload(query: str,repository: str | None,ref: str | None,conversation_id: str | None) -> dict[str, Any]:
    """
    Build the structured payload sent to the existing FastAPI query API.

    Validation is performed here because this function owns construction
    of the query payload. This prevents invalid developer questions from
    reaching the API boundary.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "query must be a non-empty string."
        )

    payload: dict[str, Any] = {
        "query": query.strip(),
    }

    if repository:
        payload["repository"] = repository.strip()

    if ref:
        payload["ref"] = ref.strip()

    if conversation_id:
        payload["conversation_id"] = (
            conversation_id
        )

    return payload


def submit_query(
    api_url: str,
    query: str,
    repository: str | None = None,
    ref: str | None = None,
    conversation_id: str | None = None,
    timeout_seconds: float = 60.0) -> dict[str, Any]:
    """
    Submit one developer question to the existing FastAPI query endpoint.

    The UI does not perform retrieval, generation, validation, or
    conversation orchestration itself. Those responsibilities remain
    behind the existing API boundary.
    """

    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError(
            "api_url must be a non-empty string."
        )

    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be greater than zero."
        )

    payload = _build_query_payload(
        query=query,
        repository=repository,
        ref=ref,
        conversation_id=conversation_id,
    )

    url = (
        api_url.rstrip("/")
        + "/query"
    )

    try:
        response = httpx.post(
            url,
            json=payload,
            timeout=timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to connect to the query API."
        ) from exc

    if response.status_code >= 400:
        try:
            detail = response.json().get(
                "detail",
                "Query API request failed.",
            )
        except ValueError:
            detail = "Query API request failed."

        raise RuntimeError(
            str(detail)
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Query API returned invalid JSON."
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Query API returned an invalid response."
        )

    return data


def _parse_rag_response(data: dict[str, Any]) -> RAGResponse:
    """Convert the structured API result into the existing response model."""

    result = data.get("result")

    if not isinstance(result, dict):
        raise RuntimeError(
            "Query API response does not contain a valid result."
        )

    try:
        return RAGResponse.model_validate(
            result
        )
    except Exception as exc:
        raise RuntimeError(
            "Query API returned an invalid RAG response."
        ) from exc


def _format_source(source: SourceReference) -> str:
    """Create a concise human-readable source reference."""

    location = source.source_path_or_object_id

    if (
        source.start_line is not None
        and source.end_line is not None):
        location = (
            f"{location}:"
            f"{source.start_line}-"
            f"{source.end_line}"
        )
    elif source.start_line is not None:
        location = (
            f"{location}:"
            f"{source.start_line}"
        )

    if source.symbol:
        location = (
            f"{location} — "
            f"{source.symbol}"
        )

    return location


def render_sources(sources: list[SourceReference]) -> None:
    """Render repository provenance and source metadata."""

    if not sources:
        st.info(
            "No source references were returned.")
        return

    for index, source in enumerate(sources,start=1):
        with st.expander(
            f"Source {index}: "
            f"{_format_source(source)}"):
            st.write(
                f"**Artifact type:** "
                f"{source.artifact_type}"
            )

            st.write(
                f"**Path / object ID:** "
                f"{source.source_path_or_object_id}"
            )

            if source.symbol:
                st.write(
                    f"**Symbol:** {source.symbol}"
                )

            if source.parent_symbol:
                st.write(
                    f"**Parent symbol:** "
                    f"{source.parent_symbol}"
                )

            if source.start_line is not None:
                if source.end_line is not None:
                    st.write(
                        f"**Lines:** "
                        f"{source.start_line}-"
                        f"{source.end_line}"
                    )
                else:
                    st.write(
                        f"**Line:** "
                        f"{source.start_line}"
                    )

            if source.language:
                st.write(
                    f"**Language:** "
                    f"{source.language}"
                )

            if source.repository:
                st.write(
                    f"**Repository:** "
                    f"{source.repository}"
                )

            if source.commit_sha:
                st.write(
                    f"**Commit:** "
                    f"{source.commit_sha}"
                )

            if source.ref:
                st.write(
                    f"**Ref:** {source.ref}"
                )

            if source.source_url:
                st.markdown(
                    f"[Open source artifact]"
                    f"({source.source_url})"
                )

            st.caption(
                f"Chunk ID: {source.stable_id}"
            )

            st.caption(
                f"Artifact ID: {source.artifact_id}"
            )


def render_response(response: RAGResponse,correlation_id: str | None = None) -> None:
    """Render one validated RAG response and its provenance."""

    st.markdown("### Answer")
    st.markdown(response.answer)

    if response.confidence is not None:
        st.metric(
            "Confidence",
            f"{response.confidence:.2f}",
        )

    if response.limitation:
        st.warning(
            f"**Limitation:** "
            f"{response.limitation}"
        )

    if response.repository_revision:
        st.caption(
            "Repository revision: "
            f"{response.repository_revision}"
        )

    st.markdown("### Sources")

    render_sources(
        response.sources
    )

    if response.evidence_metadata:
        with st.expander("Retrieval / evidence metadata"):
            st.json(
                response.evidence_metadata
            )

    if correlation_id:
        st.caption(
            f"Correlation ID: {correlation_id}"
        )


def _initialize_session_state() -> None:
    """Initialize only UI-owned Streamlit session state."""

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None

    if "messages" not in st.session_state:
        st.session_state.messages = []


def _reset_conversation() -> None:
    """Reset the UI conversation state."""

    st.session_state.conversation_id = None
    st.session_state.messages = []


def _render_previous_messages() -> None:
    """Render responses already returned during this UI session."""

    for message in st.session_state.messages:
        role = message["role"]

        with st.chat_message(role):
            if role == "user":
                st.markdown(
                    message["content"]
                )
            else:
                response = message["response"]

                render_response(
                    response,
                    message.get(
                        "correlation_id"
                    ),
                )


def main() -> None:
    """Run the first-release Streamlit chat application."""

    st.set_page_config(
        page_title="Codebase Intelligence",
        page_icon="💬",
        layout="wide",
    )

    _initialize_session_state()

    st.title(
        "Codebase Intelligence System"
    )

    st.caption(
        "Developer Question Answering "
        "with Grounded Repository Evidence"
    )

    with st.sidebar:
        st.header("Query Configuration")

        api_url = st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
        )

        repository = st.text_input(
            "Repository",
            value="",
            placeholder="scrapy/scrapy",
        )

        ref = st.text_input(
            "Ref",
            value="",
            placeholder="master",
        )

        if st.button(
            "Reset conversation",
            use_container_width=True):
            _reset_conversation()
            st.rerun()

        if st.session_state.conversation_id:
            st.caption(
                "Conversation ID: "
                f"{st.session_state.conversation_id}"
            )

    _render_previous_messages()

    query = st.chat_input("Ask a question about the repository...")

    if not query:
        return

    with st.chat_message("user"):
        st.markdown(query)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    try:
        data = submit_query(
            api_url=api_url,
            query=query,
            repository=repository or None,
            ref=ref or None,
            conversation_id=(
                st.session_state.conversation_id
            ),
        )

        response = _parse_rag_response(
            data
        )

        conversation_id = data.get(
            "conversation_id"
        )

        if conversation_id:
            st.session_state.conversation_id = (
                conversation_id
            )

        correlation_id = data.get(
            "correlation_id"
        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "response": response,
                "correlation_id": correlation_id,
            }
        )

        with st.chat_message("assistant"):
            render_response(
                response,
                correlation_id,
            )

    except Exception as exc:
        st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()