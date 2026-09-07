from src.code_processing import CodeProcessing
from src.ingestion_control import RepositoryArtifact


def make_artifact(
    artifact_type: str,
    content: str,
    path: str,
) -> RepositoryArtifact:
    return RepositoryArtifact(
        stable_id="artifact-123",
        repository="scrapy/scrapy",
        artifact_type=artifact_type,
        source_path_or_object_id=path,
        source_url=f"https://github.com/scrapy/scrapy/blob/master/{path}",
        content=content,
        language="Python" if path.endswith(".py") else None,
        commit_sha="abc123",
        ref="master",
        ingestion_timestamp="2026-09-05T10:00:00Z",
        metadata={},
    )


def test_processes_python_function() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "import os\n\n"
            "def hello(name):\n"
            "    return name\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1
    assert chunks[0].symbol == "hello"
    assert chunks[0].metadata["parser"] == "python_ast"
    assert chunks[0].metadata["structure"] == "function"
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 4


def test_processes_python_class_and_method() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "class Spider:\n"
            "    def crawl(self):\n"
            "        return True\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 2

    class_chunk = chunks[0]
    method_chunk = chunks[1]

    assert class_chunk.symbol == "Spider"
    assert class_chunk.parent_symbol is None
    assert class_chunk.metadata["structure"] == "class"

    assert method_chunk.symbol == "crawl"
    assert method_chunk.parent_symbol == "Spider"
    assert method_chunk.metadata["structure"] == "function"


def test_extracts_module_imports() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "import os\n"
            "from pathlib import Path\n\n"
            "def hello():\n"
            "    return Path.cwd()\n"
        ),
    )

    chunks = processor.process(artifact)

    assert chunks[0].metadata["imports"] == [
        "os",
        "pathlib.Path",
    ]


def test_processes_python_test_as_python_structure() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="test",
        path="tests/test_core.py",
        content=(
            "def test_example():\n"
            "    assert True\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1
    assert chunks[0].artifact_type == "test"
    assert chunks[0].symbol == "test_example"
    assert chunks[0].metadata["parser"] == "python_ast"


def test_processes_markdown_sections() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="README.md",
        content=(
            "# Introduction\n"
            "Scrapy is a framework.\n\n"
            "## Installation\n"
            "Install the package.\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 2
    assert chunks[0].symbol == "Introduction"
    assert chunks[1].symbol == "Installation"
    assert chunks[0].metadata["structure"] == "section"


def test_processes_rst_sections() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="docs/index.rst",
        content=(
            "Introduction\n"
            "============\n"
            "Scrapy is a framework.\n\n"
            "Installation\n"
            "------------\n"
            "Install the package.\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 2
    assert chunks[0].symbol == "Introduction"
    assert chunks[1].symbol == "Installation"


def test_processes_configuration() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="configuration",
        path="pyproject.toml",
        content=(
            "[tool.example]\n"
            "name = 'scrapy'\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1
    assert chunks[0].metadata["parser"] == "configuration"
    assert chunks[0].metadata["structure"] == "configuration"


def test_processes_github_artifact() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="issue",
        path="issue:123",
        content="The crawler fails when following redirects.",
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1
    assert chunks[0].artifact_type == "issue"
    assert chunks[0].content == (
        "The crawler fails when following redirects."
    )


def test_preserves_artifact_provenance() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/core.py",
        content=(
            "def hello():\n"
            "    return True\n"
        ),
    )

    chunks = processor.process(artifact)

    chunk = chunks[0]

    assert chunk.artifact_id == artifact.stable_id
    assert chunk.repository == artifact.repository
    assert (
        chunk.source_path_or_object_id
        == artifact.source_path_or_object_id
    )
    assert chunk.source_url == artifact.source_url
    assert chunk.commit_sha == artifact.commit_sha
    assert chunk.ref == artifact.ref


def test_chunk_stable_id_is_deterministic() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/core.py",
        content=(
            "def hello():\n"
            "    return True\n"
        ),
    )

    first = processor.process(artifact)
    second = processor.process(artifact)

    assert first[0].stable_id == second[0].stable_id