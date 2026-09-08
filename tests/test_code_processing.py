from src.code_processing import CodeProcessing
from src.ingestion_control import RepositoryArtifact


def make_artifact(artifact_type: str,content: str,path: str,language: str | None = None,metadata: dict | None = None) -> RepositoryArtifact:
    return RepositoryArtifact(
        stable_id="artifact-123",
        repository="scrapy/scrapy",
        artifact_type=artifact_type,
        source_path_or_object_id=path,
        source_url=(
            f"https://github.com/scrapy/scrapy/blob/master/{path}"
        ),
        content=content,
        language=(
            language
            if language is not None
            else "Python" if path.endswith(".py") else None
        ),
        commit_sha="abc123",
        ref="master",
        ingestion_timestamp="2026-09-05T10:00:00Z",
        metadata=metadata or {},
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

    chunk = chunks[0]

    assert chunk.symbol == "hello"
    assert chunk.parent_symbol is None
    assert chunk.module == "scrapy.example"
    assert chunk.start_line == 3
    assert chunk.end_line == 4

    assert chunk.metadata["parser"] == "python_ast"
    assert chunk.metadata["structure"] == "function"
    assert chunk.metadata["symbol_type"] == "function"
    assert chunk.metadata["package"] == "scrapy"

    assert chunk.metadata["imports"] == ["os"]


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
    assert class_chunk.metadata["symbol_type"] == "class"

    assert method_chunk.symbol == "crawl"
    assert method_chunk.parent_symbol == "Spider"
    assert method_chunk.metadata["structure"] == "function"
    assert method_chunk.metadata["symbol_type"] == "function"

    assert (
        method_chunk.metadata["relationships"]["parent_symbol"]
        == "Spider"
    )

    assert (
        method_chunk.metadata["relationships"]["relationship_type"]
        == "class_method"
    )


def test_processes_async_function() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "async def fetch():\n"
            "    return True\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.symbol == "fetch"
    assert chunk.metadata["structure"] == "async_function"
    assert chunk.metadata["symbol_type"] == "function"


def test_extracts_module_imports() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "import os\n"
            "import scrapy\n"
            "from pathlib import Path\n\n"
            "def hello():\n"
            "    return Path.cwd()\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    assert chunks[0].metadata["imports"] == [
        "os",
        "pathlib.Path",
        "scrapy",
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

    chunk = chunks[0]

    assert chunk.artifact_type == "test"
    assert chunk.symbol == "test_example"
    assert chunk.metadata["parser"] == "python_ast"
    assert chunk.metadata["symbol_type"] == "function"


def test_preserves_python_decorator() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/example.py",
        content=(
            "@decorator\n"
            "def hello():\n"
            "    return True\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.symbol == "hello"
    assert chunk.start_line == 1
    assert chunk.end_line == 3

    assert chunk.content == (
        "@decorator\n"
        "def hello():\n"
        "    return True"
    )


def test_processes_markdown_sections_and_hierarchy() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="README.md",
        language="markdown",
        content=(
            "# Introduction\n"
            "Scrapy is a framework.\n\n"
            "## Installation\n"
            "Install the package.\n\n"
            "### Windows\n"
            "Install on Windows.\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 3

    introduction = chunks[0]
    installation = chunks[1]
    windows = chunks[2]

    assert introduction.symbol == "Introduction"
    assert introduction.metadata["heading_level"] == 1
    assert introduction.metadata["document_title"] == "Introduction"
    assert introduction.metadata["heading_path"] == [
        "Introduction"
    ]
    assert introduction.parent_symbol is None

    assert installation.symbol == "Installation"
    assert installation.metadata["heading_level"] == 2
    assert installation.metadata["document_title"] == "Introduction"
    assert installation.metadata["heading_path"] == [
        "Introduction",
        "Installation",
    ]
    assert installation.parent_symbol == "Introduction"

    assert windows.symbol == "Windows"
    assert windows.metadata["heading_level"] == 3
    assert windows.metadata["heading_path"] == [
        "Introduction",
        "Installation",
        "Windows",
    ]
    assert windows.parent_symbol == "Installation"


def test_processes_rst_sections_and_hierarchy() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="docs/index.rst",
        language="rst",
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

    introduction = chunks[0]
    installation = chunks[1]

    assert introduction.symbol == "Introduction"
    assert introduction.metadata["heading_level"] == 1
    assert introduction.metadata["document_title"] == "Introduction"
    assert introduction.metadata["heading_path"] == [
        "Introduction"
    ]

    assert installation.symbol == "Installation"
    assert installation.metadata["heading_level"] == 2
    assert installation.metadata["document_title"] == "Introduction"
    assert installation.metadata["heading_path"] == [
        "Introduction",
        "Installation",
    ]
    assert installation.parent_symbol == "Introduction"


def test_processes_markdown_preamble() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="README.md",
        content=(
            "Project documentation.\n"
            "\n"
            "# Introduction\n"
            "\n"
            "Welcome to the project.\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 2

    preamble = chunks[0]
    section = chunks[1]

    assert preamble.symbol is None
    assert preamble.start_line == 1
    assert preamble.end_line == 2
    assert preamble.metadata["structure"] == "preamble"

    assert section.symbol == "Introduction"
    assert section.start_line == 3
    assert section.end_line == 5


def test_processes_documentation_without_headings() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="documentation",
        path="docs/example.md",
        language="markdown",
        content=(
            "This is a document without headings.\n"
            "It contains multiple lines.\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.symbol is None
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert chunk.metadata["structure"] == "document"
    assert chunk.metadata["heading_path"] == []


def test_processes_configuration() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="configuration",
        path="pyproject.toml",
        language="toml",
        content=(
            "[tool.example]\n"
            "name = 'scrapy'\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.artifact_type == "configuration"
    assert chunk.metadata["parser"] == "configuration"
    assert chunk.metadata["structure"] == "configuration"
    assert chunk.metadata["symbol_type"] == "configuration"


def test_processes_github_artifact_and_preserves_relationships() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="issue",
        path="issue:123",
        content="The crawler fails when following redirects.",
        language=None,
        metadata={
            "issue_number": 123,
            "related_pull_request": 456,
        },
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.artifact_type == "issue"
    assert chunk.content == (
        "The crawler fails when following redirects."
    )

    assert chunk.metadata["relationships"] == {
        "issue_number": 123,
        "related_pull_request": 456,
    }


def test_does_not_invent_artifact_relationships() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="commit",
        path="commit:abc123",
        content="Updated crawler behavior.",
        language=None,
        metadata={},
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.metadata["relationships"] == {}


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

    assert len(chunks) == 1

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


def test_enriches_revision_metadata() -> None:
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

    assert len(chunks) == 1

    revision = chunks[0].metadata["revision"]

    assert revision == {
        "commit_sha": "abc123",
        "ref": "master",
    }


def test_enriches_source_location_metadata() -> None:
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

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.metadata["source"] == {
        "repository": "scrapy/scrapy",
        "path_or_object_id": "scrapy/core.py",
        "url": (
            "https://github.com/scrapy/scrapy/blob/master/"
            "scrapy/core.py"
        ),
    }

    assert chunk.metadata["location"] == {
        "start_line": 1,
        "end_line": 2,
    }


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


def test_syntax_error_does_not_crash_processing() -> None:
    processor = CodeProcessing()

    artifact = make_artifact(
        artifact_type="code",
        path="scrapy/broken.py",
        content=(
            "def broken(:\n"
            "    return True\n"
        ),
    )

    chunks = processor.process(artifact)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.symbol is None
    assert chunk.start_line == 1
    assert chunk.end_line == 2
    assert chunk.metadata["parser"] == "python_ast"
    assert chunk.metadata["structure"] == "module"
    assert "parse_error" in chunk.metadata