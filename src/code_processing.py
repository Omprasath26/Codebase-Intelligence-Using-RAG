"""Code-aware processing for normalized repository artifacts."""

from __future__ import annotations

import ast
import hashlib
import re
from typing import Any

from pydantic import BaseModel, Field

from src.ingestion_control import RepositoryArtifact
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeChunk(BaseModel):
    """A structurally meaningful unit derived from a repository artifact."""

    stable_id: str
    artifact_id: str
    repository: str
    artifact_type: str
    content: str
    source_path_or_object_id: str
    source_url: str | None = None
    commit_sha: str | None = None
    ref: str
    language: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    module: str | None = None
    symbol: str | None = None
    parent_symbol: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CodeProcessing:
    """Parse normalized artifacts and produce structural knowledge chunks."""

    def process(
        self,
        artifact: RepositoryArtifact,
    ) -> list[KnowledgeChunk]:
        """Process one normalized repository artifact."""

        if artifact.artifact_type in {"code", "test"}:
            return self._process_python(artifact)

        if artifact.artifact_type == "documentation":
            return self._process_documentation(artifact)

        if artifact.artifact_type == "configuration":
            return self._process_configuration(artifact)

        return self._process_other_artifact(artifact)

    def _process_python(
        self,
        artifact: RepositoryArtifact,
    ) -> list[KnowledgeChunk]:
        """Parse Python code or tests and create structural chunks."""

        try:
            tree = ast.parse(artifact.content)
        except SyntaxError as exc:
            logger.warning(
                "Unable to parse Python artifact %s: %s",
                artifact.source_path_or_object_id,
                exc,
            )
            return [
                self._create_chunk(
                    artifact=artifact,
                    content=artifact.content,
                    start_line=1,
                    end_line=self._line_count(artifact.content),
                    symbol=None,
                    parent_symbol=None,
                    module=self._module_name(artifact),
                    metadata={
                        "parser": "python_ast",
                        "structure": "module",
                        "parse_error": str(exc),
                    },
                )
            ]

        imports = self._extract_module_imports(tree)
        module = self._module_name(artifact)

        chunks: list[KnowledgeChunk] = []

        for node in tree.body:
            if isinstance(
                node,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                chunks.extend(
                    self._process_python_node(
                        artifact=artifact,
                        node=node,
                        module=module,
                        imports=imports,
                        parent_symbol=None,
                    )
                )

        if not chunks:
            chunks.append(
                self._create_chunk(
                    artifact=artifact,
                    content=artifact.content,
                    start_line=1,
                    end_line=self._line_count(artifact.content),
                    symbol=None,
                    parent_symbol=None,
                    module=module,
                    metadata={
                        "parser": "python_ast",
                        "structure": "module",
                        "imports": imports,
                    },
                )
            )

        return chunks

    def _process_python_node(
        self,
        artifact: RepositoryArtifact,
        node: ast.AST,
        module: str,
        imports: list[str],
        parent_symbol: str | None,
    ) -> list[KnowledgeChunk]:
        """Create a chunk for a Python class/function and its children."""

        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        symbol = getattr(node, "name", None)

        if start_line is None or end_line is None or symbol is None:
            return []

        content_lines = artifact.content.splitlines()
        content = "\n".join(
            content_lines[start_line - 1:end_line]
        )

        if isinstance(node, ast.ClassDef):
            structure = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            structure = "async_function"
        else:
            structure = "function"

        chunk = self._create_chunk(
            artifact=artifact,
            content=content,
            start_line=start_line,
            end_line=end_line,
            symbol=symbol,
            parent_symbol=parent_symbol,
            module=module,
            metadata={
                "parser": "python_ast",
                "structure": structure,
                "imports": imports,
            },
        )

        chunks = [chunk]

        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    chunks.extend(
                        self._process_python_node(
                            artifact=artifact,
                            node=child,
                            module=module,
                            imports=imports,
                            parent_symbol=symbol,
                        )
                    )

        return chunks

    def _extract_module_imports(
        self,
        tree: ast.Module,
    ) -> list[str]:
        """Extract imports declared at module level."""

        imports: list[str] = []

        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                for alias in node.names:
                    if module:
                        imports.append(
                            f"{module}.{alias.name}"
                        )
                    else:
                        imports.append(alias.name)

        return sorted(set(imports))

    def _process_documentation(
        self,
        artifact: RepositoryArtifact,
    ) -> list[KnowledgeChunk]:
        """Split Markdown or RST documentation into structural sections."""

        lines = artifact.content.splitlines()

        if not lines:
            return [
                self._create_chunk(
                    artifact=artifact,
                    content="",
                    start_line=1,
                    end_line=1,
                    symbol=None,
                    parent_symbol=None,
                    module=None,
                    metadata={
                        "parser": "documentation",
                        "structure": "document",
                    },
                )
            ]

        headings = self._find_documentation_headings(lines)

        if not headings:
            return [
                self._create_chunk(
                    artifact=artifact,
                    content=artifact.content,
                    start_line=1,
                    end_line=len(lines),
                    symbol=None,
                    parent_symbol=None,
                    module=None,
                    metadata={
                        "parser": "documentation",
                        "structure": "document",
                    },
                )
            ]

        chunks: list[KnowledgeChunk] = []

        first_heading_line = headings[0][0]

        if first_heading_line > 1:
            preamble = "\n".join(
                lines[:first_heading_line - 1]
            ).strip()

            if preamble:
                chunks.append(
                    self._create_chunk(
                        artifact=artifact,
                        content=preamble,
                        start_line=1,
                        end_line=first_heading_line - 1,
                        symbol=None,
                        parent_symbol=None,
                        module=None,
                        metadata={
                            "parser": "documentation",
                            "structure": "preamble",
                        },
                    )
                )

        for index, (start_line, heading) in enumerate(headings):
            if index + 1 < len(headings):
                end_line = headings[index + 1][0] - 1
            else:
                end_line = len(lines)

            content = "\n".join(
                lines[start_line - 1:end_line]
            ).strip()

            chunks.append(
                self._create_chunk(
                    artifact=artifact,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=heading,
                    parent_symbol=None,
                    module=None,
                    metadata={
                        "parser": "documentation",
                        "structure": "section",
                    },
                )
            )

        return chunks

    def _find_documentation_headings(
        self,
        lines: list[str],
    ) -> list[tuple[int, str]]:
        """Find Markdown and simple RST-style documentation headings."""

        headings: list[tuple[int, str]] = []

        markdown_pattern = re.compile(
            r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$"
        )

        for index, line in enumerate(lines):
            markdown_match = markdown_pattern.match(line)

            if markdown_match:
                headings.append(
                    (
                        index + 1,
                        markdown_match.group(1).strip(),
                    )
                )
                continue

            if (
                index + 1 < len(lines)
                and line.strip()
                and re.fullmatch(r"\s*[=\-~^]+", lines[index + 1])
                and len(lines[index + 1].strip()) >= len(line.strip())
            ):
                headings.append(
                    (
                        index + 1,
                        line.strip(),
                    )
                )

        return headings

    def _process_configuration(
        self,
        artifact: RepositoryArtifact,
    ) -> list[KnowledgeChunk]:
        """Create a structured chunk for a configuration artifact."""

        return [
            self._create_chunk(
                artifact=artifact,
                content=artifact.content,
                start_line=1,
                end_line=self._line_count(artifact.content),
                symbol=None,
                parent_symbol=None,
                module=None,
                metadata={
                    "parser": "configuration",
                    "structure": "configuration",
                },
            )
        ]

    def _process_other_artifact(
        self,
        artifact: RepositoryArtifact,
    ) -> list[KnowledgeChunk]:
        """Create a knowledge chunk for a non-code normalized artifact."""

        return [
            self._create_chunk(
                artifact=artifact,
                content=artifact.content,
                start_line=1,
                end_line=self._line_count(artifact.content),
                symbol=None,
                parent_symbol=None,
                module=None,
                metadata={
                    "parser": "artifact",
                    "structure": artifact.artifact_type,
                },
            )
        ]

    def _create_chunk(
        self,
        artifact: RepositoryArtifact,
        content: str,
        start_line: int | None,
        end_line: int | None,
        symbol: str | None,
        parent_symbol: str | None,
        module: str | None,
        metadata: dict[str, Any],
    ) -> KnowledgeChunk:
        """Create a chunk while preserving artifact provenance."""

        identity = "|".join(
            [
                artifact.stable_id,
                str(start_line),
                str(end_line),
                symbol or "",
                parent_symbol or "",
            ]
        )

        stable_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        return KnowledgeChunk(
            stable_id=stable_id,
            artifact_id=artifact.stable_id,
            repository=artifact.repository,
            artifact_type=artifact.artifact_type,
            content=content,
            source_path_or_object_id=artifact.source_path_or_object_id,
            source_url=artifact.source_url,
            commit_sha=artifact.commit_sha,
            ref=artifact.ref,
            language=artifact.language,
            start_line=start_line,
            end_line=end_line,
            module=module,
            symbol=symbol,
            parent_symbol=parent_symbol,
            metadata={
                **metadata,
                "artifact_metadata": artifact.metadata,
            },
        )

    def _module_name(
        self,
        artifact: RepositoryArtifact,
    ) -> str:
        """Convert a Python repository path into a module-style name."""

        path = artifact.source_path_or_object_id.replace("\\", "/")

        if path.endswith(".py"):
            path = path[:-3]

        return path.replace("/", ".")

    def _line_count(
        self,
        content: str,
    ) -> int:
        """Return the number of lines in content."""

        return len(content.splitlines()) or 1