"""Code-aware processing, metadata enrichment, and relationships."""

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
    """Parse artifacts and produce enriched structural knowledge chunks."""

    def process(self,artifact: RepositoryArtifact) -> list[KnowledgeChunk]:
        """Route an artifact to the appropriate processing strategy."""

        if artifact.artifact_type in {"code", "test"}:
            return self._process_python(artifact)

        if artifact.artifact_type == "documentation":
            return self._process_documentation(artifact)

        if artifact.artifact_type == "configuration":
            return self._process_configuration(artifact)

        return self._process_other_artifact(artifact)

    
    # Python processing
    

    def _process_python(self,artifact: RepositoryArtifact) -> list[KnowledgeChunk]:
        """Parse Python source and create structural chunks."""

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
        package = self._package_name(module)

        chunks: list[KnowledgeChunk] = []

        for node in tree.body:
            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                chunks.extend(
                    self._process_python_node(
                        artifact=artifact,
                        node=node,
                        module=module,
                        package=package,
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
                        "symbol_type": "module",
                        "imports": imports,
                        "package": package,
                    },
                )
            )

        return chunks

    def _process_python_node(self,artifact: RepositoryArtifact,node: ast.AST,module: str,package: str,imports: list[str],parent_symbol: str | None) -> list[KnowledgeChunk]:
        """Create an enriched chunk for a Python class or function."""

        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)
        symbol = getattr(node, "name", None)

        if start_line is None or end_line is None or symbol is None:
            return []

        decorators = getattr(node, "decorator_list", [])

        if decorators:
            decorator_lines = [
                getattr(decorator, "lineno", start_line)
                for decorator in decorators
            ]

            start_line = min(
                [start_line, *decorator_lines]
            )

        content_lines = artifact.content.splitlines()

        content = "\n".join(
            content_lines[start_line - 1 : end_line]
        )

        if isinstance(node, ast.ClassDef):
            structure = "class"
            symbol_type = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            structure = "async_function"
            symbol_type = "function"
        else:
            structure = "function"
            symbol_type = "function"

        relationship_type = (
            "class_method"
            if parent_symbol is not None
            else "module_symbol"
        )

        metadata = {
            "parser": "python_ast",
            "structure": structure,
            "symbol_type": symbol_type,
            "imports": imports,
            "package": package,
            "relationships": {
                "parent_symbol": parent_symbol,
                "relationship_type": relationship_type,
            },
        }

        chunk = self._create_chunk(
            artifact=artifact,
            content=content,
            start_line=start_line,
            end_line=end_line,
            symbol=symbol,
            parent_symbol=parent_symbol,
            module=module,
            metadata=metadata,
        )

        chunks = [chunk]

        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(
                    child,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    chunks.extend(
                        self._process_python_node(
                            artifact=artifact,
                            node=child,
                            module=module,
                            package=package,
                            imports=imports,
                            parent_symbol=symbol,
                        )
                    )

        return chunks

    def _extract_module_imports(self,tree: ast.Module) -> list[str]:
        """Extract module-level Python imports."""

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

    
    # Documentation processing
    

    def _process_documentation(self,artifact: RepositoryArtifact) -> list[KnowledgeChunk]:
        """Split documentation into heading-based structural sections."""

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
                        "document_title": None,
                        "heading_level": None,
                        "heading_path": [],
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
                        "document_title": None,
                        "heading_level": None,
                        "heading_path": [],
                    },
                )
            ]

        chunks: list[KnowledgeChunk] = []

        first_heading_line = headings[0][0]

        if first_heading_line > 1:
            preamble = "\n".join(
                lines[: first_heading_line - 1]).strip()

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
                            "document_title": headings[0][1],
                            "heading_level": None,
                            "heading_path": [],
                        },
                    )
                )

        hierarchy: list[tuple[int, str]] = []

        document_title = headings[0][1]

        for index, heading in enumerate(headings):
            start_line, heading_text, heading_level = heading

            if index + 1 < len(headings):
                end_line = headings[index + 1][0] - 1
            else:
                end_line = len(lines)

            content = "\n".join(
                lines[start_line - 1 : end_line]
            ).strip()

            hierarchy = self._update_heading_hierarchy(
                hierarchy=hierarchy,
                heading_level=heading_level,
                heading_text=heading_text,
            )

            heading_path = [
                item[1]
                for item in hierarchy
            ]

            parent_section = (
                hierarchy[-2][1]
                if len(hierarchy) > 1
                else None
            )

            metadata = {
                "parser": "documentation",
                "structure": "section",
                "document_title": document_title,
                "heading_level": heading_level,
                "heading_path": heading_path,
                "relationships": {
                    "parent_section": parent_section,
                    "relationship_type": (
                        "document_section"
                        if parent_section is None
                        else "nested_document_section"
                    ),
                },
            }

            chunks.append(
                self._create_chunk(
                    artifact=artifact,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    symbol=heading_text,
                    parent_symbol=parent_section,
                    module=None,
                    metadata=metadata,
                )
            )

        return chunks

    def _find_documentation_headings(self,lines: list[str]) -> list[tuple[int, str, int]]:
        """Find Markdown and simple RST-style headings."""

        headings: list[tuple[int, str, int]] = []

        markdown_pattern = re.compile(
            r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$"
        )

        rst_level_map = {
            "=": 1,
            "-": 2,
            "~": 3,
            "^": 4,
        }

        for index, line in enumerate(lines):
            markdown_match = markdown_pattern.match(line)

            if markdown_match:
                marker = markdown_match.group(1)
                heading_text = markdown_match.group(2).strip()

                headings.append(
                    (
                        index + 1,
                        heading_text,
                        len(marker),
                    )
                )

                continue

            if (
                index + 1 < len(lines)
                and line.strip()
                and re.fullmatch(
                    r"\s*([=\-~^]+)",
                    lines[index + 1],
                )
            ):
                underline = lines[index + 1].strip()

                if len(underline) >= len(line.strip()):
                    marker = underline[0]

                    if marker in rst_level_map:
                        headings.append(
                            (
                                index + 1,
                                line.strip(),
                                rst_level_map[marker],
                            )
                        )

        return headings

    def _update_heading_hierarchy(self,hierarchy: list[tuple[int, str]],heading_level: int,heading_text: str) -> list[tuple[int, str]]:
        """Update the current documentation heading hierarchy."""

        while hierarchy and hierarchy[-1][0] >= heading_level:
            hierarchy.pop()

        hierarchy.append(
            (
                heading_level,
                heading_text,
            )
        )

        return hierarchy

    
    # Configuration and generic artifacts
    

    def _process_configuration(self,artifact: RepositoryArtifact) -> list[KnowledgeChunk]:
        """Create a chunk for a configuration artifact."""

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
                    "symbol_type": "configuration",
                    "relationships": {},
                },
            )
        ]

    def _process_other_artifact(self,artifact: RepositoryArtifact) -> list[KnowledgeChunk]:
        """Create a chunk for non-code engineering artifacts."""

        relationships = self._extract_artifact_relationships(
            artifact.metadata
        )

        metadata = {
            "parser": "artifact",
            "structure": artifact.artifact_type,
            "symbol_type": artifact.artifact_type,
            "relationships": relationships,
        }

        return [
            self._create_chunk(
                artifact=artifact,
                content=artifact.content,
                start_line=1,
                end_line=self._line_count(artifact.content),
                symbol=None,
                parent_symbol=None,
                module=None,
                metadata=metadata,
            )
        ]

    
    # Metadata enrichment
    

    def _create_chunk(self,artifact: RepositoryArtifact,content: str,start_line: int | None,end_line: int | None,symbol: str | None,parent_symbol: str | None,module: str | None,metadata: dict[str, Any]) -> KnowledgeChunk:
        """Create a deterministic chunk with enriched provenance."""

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

        enriched_metadata = {
            **metadata,
            "artifact_metadata": artifact.metadata,
            "revision": {
                "commit_sha": artifact.commit_sha,
                "ref": artifact.ref,
            },
            "source": {
                "repository": artifact.repository,
                "path_or_object_id": (
                    artifact.source_path_or_object_id
                ),
                "url": artifact.source_url,
            },
            "location": {
                "start_line": start_line,
                "end_line": end_line,
            },
        }

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
            metadata=enriched_metadata,
        )

    def _extract_artifact_relationships(self,artifact_metadata: dict[str, Any]) -> dict[str, Any]:
        """Preserve relationship information supplied by ingestion.

        Relationships are copied from source metadata only. No repository
        relationship is inferred when the source artifact does not provide it.
        """

        relationship_keys = {
            "issue_number",
            "pull_request_number",
            "pull_request",
            "issue",
            "commit",
            "commit_sha",
            "review",
            "review_id",
            "comment",
            "comment_id",
            "related_issue",
            "related_pull_request",
            "related_commit",
        }

        relationships = {
            key: artifact_metadata[key]
            for key in sorted(artifact_metadata)
            if key in relationship_keys
        }

        return relationships

    
    # Module/package helpers
    

    def _module_name(self,artifact: RepositoryArtifact) -> str:
        """Convert a Python repository path to a module-style name."""

        path = artifact.source_path_or_object_id.replace(
            "\\",
            "/",
        )

        if path.endswith(".py"):
            path = path[:-3]

        return path.replace("/", ".")

    def _package_name(self,module: str) -> str:
        """Return the package portion of a module name."""

        if "." not in module:
            return module

        return module.rsplit(".", 1)[0]

    def _line_count(self,content: str) -> int:
        """Return the number of lines in content."""

        return len(content.splitlines()) or 1