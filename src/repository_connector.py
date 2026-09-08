"""GitHub repository connector for Problem Statement 1."""

from __future__ import annotations

import base64
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import Settings
from src.utils.logger import get_logger


logger = get_logger(__name__)


class GitHubRateLimitError(requests.HTTPError):
    """Raised when GitHub rate limiting prevents a request from completing."""


@dataclass(frozen=True)
class RepositoryReference:
    """Validated repository owner, name, and selected GitHub ref."""

    owner: str
    name: str
    ref: str


class GitHubAPIClient:
    """Handles GitHub REST API communication only."""

    def __init__(self,token: str | None = None,timeout: int = 30) -> None:
        self.base_url = "https://api.github.com"
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    @retry(
        retry=retry_if_exception_type(
            (
                requests.ConnectionError,
                requests.Timeout,
                GitHubRateLimitError,
            )
        ),
        wait=wait_exponential(
            multiplier=1,
            min=1,
            max=8,
        ),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def get(self,path: str,params: dict[str, Any] | None = None) -> Any:
        """Perform one GET request against the GitHub REST API."""

        response = self.session.get(
            f"{self.base_url}{path}",
            params=params,
            timeout=self.timeout,
        )

        if response.status_code == 429 or (
            response.status_code == 403
            and response.headers.get("X-RateLimit-Remaining") == "0"
        ):
            retry_after = response.headers.get("Retry-After")

            message = "GitHub rate limit reached."

            if retry_after:
                message += (
                    f" Retry-After: {retry_after} seconds."
                )

            raise GitHubRateLimitError(
                message,
                response=response,
            )

        response.raise_for_status()

        return response.json()

    def get_paginated(self,path: str,params: dict[str, Any] | None = None) -> list[Any]:
        """Retrieve all pages from a GitHub list endpoint."""

        page = 1
        results: list[Any] = []

        while True:
            page_params = dict(params or {})
            page_params.update(
                {
                    "page": page,
                    "per_page": 100,
                }
            )

            batch = self.get(
                path,
                page_params,
            )

            if not isinstance(batch, list):
                return results

            results.extend(batch)

            if len(batch) < 100:
                return results

            page += 1


class RepositoryConnector:
    """Validates a repository and retrieves raw GitHub repository data."""

    _URL_PATTERN = re.compile(
        r"^https?://github\.com/"
        r"(?P<owner>[^/]+)/"
        r"(?P<repo>[^/#]+?)(?:\.git)?/?$"
    )

    def __init__(self,settings: Settings) -> None:
        self.settings = settings

        self.client = GitHubAPIClient(
            token=settings.github_token,
        )

    def validate_reference(self) -> RepositoryReference:
        """Validate the repository URL and selected GitHub ref."""

        match = self._URL_PATTERN.match(
            self.settings.repository.url.strip()
        )

        if not match:
            raise ValueError(
                "Invalid GitHub repository URL."
            )

        reference = RepositoryReference(
            owner=match.group("owner"),
            name=match.group("repo"),
            ref=self.settings.repository.ref.strip(),
        )

        if not reference.ref:
            raise ValueError(
                "Repository ref cannot be empty."
            )

        self.client.get(
            f"/repos/{reference.owner}/{reference.name}/commits/"
            f"{quote(reference.ref, safe='')}"
        )

        logger.info(
            "Validated repository %s/%s at ref %s.",
            reference.owner,
            reference.name,
            reference.ref,
        )

        return reference

    def fetch_repository_metadata(self,reference: RepositoryReference) -> dict[str, Any]:
        """Retrieve raw repository metadata and the selected revision."""

        repository = self.client.get(
            f"/repos/{reference.owner}/{reference.name}"
        )

        revision = self.client.get(
            f"/repos/{reference.owner}/{reference.name}/commits/"
            f"{quote(reference.ref, safe='')}"
        )

        return {
            "owner": reference.owner,
            "name": reference.name,
            "ref": reference.ref,
            "commit_sha": revision["sha"],
            "source_url": repository["html_url"],
            "default_branch": repository.get("default_branch"),
            "language": repository.get("language"),
        }

    def fetch_tree(self,reference: RepositoryReference,commit_sha: str) -> list[dict[str, Any]]:
        """Retrieve the repository tree for an exact commit."""

        response = self.client.get(
            f"/repos/{reference.owner}/{reference.name}/git/trees/"
            f"{quote(commit_sha, safe='')}",
            {"recursive": "1"},
        )

        if response.get("truncated"):
            raise RuntimeError(
                "GitHub returned a truncated repository tree."
            )

        return response.get("tree", [])

    def fetch_file_content(self,reference: RepositoryReference,path: str,commit_sha: str) -> dict[str, Any]:
        """Retrieve raw file content from an exact repository revision."""

        response = self.client.get(
            f"/repos/{reference.owner}/{reference.name}/contents/"
            f"{quote(path, safe='')}",
            {"ref": commit_sha},
        )

        content = response.get("content", "")
        encoding = response.get("encoding")

        if encoding == "base64":
            try:
                decoded_content = base64.b64decode(
                    content.replace("\n", "")
                ).decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    "Unsupported or malformed encoding for "
                    "repository file: %s",
                    path,
                )

                return {
                    "path": response.get("path", path),
                    "sha": response.get("sha"),
                    "size": response.get("size"),
                    "content": None,
                    "source_url": response.get("html_url"),
                    "commit_sha": commit_sha,
                    "status": "failed",
                    "failure_reason": (
                        "unsupported_or_invalid_encoding"
                    ),
                }
        else:
            decoded_content = content

        return {
            "path": response.get("path", path),
            "sha": response.get("sha"),
            "size": response.get("size"),
            "content": decoded_content,
            "source_url": response.get("html_url"),
            "commit_sha": commit_sha,
        }

    def fetch_files(self,reference: RepositoryReference,paths: list[str],commit_sha: str) -> list[dict[str, Any]]:
        """Fetch multiple files using bounded concurrency."""

        if not paths:
            return []

        max_workers = min(8, len(paths))

        results: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            futures = {
                executor.submit(
                    self.fetch_file_content,
                    reference,
                    path,
                    commit_sha,
                ): path
                for path in paths
            }

            for future in as_completed(futures):
                path = futures[future]

                try:
                    results.append(
                        future.result()
                    )

                except Exception as exc:
                    logger.exception(
                        "Failed to fetch repository file: %s",
                        path,
                    )

                    failures.append(
                        {
                            "path": path,
                            "error": str(exc),
                        }
                    )

        if failures:
            raise RuntimeError(
                f"Failed to fetch {len(failures)} repository file(s). "
                f"Failures: {failures}"
            )

        return results

    def fetch_issues(self,reference: RepositoryReference) -> list[dict[str, Any]]:
        """Retrieve repository issues from GitHub."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/issues",
            {"state": "all"},
        )

    def fetch_issue_comments(self,reference: RepositoryReference,issue_number: int) -> list[dict[str, Any]]:
        """Retrieve comments for a GitHub issue."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/issues/"
            f"{issue_number}/comments"
        )

    def fetch_pull_requests(self,reference: RepositoryReference) -> list[dict[str, Any]]:
        """Retrieve repository pull requests."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/pulls",
            {"state": "all"},
        )

    def fetch_pull_request_reviews(self,reference: RepositoryReference,pull_number: int) -> list[dict[str, Any]]:
        """Retrieve reviews for a pull request."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/pulls/"
            f"{pull_number}/reviews"
        )

    def fetch_commits(self,reference: RepositoryReference) -> list[dict[str, Any]]:
        """Retrieve repository commits."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/commits"
        )

    def fetch_releases(self,reference: RepositoryReference,) -> list[dict[str, Any]]:
        """Retrieve repository releases."""

        return self.client.get_paginated(
            f"/repos/{reference.owner}/{reference.name}/releases"
        )