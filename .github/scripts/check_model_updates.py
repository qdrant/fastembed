"""Report HuggingFace repositories of the supported models which got new commits recently

Two kinds of repositories are checked:
    - source: the repository fastembed downloads a model from (`sources.hf`). Fastembed downloads
        the latest revision, so changed files reach users without a fastembed release.
    - upstream: the repository a model was exported from: the model name, or the `base_model`
        from the card of an exported source. Its changes might require re-exporting the onnx
        model, updating the model description or adding a warning.

Prints a report of the changed repositories and optionally writes an issue per affected model.
Exits with 1 if any repository could not be checked.

Usage:
    python .github/scripts/check_model_updates.py --days 7 --issues-file issues.json
"""

import argparse
import fnmatch
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.hf_api import GitCommitInfo, ModelInfo, RepoFile
from huggingface_hub.utils import RepositoryNotFoundError

from fastembed import (
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
    SparseTextEmbedding,
    TextEmbedding,
)
from fastembed.common.model_management import ModelManagement
from fastembed.rerank.cross_encoder import TextCrossEncoder

EMBEDDING_CLASSES: list[type[ModelManagement[Any]]] = [
    TextEmbedding,
    SparseTextEmbedding,
    LateInteractionTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    TextCrossEncoder,
]

# downloaded along with model_file and additional_files, keep in sync with
# ModelManagement.download_files_from_huggingface
DEFAULT_DOWNLOADED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
]
DOC_FILES = [
    "*.md",
    "LICENSE*",
    "NOTICE*",
    ".gitattributes",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.svg",
]
# organizations publishing onnx exports of other models under the same model names
EXPORTER_ORGS = {"qdrant", "xenova", "onnx-community"}
HF_URL = "https://huggingface.co"
MAX_LISTED = 15
MAX_ISSUE_BODY = 60000  # issue bodies are limited to 65536 characters

# severities of changes, indices in SECTIONS
DOWNLOADED_FILES, MODEL_FILES, DOCS = range(3)
SECTIONS = [
    (
        "Files fastembed downloads have changed",
        "Users get these files without a fastembed release, check that the models still work and "
        "produce the expected embeddings.",
    ),
    (
        "Model files have changed",
        "Consider re-exporting the onnx models, updating the model descriptions or adding "
        "warnings.",
    ),
    (
        "Model cards or docs have changed",
        "Check for license changes, deprecations and usage notes.",
    ),
]

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class ModelUsage:
    """A supported model using a repository"""

    model: str
    embedding_class: str
    downloaded_patterns: list[str] = field(default_factory=list)  # empty for upstream usages

    @property
    def label(self) -> str:
        return f"{self.model} ({self.embedding_class})"

    def downloaded(self, paths: list[str]) -> list[str]:
        return [
            path
            for path in paths
            if any(fnmatch.fnmatch(path, pattern) for pattern in self.downloaded_patterns)
        ]


@dataclass
class TrackedRepo:
    repo_id: str
    info: ModelInfo
    sources: list[ModelUsage] = field(default_factory=list)  # models downloaded from the repo
    exports: list[ModelUsage] = field(default_factory=list)  # models exported from the repo
    commits: list[GitCommitInfo] = field(default_factory=list)  # within the window, newest first
    base_revision: str | None = None  # the last commit before the window, None for a new repo
    changes: dict[str, str] = field(default_factory=dict)  # path -> added / modified / removed

    def changed_downloaded_files(self, model: str | None = None) -> dict[str, list[str]]:
        """Changed files fastembed downloads by model labels, of all models or of `model` only"""
        paths = sorted(self.changes)
        return {
            usage.label: files
            for usage in self.sources
            if model in (None, usage.model) and (files := usage.downloaded(paths))
        }

    @property
    def changed_model_files(self) -> list[str]:
        return [
            path
            for path in sorted(self.changes)
            if not any(fnmatch.fnmatch(path.rsplit("/", 1)[-1], pattern) for pattern in DOC_FILES)
        ]

    def severity(self, model: str | None = None) -> int:
        """Severity of the changes for all models or for `model` only"""
        if self.changed_downloaded_files(model):
            return DOWNLOADED_FILES
        if self.changed_model_files:
            return MODEL_FILES
        return DOCS


def parallel_map(func: Callable[[T], R], items: list[T]) -> list[R | Exception]:
    def call(item: T) -> R | Exception:
        try:
            return func(item)
        except Exception as e:
            return e

    with ThreadPoolExecutor(max_workers=8) as executor:
        return list(executor.map(call, items))


def base_models(info: ModelInfo) -> list[str]:
    base_model = info.card_data.get("base_model") if info.card_data else None
    if isinstance(base_model, str):
        return [base_model]
    return list(base_model or [])


def collect_repos(api: HfApi) -> tuple[list[TrackedRepo], list[str]]:
    """Find source and upstream repositories of all supported models

    Returns:
        tracked repositories and errors of the repositories which could not be checked
    """
    models: list[tuple[str, ModelUsage]] = [  # source repo id, usage
        (
            description.sources.hf,
            ModelUsage(
                model=description.model,
                embedding_class=cls.__name__,
                downloaded_patterns=DEFAULT_DOWNLOADED_FILES
                + [description.model_file]
                + description.additional_files,
            ),
        )
        for cls in EMBEDDING_CLASSES
        for description in cls._list_supported_models()
        if description.sources.hf
    ]
    repos: dict[str, TrackedRepo] = {}
    errors: list[str] = []

    source_ids = sorted({source_id for source_id, _ in models})
    for repo_id, info in zip(source_ids, parallel_map(api.model_info, source_ids)):
        if isinstance(info, Exception):
            errors.append(f"source `{repo_id}`: {type(info).__name__}: {info}")
        else:
            repos[repo_id.lower()] = TrackedRepo(repo_id=info.id, info=info)

    upstream_candidates: list[tuple[ModelUsage, TrackedRepo, set[str]]] = []
    for source_id, usage in models:
        source = repos.get(source_id.lower())
        if source is None:
            continue
        source.sources.append(usage)
        candidates = {usage.model}
        # base_model of a repo serving its own model is the base it was trained from, only
        # base_model of an export points to the model it was exported from
        is_export = (
            usage.model.lower() != source_id.lower()
            or source_id.split("/")[0].lower() in EXPORTER_ORGS
        )
        if is_export:
            candidates.update(base_models(source.info))
        upstream_candidates.append((usage, source, candidates))

    candidate_ids = sorted(
        {
            candidate
            for _, _, candidates in upstream_candidates
            for candidate in candidates
            if candidate.lower() not in repos
        }
    )
    for repo_id, info in zip(candidate_ids, parallel_map(api.model_info, candidate_ids)):
        # model names of variants, e.g. `-Q` ones, are not repositories
        if isinstance(info, RepositoryNotFoundError):
            continue
        if isinstance(info, Exception):
            errors.append(f"upstream `{repo_id}`: {type(info).__name__}: {info}")
            continue
        # resolved ids merge case variants, e.g. `snowflake/...` and `Snowflake/...`
        repo = repos.setdefault(info.id.lower(), TrackedRepo(repo_id=info.id, info=info))
        repos.setdefault(repo_id.lower(), repo)

    for usage, source, candidates in upstream_candidates:
        for candidate in sorted(candidates):
            upstream = repos.get(candidate.lower())
            export = ModelUsage(usage.model, usage.embedding_class)
            if upstream is not None and upstream is not source and export not in upstream.exports:
                upstream.exports.append(export)

    # case variants share a repo object
    unique = {id(repo): repo for repo in repos.values()}
    return list(unique.values()), errors


def list_files(api: HfApi, repo_id: str, revision: str) -> dict[str, str]:
    return {
        item.path: item.blob_id
        for item in api.list_repo_tree(repo_id, revision=revision, recursive=True)
        if isinstance(item, RepoFile)
    }


def fetch_changes(api: HfApi, repo: TrackedRepo, since: datetime) -> TrackedRepo:
    """Fill the commits made since `since` and the files they changed"""
    commits = api.list_repo_commits(repo.repo_id)
    repo.commits = [commit for commit in commits if commit.created_at >= since]
    if not repo.commits:
        return repo

    repo.base_revision = next(
        (commit.commit_id for commit in commits if commit.created_at < since), None
    )
    old = list_files(api, repo.repo_id, repo.base_revision) if repo.base_revision else {}
    new = list_files(api, repo.repo_id, repo.commits[0].commit_id)
    for path in old.keys() | new.keys():
        if path not in old:
            repo.changes[path] = "added"
        elif path not in new:
            repo.changes[path] = "removed"
        elif old[path] != new[path]:
            repo.changes[path] = "modified"
    return repo


def format_list(items: list[str]) -> str:
    listed = ", ".join(items[:MAX_LISTED])
    if len(items) > MAX_LISTED:
        listed += f" and {len(items) - MAX_LISTED} more"
    return listed


def render_repo(repo: TrackedRepo) -> list[str]:
    repo_url = f"{HF_URL}/{repo.repo_id}"
    lines = [f"#### [{repo.repo_id}]({repo_url})", ""]

    downloaded = repo.changed_downloaded_files()
    for usage in repo.sources:
        files = downloaded.get(usage.label)
        changed = f": `{'`, `'.join(files)}` changed" if files else ""
        lines.append(f"- source of `{usage.label}`{changed}")
    for usage in repo.exports:
        lines.append(f"- upstream of `{usage.label}`")

    since = f"`{repo.base_revision[:8]}`" if repo.base_revision else "the repository creation"
    changes = [f"`{path}` ({status})" for path, status in sorted(repo.changes.items())]
    lines.append(f"- changed files since {since}: {format_list(changes) or 'none'}")

    lines.append("- commits:")
    for commit in repo.commits[:MAX_LISTED]:
        lines.append(
            f"  - {commit.created_at:%Y-%m-%d} [`{commit.commit_id[:8]}`]"
            f"({repo_url}/commit/{commit.commit_id}) {commit.title}"
        )
    if len(repo.commits) > MAX_LISTED:
        lines.append(f"  - and {len(repo.commits) - MAX_LISTED} more")
    lines.append("")
    return lines


def sorted_repos(repos: list[TrackedRepo]) -> list[TrackedRepo]:
    return sorted(repos, key=lambda repo: repo.repo_id.lower())


def render_report(repos: list[TrackedRepo], errors: list[str], since: datetime) -> str:
    lines = [f"## Model repositories changed since {since:%Y-%m-%d %H:%M} UTC", ""]
    if not repos and not errors:
        lines.append("No changes.")
    for severity, (title, hint) in enumerate(SECTIONS):
        section_repos = [repo for repo in repos if repo.severity() == severity]
        if not section_repos:
            continue
        lines += [f"### {title}", "", hint, ""]
        for repo in sorted_repos(section_repos):
            lines += render_repo(repo)
    if errors:
        lines += ["### Repositories which could not be checked", ""]
        lines += [f"- {error}" for error in errors]
        lines.append("")
    return "\n".join(lines)


def render_issues(repos: list[TrackedRepo], since: datetime) -> list[dict[str, str]]:
    """An issue per model with the changes of its source and upstream repositories"""
    model_repos: dict[str, list[TrackedRepo]] = {}
    for repo in repos:
        for usage in repo.sources + repo.exports:
            related = model_repos.setdefault(usage.model, [])
            if all(repo is not other for other in related):
                related.append(repo)

    issues = []
    for model, related in sorted(model_repos.items(), key=lambda item: item[0].lower()):
        title, hint = SECTIONS[min(repo.severity(model) for repo in related)]
        lines = [
            f"### {title}",
            "",
            hint,
            "",
            f"Repositories of `{model}` changed since {since:%Y-%m-%d %H:%M} UTC:",
            "",
        ]
        for repo in sorted_repos(related):
            lines += render_repo(repo)
        issues.append(
            {"title": f"Model update: {model}", "body": "\n".join(lines)[:MAX_ISSUE_BODY]}
        )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="report commits made since 00:00 UTC this many days ago (default: 7)",
    )
    parser.add_argument(
        "--issues-file",
        help="write a json list of issues, {title, body} per affected model, to this file",
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    since = today - timedelta(days=args.days)

    api = HfApi()
    repos, errors = collect_repos(api)
    recent = [
        repo
        for repo in repos
        if repo.info.last_modified is None or repo.info.last_modified >= since
    ]
    changed: list[TrackedRepo] = []
    for repo, result in zip(recent, parallel_map(lambda r: fetch_changes(api, r, since), recent)):
        if isinstance(result, Exception):
            errors.append(f"`{repo.repo_id}`: {type(result).__name__}: {result}")
        elif result.commits:
            changed.append(result)

    report = render_report(changed, errors, since)
    print(report)
    if summary_path := os.getenv("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as summary:
            summary.write(report + "\n")
    if args.issues_file:
        with open(args.issues_file, "w") as issues_file:
            json.dump(render_issues(changed, since), issues_file)

    print(f"Checked {len(repos)} repositories, {len(changed)} changed", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
