#!/usr/bin/env python3
"""Validate the metadata and local links of Part 2 Markdown pages."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml


ALLOWED_LEVELS = {"basic", "intermediate", "advanced"}
ALLOWED_ROLES = {"core", "interview", "research"}
ALLOWED_STATUSES = {"planned", "draft", "complete"}
REQUIRED_FIELDS = {
    "level",
    "roles",
    "prerequisites",
    "estimated_time",
    "status",
}
ESTIMATED_TIME = re.compile(r"^[1-9][0-9]*min$")
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
PLACEHOLDER_MARKERS = ("TODO", "TBD", "待补充", "正文内容将在")
DISPLAY_MATH = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
BARE_QQUAD = re.compile(r"(?<!\\)\bqquad\b")


class _MkDocsConfigLoader(yaml.SafeLoader):
    """Safe loader that treats MkDocs Python-name tags as opaque strings."""


_MkDocsConfigLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:",
    lambda _loader, suffix, _node: suffix,
)


def _split_front_matter(text: str) -> tuple[dict[str, Any] | None, str, str | None]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text, "missing YAML front matter"

    try:
        closing = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration:
        return None, text, "unclosed YAML front matter"

    raw_metadata = "\n".join(lines[1:closing])
    try:
        metadata = yaml.safe_load(raw_metadata)
    except yaml.YAMLError as error:
        return None, text, f"invalid YAML front matter: {error.problem or 'parse error'}"

    if not isinstance(metadata, dict):
        return None, text, "YAML front matter must be a mapping"

    body = "\n".join(lines[closing + 1 :])
    return metadata, body, None


def _validate_metadata(
    path: Path,
    metadata: dict[str, Any],
    require_complete: bool,
) -> list[str]:
    errors: list[str] = []

    for field in sorted(REQUIRED_FIELDS - metadata.keys()):
        errors.append(f"missing metadata field '{field}'")

    level = metadata.get("level")
    if level is not None and level not in ALLOWED_LEVELS:
        errors.append(f"unsupported level '{level}'")

    roles = metadata.get("roles")
    if roles is not None:
        if not isinstance(roles, list) or not roles:
            errors.append("roles must be a non-empty list")
        else:
            for role in roles:
                if role not in ALLOWED_ROLES:
                    errors.append(f"unsupported role '{role}'")

    prerequisites = metadata.get("prerequisites")
    if prerequisites is not None and not isinstance(prerequisites, list):
        errors.append("prerequisites must be a list")

    estimated_time = metadata.get("estimated_time")
    if estimated_time is not None and (
        not isinstance(estimated_time, str) or not ESTIMATED_TIME.fullmatch(estimated_time)
    ):
        errors.append("estimated_time must use '<minutes>min'")

    status = metadata.get("status")
    if status is not None and status not in ALLOWED_STATUSES:
        errors.append(f"unsupported status '{status}'")
    if require_complete and status != "complete":
        errors.append("status must be complete")

    return errors


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def _validate_links(path: Path, body: str) -> list[str]:
    errors: list[str] = []
    for match in MARKDOWN_LINK.finditer(body):
        target = _link_target(match.group(1))
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue

        local_part = unquote(target.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0])
        if not local_part:
            continue

        resolved = (path.parent / local_part).resolve()
        exists = resolved.exists() or (resolved.is_dir() and (resolved / "index.md").exists())
        if not exists:
            errors.append(f"missing link target {target}")
    return errors


def _validate_body(body: str, status: Any) -> list[str]:
    errors: list[str] = []
    h1_count = len(re.findall(r"(?m)^#\s+\S", body))
    if h1_count != 1:
        errors.append(f"expected exactly one H1, found {h1_count}")

    if status == "complete":
        for marker in PLACEHOLDER_MARKERS:
            if marker in body:
                errors.append(f"complete page contains placeholder marker '{marker}'")
    if any(BARE_QQUAD.search(math.group()) for math in DISPLAY_MATH.finditer(body)):
        errors.append("suspicious bare 'qquad' in math")
    return errors


def validate_part2(root: Path, require_complete: bool = False) -> list[str]:
    root = root.resolve()
    errors: list[str] = []

    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        metadata, body, front_matter_error = _split_front_matter(text)
        if front_matter_error:
            errors.append(f"{relative}: {front_matter_error}")
            continue

        assert metadata is not None
        page_errors = [
            *_validate_metadata(path, metadata, require_complete),
            *_validate_body(body, metadata.get("status")),
            *_validate_links(path, body),
        ]
        errors.extend(f"{relative}: {error}" for error in page_errors)

    return sorted(errors)


def _part2_nav_targets(config_path: Path) -> set[str]:
    config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=_MkDocsConfigLoader)
    nav = config.get("nav", []) if isinstance(config, dict) else []

    part2_items: Any = []
    for item in nav:
        if isinstance(item, dict):
            for title, children in item.items():
                if str(title).startswith("Part 2"):
                    part2_items = children
                    break

    targets: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, str) and value.endswith(".md"):
            prefix = "02-deep-learning/"
            targets.add(value[len(prefix) :] if value.startswith(prefix) else value)
        elif isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)

    collect(part2_items)
    return targets


def validate_navigation(root: Path, config_path: Path) -> list[str]:
    pages = {path.relative_to(root).as_posix() for path in root.rglob("*.md")}
    targets = _part2_nav_targets(config_path)
    errors = [
        f"{page}: page is missing from Part 2 navigation" for page in pages - targets
    ]
    errors.extend(
        f"{target}: navigation target does not exist" for target in targets - pages
    )
    return sorted(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/02-deep-learning"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--mkdocs-config", type=Path)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Part 2 root does not exist: {args.root}")
        return 2

    errors = validate_part2(args.root, require_complete=args.require_complete)
    if args.mkdocs_config:
        errors.extend(validate_navigation(args.root.resolve(), args.mkdocs_config.resolve()))
        errors.sort()
    if errors:
        print("\n".join(errors))
        return 1

    page_count = sum(1 for _ in args.root.rglob("*.md"))
    print(f"Validated {page_count} Part 2 pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
