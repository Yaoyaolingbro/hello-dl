import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
PART2_ROOT = REPO_ROOT / "docs" / "02-deep-learning"
VISUAL_CSS = REPO_ROOT / "docs" / "assets" / "css" / "lesson-visuals.css"
VISUAL_JS = REPO_ROOT / "docs" / "assets" / "js" / "lesson-visuals.js"
MERMAID_PAGE_ALLOWLIST = {
    "docs/02-deep-learning/02-neural-network-foundations/computational-graphs.md",
    "docs/02-deep-learning/03-training/backpropagation.md",
    "docs/02-deep-learning/03-training/initialization.md",
    "docs/02-deep-learning/03-training/minibatch-and-training-loop.md",
    "docs/02-deep-learning/03-training/normalization.md",
    "docs/02-deep-learning/04-components/attention.md",
    "docs/02-deep-learning/04-components/convolution.md",
    "docs/02-deep-learning/04-components/graph-message-passing.md",
    "docs/02-deep-learning/04-components/recurrence.md",
    "docs/02-deep-learning/04-components/residual-connections.md",
}
MERMAID_CONFIG_ALLOWLIST = {
    "- name: mermaid",
    "class: mermaid",
    "- https://unpkg.com/mermaid@11/dist/mermaid.esm.min.mjs",
}


class VisualContentTest(unittest.TestCase):
    def test_lesson_visual_assets_are_registered(self):
        config = MKDOCS_CONFIG.read_text(encoding="utf-8")

        self.assertRegex(
            config,
            r"(?m)^\s*- assets/css/lesson-visuals\.css\s*$",
        )
        self.assertRegex(
            config,
            r"(?m)^\s*- assets/js/lesson-visuals\.js\s*$",
        )

    def test_registered_lesson_visual_assets_exist(self):
        self.assertTrue(VISUAL_CSS.is_file(), VISUAL_CSS)
        self.assertTrue(VISUAL_JS.is_file(), VISUAL_JS)

    def test_interactive_figures_include_accessible_fallbacks(self):
        problems = []

        for page in sorted(PART2_ROOT.rglob("*.md")):
            content = page.read_text(encoding="utf-8")
            for number, match in enumerate(
                re.finditer(r"<figure\b[^>]*data-lesson-visual\b.*?</figure>", content, re.DOTALL),
                start=1,
            ):
                figure = match.group(0)
                if not re.search(
                    r"data-lesson-stage\b[^>]*\baria-label=(?:\"[^\"]+\"|'[^']+')",
                    figure,
                ):
                    problems.append(
                        f"{page.relative_to(REPO_ROOT)} figure {number}: "
                        "missing an aria-label on data-lesson-stage"
                    )
                if not re.search(r"<ol\b[^>]*data-lesson-steps\b", figure):
                    problems.append(
                        f"{page.relative_to(REPO_ROOT)} figure {number}: "
                        "missing an ordered textual step list"
                    )

        self.assertEqual(problems, [], "\n".join(problems))

    def test_mermaid_usage_matches_migration_allowlist(self):
        mermaid_pages = set()
        for page in sorted(PART2_ROOT.rglob("*.md")):
            if "```mermaid" in page.read_text(encoding="utf-8"):
                mermaid_pages.add(str(page.relative_to(REPO_ROOT)))

        config = MKDOCS_CONFIG.read_text(encoding="utf-8")
        mermaid_config = {
            line.strip()
            for line in config.splitlines()
            if re.search(r"(?i)mermaid", line)
        }

        self.assertSetEqual(mermaid_pages, MERMAID_PAGE_ALLOWLIST)
        self.assertSetEqual(mermaid_config, MERMAID_CONFIG_ALLOWLIST)


if __name__ == "__main__":
    unittest.main()
