import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_part2_content.py"


class Part2ContentCheckerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_page(self, relative_path: str, body: str) -> None:
        target = self.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")

    def run_checker(self, *extra_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--root",
                str(self.root),
                *extra_args,
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def valid_page(self, *, status: str = "complete", body: str = "正文内容。") -> str:
        return f"""
        ---
        level: basic
        roles:
          - core
        prerequisites: []
        estimated_time: 20min
        status: {status}
        ---

        # 示例章节

        {body}
        """

    def test_reports_unsupported_role(self):
        self.write_page(
            "chapter.md",
            self.valid_page().replace("  - core", "  - exam"),
        )

        result = self.run_checker()

        self.assertEqual(result.returncode, 1)
        self.assertIn("chapter.md: unsupported role 'exam'", result.stdout)

    def test_require_complete_rejects_planned_page(self):
        self.write_page("chapter.md", self.valid_page(status="planned"))

        result = self.run_checker("--require-complete")

        self.assertEqual(result.returncode, 1)
        self.assertIn("chapter.md: status must be complete", result.stdout)

    def test_reports_missing_relative_markdown_target(self):
        self.write_page(
            "chapter.md",
            self.valid_page(body="继续阅读[下一章](missing.md)。"),
        )

        result = self.run_checker()

        self.assertEqual(result.returncode, 1)
        self.assertIn("chapter.md: missing link target missing.md", result.stdout)

    def test_accepts_valid_pages_and_existing_links(self):
        self.write_page(
            "index.md",
            self.valid_page(body="阅读[下一章](chapter.md)，或查看[外部资料](https://example.com)。"),
        )
        self.write_page("chapter.md", self.valid_page())

        result = self.run_checker("--require-complete")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Validated 2 Part 2 pages", result.stdout)


if __name__ == "__main__":
    unittest.main()
