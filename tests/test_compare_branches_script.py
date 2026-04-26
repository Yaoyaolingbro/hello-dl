import subprocess
import unittest
from pathlib import Path


class CompareBranchesScriptTest(unittest.TestCase):
    def test_help_describes_default_branches_and_preview_behavior(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "compare-branches.sh"

        result = subprocess.run(
            ["bash", str(script), "--help"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("codex", result.stdout)
        self.assertIn("claude-code", result.stdout)
        self.assertIn("mkdocs build --strict", result.stdout)
        self.assertIn("http://127.0.0.1:8010", result.stdout)
        self.assertIn("http://127.0.0.1:8011", result.stdout)


if __name__ == "__main__":
    unittest.main()
