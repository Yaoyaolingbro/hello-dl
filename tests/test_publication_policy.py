import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PART3_INDEX = ROOT / "docs" / "03-advanced" / "index.md"
PART3_ROOT = ROOT / "docs" / "03-advanced"
PART3_IMITATION = ROOT / "docs" / "03-advanced" / "imitation-learning"
PART4_EMBODIED_INDEX = ROOT / "docs" / "04-applications" / "embodied" / "index.md"
PART4_IMITATION = ROOT / "docs" / "04-applications" / "embodied" / "imitation-learning.md"
CONFIG = ROOT / "mkdocs.yml"
WRITING_STYLE = ROOT / "writing-style.md"


class PublicationPolicyTest(unittest.TestCase):
    def test_part3_does_not_publish_imitation_learning(self):
        part3_index = PART3_INDEX.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")

        self.assertNotIn("imitation-learning", part3_index)
        self.assertNotIn("模仿学习", part3_index)
        self.assertNotIn("03-advanced/imitation-learning", config)
        self.assertFalse(PART3_IMITATION.exists())

        published_part3 = "\n".join(
            page.read_text(encoding="utf-8") for page in PART3_ROOT.rglob("*.md")
        )
        self.assertNotIn("模仿学习", published_part3)
        self.assertNotIn("Imitation Learning", published_part3)

    def test_part4_keeps_application_level_imitation_learning_without_part3_prerequisite(self):
        embodied_index = PART4_EMBODIED_INDEX.read_text(encoding="utf-8")
        imitation_page = PART4_IMITATION.read_text(encoding="utf-8")

        self.assertIn("imitation-learning.md", embodied_index)
        self.assertTrue(PART4_IMITATION.is_file())
        self.assertNotIn("Part 3 模仿学习", embodied_index)
        self.assertNotIn("Part 3 模仿学习", imitation_page)

    def test_require_complete_is_reserved_for_a_fully_written_part2(self):
        guide = WRITING_STYLE.read_text(encoding="utf-8")

        self.assertIn(
            "日常发布运行 `scripts/check_part2_content.py`，允许保留如实标记为 `planned` 的页面。",
            guide,
        )
        self.assertIn(
            "只有当 Part 2 全部章节完成后，才运行 `scripts/check_part2_content.py --require-complete`",
            guide,
        )


if __name__ == "__main__":
    unittest.main()
