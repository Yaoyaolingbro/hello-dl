import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.md"
CSS = ROOT / "docs" / "assets" / "css" / "custom.css"
CONFIG = ROOT / "mkdocs.yml"
HERO = ROOT / "docs" / "assets" / "images" / "home" / "learning-path-hero.png"


class HomepageTest(unittest.TestCase):
    def test_homepage_has_required_learning_routes(self):
        page = INDEX.read_text(encoding="utf-8")
        self.assertIn('class="home-hero"', page)
        self.assertIn('class="home-hero__copy"', page)
        self.assertIn('class="home-hero__visual"', page)
        for label in ("主线教材", "面试拆解", "科研速览"):
            self.assertIn(label, page)
        for target in (
            "01-math/",
            "02-deep-learning/",
            "03-advanced/",
            "04-applications/",
        ):
            self.assertIn(f'href="{target}"', page)

    def test_homepage_visual_and_responsive_contract(self):
        page = INDEX.read_text(encoding="utf-8")
        css = CSS.read_text(encoding="utf-8")
        self.assertTrue(HERO.is_file())
        self.assertIn("assets/images/home/learning-path-hero.png", page)
        self.assertRegex(css, r"\.home-hero\s*\{[^}]*display:\s*grid")
        self.assertIn("grid-template-columns: minmax(0, 1.2fr) minmax(18rem, 0.8fr)", css)
        self.assertIn("@media (max-width: 800px)", css)
        self.assertIn("prefers-reduced-motion: reduce", css)

    def test_theme_and_public_url_are_coordinated(self):
        css = CSS.read_text(encoding="utf-8")
        config = CONFIG.read_text(encoding="utf-8")
        for token in (
            "--dl-paper",
            "--dl-indigo",
            "--dl-teal",
            "--dl-orange",
        ):
            self.assertIn(token, css)
        self.assertIn("primary: custom", config)
        self.assertIn("accent: custom", config)
        self.assertIn("site_url: https://yaoyaolingbro.github.io/hello-dl/", config)


if __name__ == "__main__":
    unittest.main()
