import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.md"
CSS = ROOT / "docs" / "assets" / "css" / "custom.css"
CONFIG = ROOT / "mkdocs.yml"
HERO_PNG = ROOT / "docs" / "assets" / "images" / "home" / "learning-path-hero.png"
HERO_AVIF = ROOT / "docs" / "assets" / "images" / "home" / "learning-path-hero.avif"


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
        self.assertTrue(HERO_PNG.is_file())
        self.assertIn("assets/images/home/learning-path-hero.png", page)
        self.assertRegex(css, r"\.home-hero\s*\{[^}]*display:\s*grid")
        self.assertIn("grid-template-columns: minmax(0, 1.2fr) minmax(18rem, 0.8fr)", css)
        self.assertIn("@media (max-width: 800px)", css)
        self.assertIn("prefers-reduced-motion: reduce", css)

    def test_mobile_hero_heading_releases_the_forced_line_break(self):
        css = CSS.read_text(encoding="utf-8")
        mobile_css = css.split("@media (max-width: 800px)", 1)[1].split("@media", 1)[0]
        self.assertRegex(
            mobile_css,
            r"\.md-typeset\s+\.home-hero\s+h1\s+br\s*\{[^}]*display:\s*none",
        )

    def test_homepage_serves_a_smaller_avif_with_png_fallback(self):
        page = INDEX.read_text(encoding="utf-8")
        css = CSS.read_text(encoding="utf-8")
        self.assertTrue(HERO_AVIF.is_file())
        self.assertLess(HERO_AVIF.stat().st_size, HERO_PNG.stat().st_size * 0.6)
        self.assertRegex(
            page,
            r'<picture>\s*<source srcset="assets/images/home/learning-path-hero\.avif" '
            r'type="image/avif">\s*<img src="assets/images/home/learning-path-hero\.png"',
        )
        self.assertRegex(
            css,
            r"\.md-typeset\s+\.home-hero__visual\s+picture\s*\{"
            r"[^}]*width:\s*100%[^}]*height:\s*100%",
        )
        self.assertRegex(
            css,
            r"\.md-typeset\s+\.home-hero__visual\s+img\s*\{[^}]*object-fit:\s*contain",
        )

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

    def test_deploy_validates_navigation_and_builds_strictly(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        self.assertIn("--mkdocs-config mkdocs.yml", workflow)
        self.assertNotIn("--require-complete", workflow)
        self.assertIn("mkdocs build --strict", workflow)
        self.assertIn("mkdocs gh-deploy --force", workflow)


if __name__ == "__main__":
    unittest.main()
