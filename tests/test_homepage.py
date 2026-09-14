import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs" / "index.md"
CSS = ROOT / "docs" / "assets" / "css" / "custom.css"
CONFIG = ROOT / "mkdocs.yml"
HERO_PNG = ROOT / "docs" / "assets" / "images" / "home" / "learning-path-hero.png"
HERO_AVIF = ROOT / "docs" / "assets" / "images" / "home" / "learning-path-hero.avif"


def split_selector_list(selector_group):
    selectors = []
    start = 0
    parentheses_depth = 0
    bracket_depth = 0
    for index, character in enumerate(selector_group):
        if character == "(":
            parentheses_depth += 1
        elif character == ")":
            parentheses_depth = max(0, parentheses_depth - 1)
        elif character == "[":
            bracket_depth += 1
        elif character == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif character == "," and parentheses_depth == bracket_depth == 0:
            selectors.append(selector_group[start:index].strip())
            start = index + 1
    selectors.append(selector_group[start:].strip())
    return selectors


def strip_balanced_content(text, opening, closing):
    depth = 0
    stripped = []
    for character in text:
        if character == opening:
            depth += 1
        if depth == 0:
            stripped.append(character)
        if character == closing and depth:
            depth -= 1
    return "".join(stripped)


def selector_subject(selector):
    selector_without_attributes = strip_balanced_content(selector, "[", "]")
    parentheses_depth = 0
    subject_start = 0
    for index, character in enumerate(selector_without_attributes):
        if character == "(":
            parentheses_depth += 1
        elif character == ")":
            parentheses_depth = max(0, parentheses_depth - 1)
        elif parentheses_depth == 0 and (
            character.isspace() or character in ">+~"
        ):
            subject_start = index + 1
    return selector_without_attributes[subject_start:].strip()


def find_closing_brace(css, opening_brace):
    depth = 0
    for index in range(opening_brace, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(css)


def iter_css_rules(css, parent_selectors=()):
    """Yield effective selectors from the balanced rule shapes used in custom.css."""
    position = 0
    while (opening_brace := css.find("{", position)) != -1:
        prelude = css[position:opening_brace].rsplit(";", 1)[-1].strip()
        closing_brace = find_closing_brace(css, opening_brace)
        declarations = css[opening_brace + 1 : closing_brace]
        if prelude.startswith("@"):
            effective_selectors = parent_selectors
        else:
            selectors = split_selector_list(prelude)
            if parent_selectors:
                effective_selectors = [
                    child.replace("&", parent)
                    if "&" in child
                    else f"{parent} {child}"
                    for parent in parent_selectors
                    for child in selectors
                ]
            else:
                effective_selectors = selectors
            for selector in effective_selectors:
                yield selector, declarations
        yield from iter_css_rules(declarations, effective_selectors)
        position = closing_brace + 1


def find_homepage_width_overrides(css):
    css_without_comments = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    overrides = []
    for selector, declarations in iter_css_rules(css_without_comments):
        if not re.search(
            r"(?:^|;)\s*max-width\s*:", declarations, flags=re.IGNORECASE
        ):
            continue
        selector_without_attributes = strip_balanced_content(selector, "[", "]")
        subject_without_functions = strip_balanced_content(
            selector_subject(selector), "(", ")"
        )
        if re.search(
            r"\.md-main__inner(?![A-Za-z0-9_-])", subject_without_functions
        ) and re.search(
            r"\.home-page(?![A-Za-z0-9_-])", selector_without_attributes
        ):
            overrides.append(selector.strip())
    return overrides


class HomepageTest(unittest.TestCase):
    def test_homepage_inherits_the_standard_material_content_width(self):
        css = CSS.read_text(encoding="utf-8")
        self.assertEqual([], find_homepage_width_overrides(css))
        self.assertEqual(
            ["body:has(.home-page) .md-main__inner"],
            find_homepage_width_overrides(
                "body:has(.home-page) .md-main__inner { max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                ".md-main__inner .article, .home-page { max-width: 40rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                "/* .md-main__inner */ .home-page { max-width: 40rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                ".md-main__innerish .home-page-preview { max-width: 40rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                ".md-main__inner .home-page__section { max-width: 40rem; }"
            ),
        )
        self.assertEqual(
            [".md-main__inner:has(.article, .home-page)"],
            find_homepage_width_overrides(
                ".md-main__inner:has(.article, .home-page) { max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                ".md-main__inner .home-page { --home-max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                ".md-main__inner .home-page { max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            [],
            find_homepage_width_overrides(
                '.md-main__inner[data-example=".home-page"] { max-width: 80rem; }'
            ),
        )
        self.assertEqual(
            ['.md-main__inner[data-x="home page"]:has(.home-page)'],
            find_homepage_width_overrides(
                '.md-main__inner[data-x="home page"]:has(.home-page) '
                "{ max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            ['.md-main__inner[data-x="home\tpage"]:has(.home-page)'],
            find_homepage_width_overrides(
                '.md-main__inner[data-x="home\tpage"]:has(.home-page) '
                "{ max-width: 80rem; }"
            ),
        )
        self.assertEqual(
            [".home-page .md-main__inner"],
            find_homepage_width_overrides(
                ".home-page .md-main__inner { Max-Width: 80rem; }"
            ),
        )
        self.assertEqual(
            [".home-page .md-main__inner"],
            find_homepage_width_overrides(
                ".home-page { .md-main__inner { max-width: 80rem; } }"
            ),
        )

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
