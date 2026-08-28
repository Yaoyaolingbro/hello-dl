import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"
PART2_ROOT = REPO_ROOT / "docs" / "02-deep-learning"
VISUAL_CSS = REPO_ROOT / "docs" / "assets" / "css" / "lesson-visuals.css"
VISUAL_JS = REPO_ROOT / "docs" / "assets" / "js" / "lesson-visuals.js"
WRITING_STYLE = REPO_ROOT / "writing-style.md"
PART2_STYLE = REPO_ROOT / "docs" / "superpowers" / "style-part2-dl-basics.md"
WRITING_WORKFLOW = REPO_ROOT / "chapter-writing-workflow.md"
COMPUTATIONAL_GRAPHS_PAGE = (
    PART2_ROOT
    / "02-neural-network-foundations"
    / "computational-graphs.md"
)
CONVOLUTION_PAGE = PART2_ROOT / "04-components" / "convolution.md"
MERMAID_PAGE_ALLOWLIST = {
    "docs/02-deep-learning/03-training/backpropagation.md",
    "docs/02-deep-learning/03-training/initialization.md",
    "docs/02-deep-learning/03-training/minibatch-and-training-loop.md",
    "docs/02-deep-learning/03-training/normalization.md",
    "docs/02-deep-learning/04-components/attention.md",
    "docs/02-deep-learning/04-components/graph-message-passing.md",
    "docs/02-deep-learning/04-components/recurrence.md",
    "docs/02-deep-learning/04-components/residual-connections.md",
}
MERMAID_CONFIG_ALLOWLIST = {
    "- name: mermaid",
    "class: mermaid",
    "- https://unpkg.com/mermaid@11/dist/mermaid.esm.min.mjs",
}


OPENING_TAG_RE = re.compile(r"<[A-Za-z][A-Za-z0-9:-]*\b[^>]*>", re.DOTALL)


def _opening_tags_with_attribute(markup, attribute):
    attribute_re = re.compile(
        rf"(?<![\w:-]){re.escape(attribute)}(?=\s|=|/?>)",
    )
    return [
        tag
        for tag in OPENING_TAG_RE.findall(markup)
        if attribute_re.search(tag)
    ]


def _attribute_value(opening_tag, attribute):
    match = re.search(
        rf"(?<![\w:-]){re.escape(attribute)}\s*=\s*"
        r"(?:\"([^\"]*)\"|'([^']*)')",
        opening_tag,
    )
    if match is None:
        return None
    return next(value for value in match.groups() if value is not None)


class VisualContentTest(unittest.TestCase):
    def test_writing_guides_define_the_visual_contract(self):
        global_style = WRITING_STYLE.read_text(encoding="utf-8")
        part2_style = PART2_STYLE.read_text(encoding="utf-8")
        workflow = WRITING_WORKFLOW.read_text(encoding="utf-8")
        combined = "\n".join((global_style, part2_style, workflow))

        self.assertNotRegex(combined, r"(?i)mermaid")
        for phrase in ("HTML/SVG", "学习目标", "文字替代", "减少动态效果"):
            self.assertIn(phrase, global_style)
        for token in (
            'data-lesson-visual',
            'data-lesson-stage',
            'role="img"',
            'data-step',
            'data-lesson-steps',
            'figcaption',
            'lesson-visual__canvas--wide',
        ):
            self.assertIn(token, global_style)
        for phrase in (
            "作者",
            "原始来源",
            "资产授权页",
            "许可证链接",
            "修改说明",
            "NC",
            "SA",
            "分发方式兼容",
        ):
            self.assertIn(phrase, global_style)

        self.assertRegex(
            part2_style,
            r"具体问题.*可见变化.*机制命名.*必要公式.*边界",
        )
        self.assertIn('data-step-mode="single"', part2_style)
        self.assertIn("图注要告诉读者看什么", global_style)
        self.assertIn("不直接复制", global_style)

    def test_convolution_pilot(self):
        content = CONVOLUTION_PAGE.read_text(encoding="utf-8")

        self.assertNotIn("```mermaid", content)
        figures = re.findall(
            r"<figure\b[^>]*data-lesson-visual\b.*?</figure>",
            content,
            re.DOTALL,
        )
        self.assertEqual(len(figures), 1)
        figure = figures[0]
        figure_tags = _opening_tags_with_attribute(figure, "data-lesson-visual")
        self.assertEqual(len(figure_tags), 1)
        self.assertEqual(_attribute_value(figure_tags[0], "data-step-mode"), "single")

        stage_tags = _opening_tags_with_attribute(figure, "data-lesson-stage")
        self.assertEqual(len(stage_tags), 1)
        self.assertEqual(_attribute_value(stage_tags[0], "role"), "img")
        self.assertRegex(_attribute_value(stage_tags[0], "aria-label") or "", r"卷积|滑动")

        svg_tags = re.findall(r"<svg\b[^>]*>", figure, re.DOTALL)
        self.assertEqual(len(svg_tags), 1)
        canvas_classes = (_attribute_value(svg_tags[0], "class") or "").split()
        self.assertIn("lesson-visual__canvas--wide", canvas_classes)
        window_labels = re.findall(
            r"<text\b[^>]*class=(?:\"convolution-window-label\"|'convolution-window-label')[^>]*>",
            figure,
            re.DOTALL,
        )
        self.assertEqual(len(window_labels), 3)
        for label_tag in window_labels:
            self.assertEqual(_attribute_value(label_tag, "fill"), "currentColor")

        fallback_matches = re.findall(
            r"<ol\b[^>]*data-lesson-steps\b.*?</ol>",
            figure,
            re.DOTALL,
        )
        self.assertEqual(len(fallback_matches), 1)
        steps = [
            re.sub(r"\s+", " ", step).strip()
            for step in re.findall(
                r"<li\b[^>]*>(.*?)</li>",
                fallback_matches[0],
                re.DOTALL,
            )
        ]
        self.assertEqual(len(steps), 3)
        for step, expected in zip(
            steps,
            (
                r"x0.*x1.*x2.*y0.*(?:同一|共享)",
                r"x1.*x2.*x3.*y1.*(?:同一|共享)",
                r"x2.*x3.*x4.*y2.*(?:同一|共享)",
            ),
            strict=True,
        ):
            self.assertRegex(step, expected)

        step_groups = _opening_tags_with_attribute(figure, "data-step")
        self.assertEqual(len(step_groups), 3)
        captions = re.findall(
            r"<figcaption\b[^>]*>(.*?)</figcaption>",
            figure,
            re.DOTALL,
        )
        self.assertEqual(len(captions), 1)
        self.assertRegex(captions[0], r"窗口.*移动.*(?:参数不变|同一组)")

    def test_computation_graph_pilot(self):
        content = COMPUTATIONAL_GRAPHS_PAGE.read_text(encoding="utf-8")

        self.assertNotIn("```mermaid", content)
        figures = re.findall(
            r"<figure\b[^>]*data-lesson-visual\b.*?</figure>",
            content,
            re.DOTALL,
        )
        self.assertEqual(len(figures), 2)

        labels = []
        captions = []
        fallback_steps = []
        for figure in figures:
            stage_tags = _opening_tags_with_attribute(figure, "data-lesson-stage")
            self.assertEqual(len(stage_tags), 1)
            stage_tag = stage_tags[0]
            self.assertEqual(_attribute_value(stage_tag, "role"), "img")
            label = _attribute_value(stage_tag, "aria-label")
            self.assertIsNotNone(label)
            labels.append(label)

            svg_tags = re.findall(r"<svg\b[^>]*>", figure, re.DOTALL)
            self.assertEqual(len(svg_tags), 1)
            canvas_classes = (_attribute_value(svg_tags[0], "class") or "").split()
            self.assertIn("lesson-visual__canvas--wide", canvas_classes)

            figure_captions = re.findall(
                r"<figcaption\b[^>]*>(.*?)</figcaption>",
                figure,
                re.DOTALL,
            )
            self.assertEqual(len(figure_captions), 1)
            captions.append(figure_captions[0])

            fallback_matches = re.findall(
                r"<ol\b[^>]*data-lesson-steps\b.*?</ol>",
                figure,
                re.DOTALL,
            )
            self.assertEqual(len(fallback_matches), 1)
            fallback_steps.append(
                [
                    re.sub(r"\s+", " ", step).strip()
                    for step in re.findall(
                        r"<li\b[^>]*>(.*?)</li>",
                        fallback_matches[0],
                        re.DOTALL,
                    )
                ]
            )

        self.assertEqual(len(set(labels)), 2)
        self.assertRegex(captions[0], r"前向.*梯度")
        self.assertRegex(captions[1], r"(?:Σ.*累加|累加.*Σ)")

        visual_css = VISUAL_CSS.read_text(encoding="utf-8")
        self.assertRegex(
            visual_css,
            r"(?s)\.lesson-visual__canvas--wide\s*\{[^}]*"
            r"min-width:\s*44rem\s*;",
        )

        self.assertEqual(len(fallback_steps[0]), 5)
        for step, expected in zip(
            fallback_steps[0],
            (
                r"输入.*x.*w.*b",
                r"a\s*=\s*6",
                r"z\s*=\s*7",
                r"L\s*=\s*49",
                r"(?:∂L\s*/\s*∂w|dL\s*/\s*dw)\s*=\s*28",
            ),
            strict=True,
        ):
            self.assertRegex(step, expected)

        self.assertEqual(len(fallback_steps[1]), 3)
        for step, expected in zip(
            fallback_steps[1],
            (r"2u", r"3", r"(?:同一|同个).*累加.*点"),
            strict=True,
        ):
            self.assertRegex(step, expected)

    def test_lesson_stage_attributes_are_order_independent(self):
        markup = (
            '<div data-lesson-stage-extra aria-label="wrong"></div>'
            '<section aria-label="stage label" role="img" data-lesson-stage></section>'
        )

        stage_tags = _opening_tags_with_attribute(markup, "data-lesson-stage")

        self.assertEqual(len(stage_tags), 1)
        self.assertEqual(_attribute_value(stage_tags[0], "aria-label"), "stage label")
        self.assertEqual(_attribute_value(stage_tags[0], "role"), "img")

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
                stage_tags = _opening_tags_with_attribute(
                    figure,
                    "data-lesson-stage",
                )
                if not any(
                    _attribute_value(tag, "aria-label") for tag in stage_tags
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
