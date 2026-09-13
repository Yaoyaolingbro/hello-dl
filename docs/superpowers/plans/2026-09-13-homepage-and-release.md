# Homepage and Current Chapters Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tidy, coordinated MkDocs homepage, publish the completed Part 2 units, merge the rewrite branch into `main`, and verify `https://yaoyaolingbro.github.io/hello-dl/`.

**Architecture:** Keep MkDocs Material as the document system. The homepage is semantic HTML inside `docs/index.md`, styled by dedicated `home-*` rules and shared theme tokens in `custom.css`; one original raster illustration is decorative, while all meaningful content remains HTML. GitHub Actions validates both completed and explicitly planned Part 2 pages, performs a strict build, and deploys `main` to `gh-pages`.

**Tech Stack:** MkDocs Material, Markdown with `md_in_html`, CSS Grid, CSS custom properties, SVG/CSS motion, Python `unittest`, GitHub Actions, GitHub Pages.

---

## File map

- `docs/index.md`: semantic homepage content and links only.
- `docs/assets/images/home/learning-path-hero.png`: original illustration without text or logos.
- `docs/assets/css/custom.css`: global theme tokens plus homepage layout, responsive rules, dark mode, and reduced-motion behavior.
- `mkdocs.yml`: canonical site URL and Material custom palette selection.
- `tests/test_homepage.py`: source-level contract for homepage structure, asset, colors, responsiveness, and URL.
- `.github/workflows/deploy.yml`: partial-release-safe validation, strict build, and Pages deployment.

### Task 1: Lock the homepage contract with a failing test

**Files:**
- Create: `tests/test_homepage.py`

- [ ] **Step 1: Write the failing source contract**

```python
import re
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
```

- [ ] **Step 2: Run the new test and confirm it fails**

Run: `python -m unittest tests.test_homepage -v`

Expected: failures for missing `home-hero`, missing image, custom palette, and the old `/deep-learning` site URL.

- [ ] **Step 3: Commit the test**

```bash
git add tests/test_homepage.py
git commit -m "test(site): define homepage visual contract"
```

### Task 2: Generate the original hero illustration

**Files:**
- Create: `docs/assets/images/home/learning-path-hero.png`

- [ ] **Step 1: Generate one landscape illustration**

Use the image generation tool with this prompt:

```text
Create a polished landscape editorial illustration for the homepage of a Chinese deep-learning textbook. Warm off-white background, disciplined modular grid, layered cards suggesting tensors and model blocks, a clear path flowing from simple mathematical shapes on the lower-left toward a refined constellation of modern AI research nodes on the upper-right. Friendly and inviting first, subtly futuristic second. Deep indigo, soft teal, pale lavender, and one restrained warm orange accent. Clean geometric forms, even spacing, quiet depth, soft paper texture, no people, no robots, no brains, no text, no letters, no equations, no logos, no watermark. Compose all important objects inside safe margins. 4:3 landscape.
```

- [ ] **Step 2: Place the generated image at the exact asset path**

Save the selected output as `docs/assets/images/home/learning-path-hero.png`. Keep the original 4:3 composition; do not bake any homepage text into it.

- [ ] **Step 3: Inspect the image**

Open the local image and confirm: no accidental text, no watermark, no clipped elements, clear grid, and coordinated indigo/teal/orange colors.

- [ ] **Step 4: Commit the asset**

```bash
git add docs/assets/images/home/learning-path-hero.png
git commit -m "feat(site): add original learning path hero"
```

### Task 3: Build the semantic homepage and coordinated theme

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/assets/css/custom.css`
- Modify: `mkdocs.yml`
- Test: `tests/test_homepage.py`

- [ ] **Step 1: Replace the current homepage with semantic sections**

`docs/index.md` must use this structure and keep all meaningful text outside the image:

```html
<div class="home-page">
  <section class="home-hero" aria-labelledby="home-title">
    <div class="home-hero__copy">
      <p class="home-kicker">一本从基础走向前沿的深度学习笔记</p>
      <h1 id="home-title">从数学直觉出发，<br>把深度学习真正学明白</h1>
      <p class="home-lead">一套面向初学者的开放教材。沿着主线掌握基础，在需要时进入面试拆解与科研速览。</p>
      <div class="home-actions">
        <a class="home-button home-button--primary" href="01-math/">开始学习</a>
        <a class="home-button home-button--secondary" href="#learning-path">查看学习路线</a>
      </div>
    </div>
    <figure class="home-hero__visual">
      <img src="assets/images/home/learning-path-hero.png" alt="从数学基础逐步连接到现代人工智能研究方向的抽象学习地图">
    </figure>
  </section>
</div>
```

After the hero, add one quiet three-column route band for `主线教材`、`面试拆解`、`科研速览`, separated by rules rather than three floating SaaS-style cards. Then add a `#learning-path` section with four equal-height sequential cards linking to `01-math/` through `04-applications/`. Each Part card contains its title, one-sentence role, reader cue, and a text link. Use numbering only for these four cards because they form a real learning sequence. Avoid decorative uppercase English labels and avoid appending arrows to every link.

- [ ] **Step 2: Add shared theme tokens and grid rules**

Add the following root tokens and layout contract to `docs/assets/css/custom.css`, followed by focused rules for buttons, route cards, Part cards, dark mode, hover/focus, and image containment:

```css
:root {
  --dl-paper: #fbfaf6;
  --dl-surface: #ffffff;
  --dl-ink: #20263a;
  --dl-muted: #667085;
  --dl-indigo: #5661c9;
  --dl-indigo-soft: #e8ebff;
  --dl-teal: #34877d;
  --dl-teal-soft: #dcf3ee;
  --dl-orange: #bd6e2a;
  --dl-orange-soft: #ffeedb;
  --md-primary-fg-color: var(--dl-indigo);
  --md-accent-fg-color: var(--dl-teal);
}

.home-hero {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(18rem, 0.8fr);
  gap: clamp(2rem, 5vw, 5rem);
  align-items: center;
}

@media (max-width: 800px) {
  .home-hero { grid-template-columns: 1fr; }
}

@media (prefers-reduced-motion: reduce) {
  .home-page *, .home-page *::before, .home-page *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

Use the image as a contained 4:3 object inside `.home-hero__visual`; never position it absolutely over the heading. Set the route band to three columns and the Part sequence to four columns, with two-column and one-column breakpoints. Keep the illustration as the one expressive element; surrounding surfaces should be quiet, with restrained corner radii and little or no generic drop shadow.

- [ ] **Step 3: Select Material custom colors and correct the canonical URL**

Change both palette entries in `mkdocs.yml` from named primary/accent colors to:

```yaml
primary: custom
accent: custom
```

Set:

```yaml
site_url: https://yaoyaolingbro.github.io/hello-dl/
```

Also use `Noto Sans SC` for Chinese body and display text and `JetBrains Mono` for code, matching the long-form Chinese technical reading context.

- [ ] **Step 4: Run the homepage test**

Run: `python -m unittest tests.test_homepage -v`

Expected: 3 tests pass.

- [ ] **Step 5: Run a strict build**

Run: `mkdocs build --strict`

Expected: build completes with exit code 0.

- [ ] **Step 6: Commit the homepage and theme**

```bash
git add docs/index.md docs/assets/css/custom.css mkdocs.yml tests/test_homepage.py
git commit -m "feat(site): redesign the learning homepage"
```

### Task 4: Make deployment accept an honest partial release

**Files:**
- Modify: `.github/workflows/deploy.yml`
- Test: `tests/test_homepage.py`

- [ ] **Step 1: Extend the test with workflow assertions**

Add:

```python
    def test_deploy_validates_navigation_and_builds_strictly(self):
        workflow = (ROOT / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        self.assertIn("--mkdocs-config mkdocs.yml", workflow)
        self.assertNotIn("--require-complete", workflow)
        self.assertIn("mkdocs build --strict", workflow)
        self.assertIn("mkdocs gh-deploy --force", workflow)
```

- [ ] **Step 2: Run the workflow test and confirm it fails**

Run: `python -m unittest tests.test_homepage.HomepageTest.test_deploy_validates_navigation_and_builds_strictly -v`

Expected: failure because the workflow still requires all planned pages to be complete and has no explicit strict-build step.

- [ ] **Step 3: Update the workflow**

Use these steps after dependency installation:

```yaml
      - name: Validate Part 2 content and navigation
        run: python scripts/check_part2_content.py --root docs/02-deep-learning --mkdocs-config mkdocs.yml

      - name: Build strictly
        run: mkdocs build --strict

      - name: Deploy
        run: mkdocs gh-deploy --force
```

- [ ] **Step 4: Run the workflow contract and local validation**

Run: `python -m unittest tests.test_homepage -v`

Expected: 4 tests pass.

Run: `python scripts/check_part2_content.py --root docs/02-deep-learning --mkdocs-config mkdocs.yml`

Expected: `Validated 46 Part 2 pages`.

- [ ] **Step 5: Commit the release workflow**

```bash
git add .github/workflows/deploy.yml tests/test_homepage.py
git commit -m "ci: deploy completed chapters with strict validation"
```

### Task 5: Run the full test and browser QA matrix

**Files:**
- Modify only if a verified defect is found in the files from Tasks 2—4.

- [ ] **Step 1: Run all automated checks**

Run: `python -m unittest discover -s tests -p 'test_*.py' -v`

Expected: all Python tests pass.

Run: `node --test tests/test_lesson_visuals.mjs`

Expected: all lesson visual tests pass.

Run: `python scripts/check_part2_content.py --root docs/02-deep-learning --mkdocs-config mkdocs.yml`

Expected: `Validated 46 Part 2 pages`.

Run: `mkdocs build --strict`

Expected: exit code 0.

- [ ] **Step 2: Start the local site and check desktop light mode**

Run: `mkdocs serve -a 127.0.0.1:8000`

Check: hero text remains inside the left grid column; the illustration remains inside its frame; route cards and Part cards align; no horizontal overflow; navigation, formulas, and lesson controls work.

- [ ] **Step 3: Check mobile and dark mode**

At a 390×844 viewport, confirm a single-column hero and card layout, readable buttons, and no clipping. Toggle dark mode and confirm contrast for text, cards, buttons, code, formulas, and the hero image frame.

- [ ] **Step 4: Check reduced motion**

Emulate `prefers-reduced-motion: reduce`; confirm homepage decorative motion stops and existing lesson controls remain usable.

- [ ] **Step 5: Commit any QA fixes**

```bash
git add docs/index.md docs/assets/css/custom.css mkdocs.yml
git commit -m "fix(site): polish responsive homepage rendering"
```

Skip this commit only when QA produces no source changes.

### Task 6: Merge, push, deploy, and verify production

**Files:**
- No planned source changes.

- [ ] **Step 1: Confirm both worktrees are clean**

Run: `git status --short --branch` in the rewrite worktree and `git -C '/Users/yaoyaoling/Desktop/博士生资料/deep learning' status --short --branch` in the main worktree.

Expected: only ignored or known local preview directories are untracked in the rewrite worktree; the main worktree has no tracked modifications.

- [ ] **Step 2: Merge the rewrite branch into main**

Run in the main worktree:

```bash
git merge --no-ff rewrite/part2-mainline -m "merge: publish Part 2 foundations and visual homepage"
```

Expected: a merge commit with no conflicts.

- [ ] **Step 3: Run the full checks on merged main**

Repeat the four automated commands from Task 5 in the main worktree.

Expected: all checks pass from the exact commit that will be published.

- [ ] **Step 4: Push main**

Run: `git push origin main`

Expected: GitHub accepts the update. This push includes the six existing local main commits plus the reviewed rewrite merge.

- [ ] **Step 5: Wait for GitHub Actions deployment**

Run: `gh run list --workflow deploy.yml --branch main --limit 1`, then `gh run watch <run-id> --exit-status`.

Expected: the deployment workflow concludes successfully.

- [ ] **Step 6: Verify production**

Open `https://yaoyaolingbro.github.io/hello-dl/` and confirm HTTP success, correct canonical subpath, homepage layout, image, navigation, formula rendering, dark-mode toggle, and at least one computation-graph lesson interaction.

- [ ] **Step 7: Record the deployed commit**

Run: `git rev-parse --short HEAD` and report the commit together with the live URL.
