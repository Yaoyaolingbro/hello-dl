# Homepage Width Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the homepage inherit the same Material content width as ordinary pages without changing its content or responsive behavior.

**Architecture:** Remove the homepage-only override on `.md-main__inner` and guard the decision with a small source-level regression test. Keep all homepage component CSS intact, then verify the rendered widths and responsive layout in a real browser before merging and deploying.

**Tech Stack:** MkDocs Material, CSS, Python `unittest`, browser layout inspection, GitHub Actions/GitHub Pages.

---

### Task 1: Lock the standard-width contract

**Files:**
- Modify: `tests/test_homepage.py`
- Modify: `docs/assets/css/custom.css:215-218`

- [ ] **Step 1: Write the failing regression test**

Add this method to `HomepageTest`:

```python
def test_homepage_inherits_the_standard_material_content_width(self):
    css = CSS.read_text(encoding="utf-8")
    self.assertNotIn(".md-main__inner:has(.home-page)", css)
    self.assertNotIn("max-width: 76rem", css)
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
python -m unittest discover -s tests -p 'test_homepage.py' -v
```

Expected: the six existing homepage tests pass and the new width test fails because `custom.css` still contains the homepage-only `76rem` rule.

- [ ] **Step 3: Remove only the homepage width override**

Delete this block from `docs/assets/css/custom.css`:

```css
/* Only the landing page gets a wider canvas; article widths stay unchanged. */
.md-main__inner:has(.home-page) {
  max-width: 76rem;
}
```

Keep `.md-content:has(.home-page)`, `.home-page`, `.home-hero`, and every responsive rule unchanged.

- [ ] **Step 4: Run the homepage and full automated checks**

Run:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/test_lesson_visuals.mjs
python scripts/check_part2_content.py --root docs/02-deep-learning --mkdocs-config mkdocs.yml
mkdocs build --strict
git diff --check
```

Expected: 32 Python tests, 8 Node tests, 46 Part 2 pages validated, strict build succeeds, and `git diff --check` reports nothing.

- [ ] **Step 5: Commit the tested width change**

```bash
git add tests/test_homepage.py docs/assets/css/custom.css
git commit -m "fix(site): align homepage with reading width"
```

### Task 2: Verify the rendered layout and publish

**Files:**
- Verify: `docs/index.md`
- Verify: `docs/assets/css/custom.css`
- Verify: `.github/workflows/deploy.yml`

- [ ] **Step 1: Serve the branch and compare actual widths**

Open the homepage and an ordinary page at 1280px. Inspect both `.md-main__inner` elements and confirm their computed width and `max-width` match. Confirm the homepage hero copy and visual do not overlap.

- [ ] **Step 2: Verify responsive behavior**

At 768px, confirm the hero is one column and the Part cards are two columns. At 390px, confirm the hero and Part cards are one column, the forced heading break is hidden, the AVIF is loaded, and `document.documentElement.scrollWidth === innerWidth`.

- [ ] **Step 3: Review screenshots**

Capture desktop and mobile screenshots. Confirm the homepage sections share one left and right boundary, the hero image remains legible, and no unrelated typography, navigation, or color changes appear.

- [ ] **Step 4: Merge and deploy**

After browser verification, merge `fix/homepage-width-alignment` into `main`, rerun the automated checks on the merged result, push `main`, and wait for the `Deploy MkDocs to GitHub Pages` workflow to succeed.

- [ ] **Step 5: Verify production**

Open `https://yaoyaolingbro.github.io/hello-dl/` at desktop and mobile widths. Confirm it serves the new standard-width homepage with status 200, no overlap or page-level overflow, and the same public content as the tested branch.
