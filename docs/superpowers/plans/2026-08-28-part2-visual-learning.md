# Part 2 Visual Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Mermaid in Part 2 with accessible HTML/SVG teaching figures, revise the writing rules around visual explanation, and refresh completed chapters before architecture writing resumes.

**Architecture:** A small vanilla-JavaScript controller turns inline SVG step groups into a reusable player. Markdown owns the figure content and textual fallback; CSS owns layout, dark mode, responsive behavior, and reduced-motion behavior. Static figures use the same visual language without unnecessary controls.

**Tech Stack:** MkDocs Material, Markdown with `md_in_html`, inline SVG, vanilla JavaScript, CSS, Python `unittest`, Node built-in test runner.

---

### Task 1: Visual player contract

**Files:**
- Create: `tests/test_lesson_visuals.mjs`
- Create: `tests/test_visual_content.py`
- Create: `docs/assets/js/lesson-visuals.js`
- Create: `docs/assets/css/lesson-visuals.css`
- Modify: `mkdocs.yml`

- [ ] **Step 1: Write failing JavaScript tests**

Test `clampStep`, `nextStep`, `previousStep`, cumulative visibility, and non-looping end behavior by importing the public controller API from `lesson-visuals.js`.

- [ ] **Step 2: Verify the JavaScript test is red**

Run: `node --test tests/test_lesson_visuals.mjs`

Expected: FAIL because `docs/assets/js/lesson-visuals.js` does not exist.

- [ ] **Step 3: Write failing integration tests**

Assert that MkDocs loads `lesson-visuals.css` and `lesson-visuals.js`, does not load Mermaid, Part 2 contains no Mermaid fences, and each interactive figure includes an accessible stage label plus a textual step list.

- [ ] **Step 4: Verify the integration test is red**

Run: `python -m unittest tests.test_visual_content -v`

Expected: FAIL because the assets are not registered and Mermaid is still configured.

- [ ] **Step 5: Implement the minimal controller and styles**

The controller must initialize every `[data-lesson-visual]`, add previous/play/next controls, update `aria-live` status text, stop at the final step unless `data-loop="true"`, pause when hidden, and disable autoplay when `prefers-reduced-motion: reduce` matches. It must expose pure navigation helpers through `module.exports` for Node tests and `window.LessonVisuals` in the browser.

- [ ] **Step 6: Register the assets**

Add `assets/css/lesson-visuals.css` and `assets/js/lesson-visuals.js` to `mkdocs.yml`. Keep Mermaid temporarily until every existing fence is migrated; the integration test remains red on that one requirement during the migration.

- [ ] **Step 7: Verify the controller tests are green**

Run: `node --test tests/test_lesson_visuals.mjs`

Expected: all controller tests pass.

- [ ] **Step 8: Commit**

Commit message: `feat(docs): add accessible lesson visual player`

### Task 2: Computation graph pilot

**Files:**
- Modify: `docs/02-deep-learning/02-neural-network-foundations/computational-graphs.md`
- Test: `tests/test_visual_content.py`

- [ ] **Step 1: Add a failing page contract**

Require two figures with unique labels: a cumulative forward/backward graph and a branching-gradient accumulation figure. Require the scalar values `a=6`, `z=7`, `L=49`, and gradient `28` in the text or accessible fallback.

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_visual_content.VisualContentTest.test_computation_graph_pilot -v`

Expected: FAIL because the page still contains Mermaid.

- [ ] **Step 3: Replace both Mermaid blocks**

Use inline SVG groups marked `data-step`, with a nearby ordered list that explains what changes at each step. Keep equations in KaTeX and retain the existing raster overview only as a conceptual opening image.

- [ ] **Step 4: Humanize the surrounding explanation**

Open with the concrete scalar calculation, make captions tell the reader what to watch, and remove repetitive definitions or slogan-like conclusions.

- [ ] **Step 5: Verify green and build**

Run: `python -m unittest tests.test_visual_content.VisualContentTest.test_computation_graph_pilot -v`

Run: `python -m mkdocs build --strict --site-dir /private/tmp/deep-learning-computation-graph`

- [ ] **Step 6: Commit**

Commit message: `docs(part2): add interactive computation graph walkthrough`

### Task 3: Convolution pilot

**Files:**
- Modify: `docs/02-deep-learning/04-components/convolution.md`
- Test: `tests/test_visual_content.py`

- [ ] **Step 1: Add a failing page contract**

Require an interactive sliding-window figure with three positions, shared kernel labeling, output cells `y0` through `y2`, and a textual fallback.

- [ ] **Step 2: Verify red**

Run: `python -m unittest tests.test_visual_content.VisualContentTest.test_convolution_pilot -v`

Expected: FAIL because the page still contains Mermaid.

- [ ] **Step 3: Implement the SVG sliding window**

Use one input row, a moving three-cell highlight, one fixed kernel legend, and progressive output cells. The control caption must explain that the same weights move while the input window changes.

- [ ] **Step 4: Tighten the prose**

Connect the animation directly to parameter sharing and output-size arithmetic. Keep the existing precise formula and PyTorch check.

- [ ] **Step 5: Verify green and build**

Run the pilot test and a strict MkDocs build.

- [ ] **Step 6: Commit**

Commit message: `docs(part2): visualize the convolution sliding window`

### Task 4: Writing rules and author workflow

**Files:**
- Modify: `writing-style.md`
- Modify: `docs/superpowers/style-part2-dl-basics.md`
- Modify: `chapter-writing-workflow.md`
- Modify: `reference.md`

- [ ] **Step 1: Add a failing style contract**

Require the rules to ban Mermaid, define when to use interactive SVG versus static images, require a learning goal and textual fallback for every animation, require source and license records for external images, and describe the sequence “concrete problem → visible change → mechanism name → necessary formula → boundary”.

- [ ] **Step 2: Verify red**

Run the style-contract unittest and confirm it fails on the old Mermaid guidance.

- [ ] **Step 3: Rewrite the rules**

Record the Hello Algo research as an explanation reference, but state that visual assets and code are original unless an explicit compatible license is recorded. Replace absolute image quotas with task-based visual selection.

- [ ] **Step 4: Run the Chinese humanizer pass**

Remove canned transitions, excessive bold, forced triads, vague attribution, and promotional language. Score the revised prose on directness, rhythm, trust, authenticity, and concision; revise any section below 45/50.

- [ ] **Step 5: Verify green**

Run the style-contract unittest.

- [ ] **Step 6: Commit**

Commit message: `docs(style): adopt visual-first lesson guidance`

### Task 5: Migrate all remaining Part 2 Mermaid figures

**Files:**
- Modify: `docs/02-deep-learning/03-training/backpropagation.md`
- Modify: `docs/02-deep-learning/03-training/minibatch-and-training-loop.md`
- Modify: `docs/02-deep-learning/03-training/initialization.md`
- Modify: `docs/02-deep-learning/03-training/normalization.md`
- Modify: `docs/02-deep-learning/04-components/residual-connections.md`
- Modify: `docs/02-deep-learning/04-components/recurrence.md`
- Modify: `docs/02-deep-learning/04-components/attention.md`
- Modify: `docs/02-deep-learning/04-components/graph-message-passing.md`
- Modify: `mkdocs.yml`
- Modify: `docs/assets/css/custom.css`

- [ ] **Step 1: Keep the no-Mermaid contract red**

Run: `python -m unittest tests.test_visual_content.VisualContentTest.test_part2_has_no_mermaid -v`

Expected: FAIL and list every remaining file.

- [ ] **Step 2: Convert each figure according to its teaching job**

Use interactive steps for state changes and flow; use a static inline SVG for stable topology or axis comparison. Every figure receives an informative caption and accessible text equivalent.

- [ ] **Step 3: Remove Mermaid configuration**

Delete the Mermaid custom fence, CDN script, and Mermaid-only CSS after the final fence is gone.

- [ ] **Step 4: Verify green**

Run: `python -m unittest tests.test_visual_content -v`

Run: `rg -n 'mermaid' docs/02-deep-learning mkdocs.yml docs/assets`

Expected: tests pass; search has no results.

- [ ] **Step 5: Commit**

Commit message: `refactor(part2): replace Mermaid with teaching figures`

### Task 6: Editorial audit of completed Part 2 sections 01–04

**Files:**
- Modify: only pages under `docs/02-deep-learning/01-machine-learning-basics/` through `04-components/` that contain a concrete editorial issue
- Modify: `reference.md` only when a new source is used

- [ ] **Step 1: Audit each page**

Check openings, paragraph rhythm, formula introductions, captions, repeated conclusions, unsupported claims, and whether every visual has a teaching job. Do not rewrite already-natural passages merely to make a diff.

- [ ] **Step 2: Apply targeted edits**

Use concrete examples before definitions where the current opening is abstract. Preserve verified derivations and existing links.

- [ ] **Step 3: Run content and humanizer checks**

Run: `python scripts/check_part2_content.py`

Run targeted searches for banned filler terms and inspect every match in context.

- [ ] **Step 4: Commit**

Commit message: `docs(part2): refine completed foundations chapters`

### Task 7: Browser QA and handoff to architectures

**Files:**
- Modify: visual assets or chapter markup only for defects found during QA

- [ ] **Step 1: Run complete automated verification**

Run Node tests, Python unit tests, the 46-page content checker, strict MkDocs build, and `git diff --check`.

- [ ] **Step 2: Inspect representative pages in a browser**

Check computation graph, convolution, one static figure, and one training-flow figure at desktop and mobile widths in light and dark themes. Exercise previous, play/pause, next, final-step stopping, keyboard focus, and reduced-motion behavior.

- [ ] **Step 3: Fix defects through red-green tests**

Add a regression test before any behavior fix; rerun the relevant test and browser check.

- [ ] **Step 4: Final review and preview deployment**

Confirm no Mermaid remains, publish the current branch preview, and provide the URL. Only after this checkpoint may Part 2 section 05 resume under the new rules.

