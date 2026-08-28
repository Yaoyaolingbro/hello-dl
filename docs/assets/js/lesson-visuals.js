(function (global) {
  "use strict";

  function clampStep(step, stepCount) {
    if (stepCount <= 0) return 0;
    return Math.min(Math.max(Number(step) || 0, 0), stepCount - 1);
  }

  function nextStep(step, stepCount, loop) {
    if (stepCount <= 0) return 0;
    const current = clampStep(step, stepCount);
    if (current === stepCount - 1) return loop ? 0 : current;
    return current + 1;
  }

  function previousStep(step, stepCount) {
    return clampStep(clampStep(step, stepCount) - 1, stepCount);
  }

  function cumulativeVisibility(step, stepCount) {
    const current = clampStep(step, stepCount);
    return Array.from({ length: Math.max(0, stepCount) }, (_, index) => index <= current);
  }

  function stepVisibility(step, stepCount, mode) {
    if (mode !== "single") return cumulativeVisibility(step, stepCount);
    const current = clampStep(step, stepCount);
    return Array.from(
      { length: Math.max(0, stepCount) },
      (_, index) => index === current,
    );
  }

  function prunePlayers(playerSet) {
    playerSet.forEach((player) => {
      if (!player.root.isConnected) {
        player.pause();
        playerSet.delete(player);
      }
    });
  }

  function createButton(label, className) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.textContent = label;
    return button;
  }

  function createPlayer(root) {
    const steps = Array.from(root.querySelectorAll("[data-step]"));
    if (steps.length === 0 || root.dataset.lessonVisualReady === "true") return null;

    root.dataset.lessonVisualReady = "true";
    const loop = root.dataset.loop === "true";
    const interval = Math.max(250, Number(root.dataset.interval) || 1600);
    const stage = root.querySelector("[data-lesson-stage]");
    let current = 0;
    let timer = null;

    const controls = document.createElement("div");
    controls.className = "lesson-visual__controls";

    const previousButton = createButton("上一步", "lesson-visual__button lesson-visual__previous");
    const playButton = createButton("播放", "lesson-visual__button lesson-visual__play");
    const nextButton = createButton("下一步", "lesson-visual__button lesson-visual__next");
    const status = document.createElement("span");
    status.className = "lesson-visual__status";
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    status.setAttribute("aria-atomic", "true");

    if (stage) {
      if (!stage.id) {
        stage.id = `lesson-visual-stage-${createPlayer.nextId++}`;
      }
      for (const button of [previousButton, playButton, nextButton]) {
        button.setAttribute("aria-controls", stage.id);
      }
    }

    controls.append(previousButton, playButton, nextButton, status);
    root.append(controls);

    function render() {
      const visible = stepVisibility(current, steps.length, root.dataset.stepMode);
      steps.forEach((step, index) => {
        step.hidden = !visible[index];
        step.setAttribute("aria-hidden", String(!visible[index]));
      });

      const label = steps[current].dataset.stepLabel;
      status.textContent = `步骤 ${current + 1} / ${steps.length}${label ? `：${label}` : ""}`;
      previousButton.disabled = current === 0;
      nextButton.disabled = !loop && current === steps.length - 1;
    }

    function pause() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
      playButton.textContent = "播放";
      playButton.setAttribute("aria-pressed", "false");
    }

    function setStep(step) {
      current = clampStep(step, steps.length);
      render();
    }

    function advance() {
      const following = nextStep(current, steps.length, loop);
      setStep(following);
      if (!loop && current === steps.length - 1) pause();
    }

    function play() {
      if (timer !== null) return;
      if (!loop && current === steps.length - 1) setStep(0);
      playButton.textContent = "暂停";
      playButton.setAttribute("aria-pressed", "true");
      timer = setInterval(advance, interval);
    }

    previousButton.addEventListener("click", () => {
      pause();
      setStep(previousStep(current, steps.length));
    });
    playButton.addEventListener("click", () => {
      if (timer === null) play();
      else pause();
    });
    nextButton.addEventListener("click", () => {
      pause();
      setStep(nextStep(current, steps.length, loop));
    });

    playButton.setAttribute("aria-pressed", "false");
    render();

    const reducedMotion = global.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (root.dataset.autoplay === "true" && !reducedMotion) play();

    return { root, pause, play, setStep };
  }

  createPlayer.nextId = 1;
  const players = new Set();
  let visibilityListenerAdded = false;

  function initLessonVisuals(scope) {
    prunePlayers(players);
    const container = scope || document;
    container.querySelectorAll("[data-lesson-visual]").forEach((root) => {
      const player = createPlayer(root);
      if (player) players.add(player);
    });

    if (!visibilityListenerAdded) {
      document.addEventListener("visibilitychange", () => {
        prunePlayers(players);
        if (!document.hidden) return;
        players.forEach((player) => player.pause());
      });
      visibilityListenerAdded = true;
    }
  }

  const api = {
    clampStep,
    createPlayer,
    cumulativeVisibility,
    initLessonVisuals,
    nextStep,
    prunePlayers,
    previousStep,
    stepVisibility,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (global) global.LessonVisuals = api;

  if (typeof document !== "undefined") {
    if (typeof document$ !== "undefined") {
      document$.subscribe(({ body }) => initLessonVisuals(body));
    } else if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => initLessonVisuals(document));
    } else {
      initLessonVisuals(document);
    }
  }
})(typeof window !== "undefined" ? window : globalThis);
