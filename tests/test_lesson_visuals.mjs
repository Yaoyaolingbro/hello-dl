import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const {
  clampStep,
  createPlayer,
  cumulativeVisibility,
  nextStep,
  prunePlayers,
  previousStep,
  stepVisibility,
} = require("../docs/assets/js/lesson-visuals.js");

test("createPlayer inserts controls directly after the lesson stage", () => {
  const originalDocument = globalThis.document;
  const stage = {
    id: "",
    setAttribute() {},
    insertAdjacentElement(position, element) {
      assert.equal(position, "afterend");
      const stageIndex = root.children.indexOf(this);
      root.children.splice(stageIndex + 1, 0, element);
    },
  };
  const fallback = { name: "fallback" };
  const caption = { name: "caption" };
  const step = {
    dataset: { stepLabel: "第一步" },
    setAttribute() {},
  };
  const root = {
    children: [stage, fallback, caption],
    dataset: {},
    querySelectorAll() { return [step]; },
    querySelector() { return stage; },
    append(element) { this.children.push(element); },
  };

  globalThis.document = {
    createElement() {
      return {
        children: [],
        append(...elements) { this.children.push(...elements); },
        addEventListener() {},
        setAttribute() {},
      };
    },
  };

  try {
    const player = createPlayer(root);
    assert.ok(player);
    assert.match(root.children[1].className, /lesson-visual__controls/);
    assert.equal(root.children[2], fallback);
    assert.equal(root.children[3], caption);
  } finally {
    globalThis.document = originalDocument;
  }
});

test("clampStep keeps a step inside the available range", () => {
  assert.equal(clampStep(-2, 4), 0);
  assert.equal(clampStep(2, 4), 2);
  assert.equal(clampStep(8, 4), 3);
  assert.equal(clampStep(2, 0), 0);
});

test("nextStep advances one step", () => {
  assert.equal(nextStep(1, 4), 2);
});

test("previousStep moves back one step without underflow", () => {
  assert.equal(previousStep(2, 4), 1);
  assert.equal(previousStep(0, 4), 0);
});

test("cumulativeVisibility reveals the current and earlier steps", () => {
  assert.deepEqual(cumulativeVisibility(2, 4), [true, true, true, false]);
});

test("stepVisibility can show only the current frame", () => {
  assert.deepEqual(stepVisibility(2, 4, "single"), [false, false, true, false]);
  assert.deepEqual(stepVisibility(2, 4, "cumulative"), [true, true, true, false]);
  assert.deepEqual(stepVisibility(2, 4), [true, true, true, false]);
});

test("nextStep stops at the final step unless looping is enabled", () => {
  assert.equal(nextStep(3, 4), 3);
  assert.equal(nextStep(3, 4, true), 0);
});

test("prunePlayers pauses and removes disconnected players", () => {
  let connectedPauses = 0;
  let disconnectedPauses = 0;
  const connected = {
    root: { isConnected: true },
    pause() {
      connectedPauses += 1;
    },
  };
  const disconnected = {
    root: { isConnected: false },
    pause() {
      disconnectedPauses += 1;
    },
  };
  const players = new Set([connected, disconnected]);

  prunePlayers(players);

  assert.deepEqual([...players], [connected]);
  assert.equal(connectedPauses, 0);
  assert.equal(disconnectedPauses, 1);
});
