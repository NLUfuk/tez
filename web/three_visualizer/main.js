import { HybridVisualizer } from "./visualizer.js";

const API_BASE = "http://127.0.0.1:8765";
const FETCH_MS = 70;
const DRAG_THRESHOLD_PX = 6;

const state = {
  lastStep: -1,
  lastAt: performance.now(),
  hz: 0,
  pointerDown: null,
};

const statEls = {
  connection: document.getElementById("stat-connection"),
  fps: document.getElementById("stat-fps"),
  step: document.getElementById("stat-step"),
  hz: document.getElementById("stat-hz"),
  alive: document.getElementById("stat-alive"),
  spikes: document.getElementById("stat-spikes"),
  edges: document.getElementById("stat-edges"),
  efficiency: document.getElementById("stat-efficiency"),
  birth: document.getElementById("stat-birth"),
  survival: document.getElementById("stat-survival"),
  stability: document.getElementById("stat-stability"),
  information: document.getElementById("stat-information"),
  memory: document.getElementById("stat-memory"),
  cost: document.getElementById("stat-cost"),
};

const inputBirth = document.getElementById("input-birth");
const inputSurvive = document.getElementById("input-survive");
const inputGenerationMs = document.getElementById("input-generation-ms");
const btnApplyRules = document.getElementById("btn-apply-rules");
const btnApplySpeed = document.getElementById("btn-apply-speed");
const cbAutoRotate = document.getElementById("cb-auto-rotate");
const btnResetCamera = document.getElementById("btn-reset-camera");
const selectQuality = document.getElementById("select-quality");
const rngKAlive = document.getElementById("rng-k-alive");
const rngKNeighbors = document.getElementById("rng-k-neighbors");
const rngBias = document.getElementById("rng-bias");
const rngKSyn = document.getElementById("rng-k-syn");
const rngTraceDecay = document.getElementById("rng-trace-decay");
const lblKAlive = document.getElementById("lbl-k-alive");
const lblKNeighbors = document.getElementById("lbl-k-neighbors");
const lblBias = document.getElementById("lbl-bias");
const lblKSyn = document.getElementById("lbl-k-syn");
const lblTraceDecay = document.getElementById("lbl-trace-decay");
const cbFeedback = document.getElementById("cb-feedback");
const cbGraphFeedbackNbors = document.getElementById("cb-graph-feedback-nbors");
const btnApplyCoupling = document.getElementById("btn-apply-coupling");
const btnResetGrid = document.getElementById("btn-reset-grid");

function setText(el, value) {
  if (el) el.textContent = value;
}

async function postControl(body) {
  await fetch(`${API_BASE}/control`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

function parseRuleList(raw, fallback) {
  const values = String(raw)
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isInteger(n) && n >= 0 && n <= 8);
  return values.length ? values : fallback;
}

function syncCouplingSlidersFromPayload(c) {
  if (!c) return;
  if (rngKAlive) {
    rngKAlive.value = String(Number(c.k_alive));
    if (lblKAlive) lblKAlive.textContent = Number(c.k_alive).toFixed(2);
  }
  if (rngKNeighbors) {
    rngKNeighbors.value = String(Number(c.k_neighbors));
    if (lblKNeighbors) lblKNeighbors.textContent = Number(c.k_neighbors).toFixed(2);
  }
  if (rngBias) {
    rngBias.value = String(Number(c.bias));
    if (lblBias) lblBias.textContent = Number(c.bias).toFixed(2);
  }
  if (cbFeedback) cbFeedback.checked = !!c.feedback_enabled;
  if (typeof c.k_syn === "number" && rngKSyn) {
    rngKSyn.value = String(Number(c.k_syn));
    if (lblKSyn) lblKSyn.textContent = Number(c.k_syn).toFixed(2);
  }
  if (typeof c.spike_trace_decay === "number" && rngTraceDecay) {
    rngTraceDecay.value = String(Number(c.spike_trace_decay));
    if (lblTraceDecay) lblTraceDecay.textContent = Number(c.spike_trace_decay).toFixed(3);
  }
  if (cbGraphFeedbackNbors && typeof c.feedback_graph_neighbors === "boolean") {
    cbGraphFeedbackNbors.checked = !!c.feedback_graph_neighbors;
  }
}

function updateStats(payload, viz) {
  const m = payload.metrics;
  const now = performance.now();
  if (typeof m.step === "number" && m.step !== state.lastStep) {
    const dt = Math.max(0.001, (now - state.lastAt) / 1000);
    state.hz = 1 / dt;
    state.lastAt = now;
    state.lastStep = m.step;
  }
  setText(statEls.connection, "Bagli");
  setText(statEls.fps, viz ? viz.getFps().toFixed(0) : "-");
  setText(statEls.step, String(m.step));
  setText(statEls.hz, state.hz.toFixed(2));
  setText(statEls.alive, String(m.alive_count));
  setText(statEls.spikes, String(m.spike_count));
  const edgeCount = payload.graph?.edges?.length ?? null;
  setText(statEls.edges, edgeCount != null ? String(edgeCount) : String(viz?.graphEdges?.length ?? "-"));
  setText(statEls.efficiency, Number(m.efficiency_score).toFixed(3));
  setText(statEls.stability, Number(m.stability_score).toFixed(3));
  setText(statEls.information, Number(m.information_score).toFixed(3));
  setText(statEls.memory, Number(m.memory_score).toFixed(3));
  setText(statEls.cost, Number(m.cost_score).toFixed(3));
  if (payload.rules) {
    setText(statEls.birth, `B${payload.rules.birth.join("")}`);
    setText(statEls.survival, `S${payload.rules.survive.join("")}`);
  }
  if (typeof payload.generation_period_ms === "number" && inputGenerationMs) {
    inputGenerationMs.value = String(payload.generation_period_ms);
  }
  syncCouplingSlidersFromPayload(payload.coupling);
}

async function bootstrap() {
  const container = document.getElementById("canvas-root");
  const viz = new HybridVisualizer(container);

  const syncRangeLabel = (rng, lbl) => {
    if (!rng || !lbl) return;
    const v = Number(rng.value);
    lbl.textContent = Number.isFinite(v) ? v.toFixed(2) : rng.value;
  };
  syncRangeLabel(rngKAlive, lblKAlive);
  syncRangeLabel(rngKNeighbors, lblKNeighbors);
  syncRangeLabel(rngBias, lblBias);
  syncRangeLabel(rngKSyn, lblKSyn);
  syncRangeLabel(rngTraceDecay, lblTraceDecay);

  if (selectQuality) {
    selectQuality.value = viz.qualityLevel || "medium";
    selectQuality.addEventListener("change", () => {
      viz.setQuality(selectQuality.value);
    });
  }

  cbAutoRotate?.addEventListener("change", () => {
    viz.setAutoRotate(!!cbAutoRotate.checked);
  });

  btnResetCamera?.addEventListener("click", () => {
    viz.resetCameraDefault();
  });

  rngKAlive?.addEventListener("input", () => syncRangeLabel(rngKAlive, lblKAlive));
  rngKNeighbors?.addEventListener("input", () => syncRangeLabel(rngKNeighbors, lblKNeighbors));
  rngBias?.addEventListener("input", () => syncRangeLabel(rngBias, lblBias));
  rngKSyn?.addEventListener("input", () => syncRangeLabel(rngKSyn, lblKSyn));
  rngTraceDecay?.addEventListener("input", () => syncRangeLabel(rngTraceDecay, lblTraceDecay));

  btnApplyCoupling?.addEventListener("click", async () => {
    await postControl({
      action: "set_coupling",
      k_alive: Number(rngKAlive?.value ?? 4),
      k_neighbors: Number(rngKNeighbors?.value ?? 0.5),
      bias: Number(rngBias?.value ?? 0.5),
      feedback_enabled: !!cbFeedback?.checked,
      k_syn: Number(rngKSyn?.value ?? 2.8),
      spike_trace_decay: Number(rngTraceDecay?.value ?? 0.88),
      feedback_graph_neighbors: !!cbGraphFeedbackNbors?.checked,
    });
  });

  btnResetGrid?.addEventListener("click", async () => {
    await postControl({ action: "reset_grid" });
  });

  container.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    state.pointerDown = { x: e.clientX, y: e.clientY };
  });

  container.addEventListener("pointerup", async (e) => {
    if (e.button !== 0 || !state.pointerDown) return;
    const dx = e.clientX - state.pointerDown.x;
    const dy = e.clientY - state.pointerDown.y;
    state.pointerDown = null;
    if (Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) return;

    const idx = viz.getHoveredIndex(e.clientX, e.clientY);
    if (idx < 0) return;
    if (e.shiftKey) {
      await postControl({ action: "toggle_conway", index: idx });
      return;
    }
    await postControl({ action: "manual_spike", index: idx });
    viz.pulseNode(idx, 520);
  });

  btnApplyRules?.addEventListener("click", async () => {
    const birth = parseRuleList(inputBirth?.value ?? "3", [3]);
    const survive = parseRuleList(inputSurvive?.value ?? "2,3", [2, 3]);
    await postControl({ action: "set_rules", birth, survive });
  });

  btnApplySpeed?.addEventListener("click", async () => {
    const generationMs = Number(inputGenerationMs?.value ?? 50);
    await postControl({
      action: "set_generation_ms",
      generation_ms: Number.isFinite(generationMs) ? generationMs : 50,
    });
  });

  async function poll() {
    try {
      const response = await fetch(`${API_BASE}/state`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json();
      if (payload && payload.gol_state && payload.spikes) {
        viz.applyFrame(payload);
        updateStats(payload, viz);
      }
    } catch (_err) {
      setText(statEls.connection, "Baglanti yok");
    } finally {
      setTimeout(poll, FETCH_MS);
    }
  }

  function loop() {
    viz.render();
    requestAnimationFrame(loop);
  }

  poll();
  requestAnimationFrame(loop);
}

bootstrap();
