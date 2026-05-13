import { HybridVisualizer } from "./visualizer.js";

const API_BASE = "http://127.0.0.1:8765";
const FETCH_MS = 70;
const DRAG_THRESHOLD_PX = 6;
const THEME_STORAGE_KEY = "viz-theme";

const state = {
  lastStep: -1,
  lastAt: performance.now(),
  hz: 0,
  pointerDown: null,
  paused: false,
  ui: {
    generationEditing: false,
    generationDirty: false,
    couplingEditing: false,
    couplingDirty: false,
  },
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
const btnTogglePause = document.getElementById("btn-toggle-pause");
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
const topologyList = document.getElementById("topology-list");
const topologyStatus = document.getElementById("topology-status");
const topologyActive = document.getElementById("topology-active");
const btnLoadTopology = document.getElementById("btn-load-topology");
const btnThemeToggle = document.getElementById("btn-theme-toggle");
const themeToggleLabel = document.getElementById("theme-toggle-label");
const perTopoGrid = document.getElementById("per-topo-grid");
const perTopoEmpty = document.getElementById("per-topo-empty");

/**
 * Per-component DOM cache: topology key -> { card root, value cells, etc. }.
 * Reusing nodes across frames avoids reflow churn at the 14 Hz poll rate
 * and keeps GC quiet during long live sessions.
 */
const perTopoCardEls = new Map();

const FALLBACK_TOPOLOGIES = [
  { key: "small_world", label: "Mevcut Topoloji (Rastgele Small-World)", kind: "small_world" },
  { key: "legacy_cluster", label: "Klasik Topoloji (Legacy Cluster)", kind: "legacy" },
  { key: "granule_test", label: "granule_test", kind: "swc" },
  { key: "medium_spiniy_test", label: "medium_spiniy_test", kind: "swc" },
  { key: "pyramidal_test", label: "pyramidal_test", kind: "swc" },
];

function setText(el, value) {
  if (el) el.textContent = value;
}

/**
 * Render the topology checkbox list. The DOM is rebuilt only when the option
 * set actually changes; user checked-state is preserved across re-renders.
 */
function renderTopologyList(options) {
  if (!topologyList) return;
  const previousChecked = new Set(getSelectedTopologyKeys());
  topologyList.innerHTML = "";
  if (!options || !options.length) {
    topologyList.innerHTML = "<div class=\"topology-status\">Topoloji listesi bos</div>";
    return;
  }
  options.forEach((opt, idx) => {
    const row = document.createElement("div");
    row.className = "chk-row";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.id = `topo-cb-${opt.key}`;
    cb.dataset.topologyKey = opt.key;
    if (previousChecked.size) {
      cb.checked = previousChecked.has(opt.key);
    } else {
      cb.checked = idx === 0;
    }
    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = opt.label || opt.key;
    const tag = document.createElement("span");
    tag.className = "kind-tag";
    tag.textContent = opt.kind === "swc" ? "SWC" : "GRAF";
    row.appendChild(cb);
    row.appendChild(lbl);
    row.appendChild(tag);
    topologyList.appendChild(row);
  });
}

function getSelectedTopologyKeys() {
  if (!topologyList) return [];
  const out = [];
  topologyList.querySelectorAll("input[type=\"checkbox\"]").forEach((cb) => {
    if (cb.checked && cb.dataset.topologyKey) out.push(cb.dataset.topologyKey);
  });
  return out;
}

async function fetchTopologyOptions() {
  try {
    const res = await fetch(`${API_BASE}/topologies`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (Array.isArray(data?.options) && data.options.length) {
      return data.options;
    }
  } catch (_err) {
    // Backend may be older or briefly down; keep the static fallback so the
    // UI is always usable.
  }
  return FALLBACK_TOPOLOGIES;
}

async function postControl(body) {
  const res = await fetch(`${API_BASE}/control`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Control request failed: HTTP ${res.status}`);
  }
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
  if (state.ui.couplingEditing || state.ui.couplingDirty) return;
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
  if (typeof payload.paused === "boolean") {
    state.paused = payload.paused;
    if (btnTogglePause) {
      btnTogglePause.textContent = state.paused ? "Devam Et" : "Durdur";
    }
  }
  if (
    typeof payload.generation_period_ms === "number"
    && inputGenerationMs
    && !state.ui.generationEditing
    && !state.ui.generationDirty
  ) {
    inputGenerationMs.value = String(payload.generation_period_ms);
  }
  syncCouplingSlidersFromPayload(payload.coupling);
  if (topologyActive) {
    if (payload.topology?.active && Array.isArray(payload.topology.selection)) {
      topologyActive.textContent = `Aktif: ${payload.topology.selection.join(", ")} (${payload.topology.n_nodes} dugum)`;
    } else {
      topologyActive.textContent = "Aktif: small-world (klasik)";
    }
  }
  // Per-topology metrics: backend ships an empty / single-element list when
  // only one component is active, in which case the renderer falls back to
  // the empty-state hint.
  renderPerComponentMetrics(payload.topology?.per_component);
}

/**
 * Read the stored UI theme. Falls back to OS preference once, then to the
 * dark default that the existing artwork was tuned against. localStorage
 * may throw in privacy modes; swallow that and return the default.
 */
function readStoredTheme() {
  try {
    const stored = localStorage.getItem(THEME_STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch (_err) {
    // Storage disabled (private mode); fall through to OS preference.
  }
  if (window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches) {
    return "light";
  }
  return "dark";
}

/**
 * Apply a theme everywhere it matters: the document root (CSS vars), the
 * Three.js scene chrome, and the toggle label. Persists the choice so it
 * survives a reload.
 */
function applyTheme(theme, viz) {
  const next = theme === "light" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  if (viz && typeof viz.setTheme === "function") {
    viz.setTheme(next);
  }
  if (themeToggleLabel) {
    themeToggleLabel.textContent = next === "light" ? "Koyu" : "Acik";
  }
  if (btnThemeToggle) {
    btnThemeToggle.setAttribute(
      "title",
      next === "light" ? "Koyu temaya gec" : "Acik temaya gec",
    );
  }
  try {
    localStorage.setItem(THEME_STORAGE_KEY, next);
  } catch (_err) {
    // Best-effort persistence; ignore quota / private-mode failures.
  }
}

function clamp01(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return 0;
  return Math.max(0, Math.min(1, v));
}

/**
 * Convert backend ``[r, g, b]`` floats (0..1) into a CSS rgb() string so
 * the per-component card border accent matches the 3D color hint.
 */
function colorTripletToCss(rgb) {
  if (!Array.isArray(rgb) || rgb.length < 3) return null;
  const r = Math.round(clamp01(rgb[0]) * 255);
  const g = Math.round(clamp01(rgb[1]) * 255);
  const b = Math.round(clamp01(rgb[2]) * 255);
  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Build the static skeleton of a per-topology card and stash references to
 * the value cells so subsequent updates only touch ``textContent``.
 */
function createPerTopoCard(component) {
  const root = document.createElement("div");
  root.className = "per-topo-card";
  const accent = colorTripletToCss(component.color);
  if (accent) root.style.borderLeftColor = accent;

  const head = document.createElement("div");
  head.className = "pt-head";
  const title = document.createElement("span");
  title.className = "pt-title";
  title.textContent = component.label || component.key;
  const tag = document.createElement("span");
  tag.className = "pt-tag";
  tag.textContent = (component.kind || "topo").toUpperCase();
  head.appendChild(title);
  head.appendChild(tag);
  root.appendChild(head);

  const valueCells = {};
  const addRow = (key, label, isEfficiency = false) => {
    const row = document.createElement("div");
    row.className = "pt-row";
    const lbl = document.createElement("span");
    lbl.className = "lbl";
    lbl.textContent = label;
    const val = document.createElement("span");
    val.className = isEfficiency ? "val eff" : "val";
    val.textContent = "-";
    row.appendChild(lbl);
    row.appendChild(val);
    root.appendChild(row);
    valueCells[key] = val;
  };

  addRow("nodes", "Dugum");
  addRow("alive_count", "Canli");
  addRow("spike_count", "Spike");
  addRow("firing_rate", "Firing rate");
  addRow("stability_score", "Stability");
  addRow("information_score", "Information");
  addRow("memory_score", "Memory");
  addRow("cost_score", "Cost");
  addRow("efficiency_score", "Efficiency", true);

  return { root, title, tag, valueCells };
}

function updatePerTopoCard(entry, component) {
  if (!entry) return;
  if (entry.title) entry.title.textContent = component.label || component.key;
  if (entry.tag) entry.tag.textContent = (component.kind || "topo").toUpperCase();
  const accent = colorTripletToCss(component.color);
  if (accent) entry.root.style.borderLeftColor = accent;
  const m = component.metrics || {};
  const cells = entry.valueCells;
  if (cells.nodes) cells.nodes.textContent = String(component.n_nodes ?? "-");
  if (cells.alive_count) cells.alive_count.textContent = String(m.alive_count ?? 0);
  if (cells.spike_count) cells.spike_count.textContent = String(m.spike_count ?? 0);
  if (cells.firing_rate) cells.firing_rate.textContent = `${(Number(m.firing_rate ?? 0) * 100).toFixed(2)}%`;
  if (cells.stability_score) cells.stability_score.textContent = Number(m.stability_score ?? 0).toFixed(3);
  if (cells.information_score) cells.information_score.textContent = Number(m.information_score ?? 0).toFixed(3);
  if (cells.memory_score) cells.memory_score.textContent = Number(m.memory_score ?? 0).toFixed(3);
  if (cells.cost_score) cells.cost_score.textContent = Number(m.cost_score ?? 0).toFixed(3);
  if (cells.efficiency_score) cells.efficiency_score.textContent = Number(m.efficiency_score ?? 0).toFixed(3);
}

/**
 * Reconcile per-component metric cards with the latest backend snapshot.
 * The "<= 1 component" branch shows a hint and clears cards so single-topology
 * sessions stay visually quiet.
 */
function renderPerComponentMetrics(perComp) {
  const list = Array.isArray(perComp) ? perComp : [];
  if (list.length <= 1) {
    if (perTopoEmpty) perTopoEmpty.style.display = "";
    perTopoCardEls.forEach((entry) => entry.root.remove());
    perTopoCardEls.clear();
    return;
  }
  if (perTopoEmpty) perTopoEmpty.style.display = "none";

  const seen = new Set();
  for (const comp of list) {
    if (!comp || !comp.key) continue;
    seen.add(comp.key);

    let cardEntry = perTopoCardEls.get(comp.key);
    if (!cardEntry) {
      cardEntry = createPerTopoCard(comp);
      perTopoCardEls.set(comp.key, cardEntry);
      perTopoGrid?.appendChild(cardEntry.root);
    }
    updatePerTopoCard(cardEntry, comp);
  }

  for (const [key, entry] of perTopoCardEls.entries()) {
    if (!seen.has(key)) {
      entry.root.remove();
      perTopoCardEls.delete(key);
    }
  }
}

async function bootstrap() {
  const container = document.getElementById("canvas-root");
  const viz = new HybridVisualizer(container);

  // Theme bootstrap must happen after the visualizer is instantiated so the
  // scene chrome (background, fog) follows the same source of truth as the
  // CSS variables on <html>.
  const initialTheme = readStoredTheme();
  applyTheme(initialTheme, viz);
  btnThemeToggle?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "light"
      ? "light"
      : "dark";
    applyTheme(current === "light" ? "dark" : "light", viz);
  });

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

  const couplingInputs = [
    rngKAlive, rngKNeighbors, rngBias, rngKSyn, rngTraceDecay, cbFeedback, cbGraphFeedbackNbors,
  ].filter(Boolean);
  couplingInputs.forEach((el) => {
    el.addEventListener("pointerdown", () => { state.ui.couplingEditing = true; });
    el.addEventListener("focus", () => { state.ui.couplingEditing = true; });
    el.addEventListener("input", () => { state.ui.couplingDirty = true; });
    el.addEventListener("change", () => { state.ui.couplingDirty = true; });
    el.addEventListener("blur", () => { state.ui.couplingEditing = false; });
  });

  inputGenerationMs?.addEventListener("focus", () => { state.ui.generationEditing = true; });
  inputGenerationMs?.addEventListener("input", () => { state.ui.generationDirty = true; });
  inputGenerationMs?.addEventListener("change", () => { state.ui.generationDirty = true; });
  inputGenerationMs?.addEventListener("blur", () => { state.ui.generationEditing = false; });

  btnApplyCoupling?.addEventListener("click", async () => {
    try {
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
      state.ui.couplingDirty = false;
      state.ui.couplingEditing = false;
    } catch (err) {
      console.error("Failed to apply coupling:", err);
    }
  });

  btnResetGrid?.addEventListener("click", async () => {
    await postControl({ action: "reset_grid" });
  });

  // Topology selection wiring
  if (topologyStatus) topologyStatus.textContent = "Yukleniyor...";
  fetchTopologyOptions().then((opts) => {
    renderTopologyList(opts);
    if (topologyStatus) topologyStatus.style.display = "none";
  });

  btnLoadTopology?.addEventListener("click", async () => {
    const selection = getSelectedTopologyKeys();
    if (!selection.length) {
      if (topologyActive) topologyActive.textContent = "Aktif: -";
      return;
    }
    btnLoadTopology.disabled = true;
    btnLoadTopology.textContent = "Yukleniyor...";
    try {
      await postControl({ action: "set_topology", selection });
      if (topologyActive) topologyActive.textContent = `Yukleniyor: ${selection.join(", ")}`;
    } finally {
      setTimeout(() => {
        btnLoadTopology.disabled = false;
        btnLoadTopology.textContent = "Topolojileri Yukle ve Calistir";
      }, 250);
    }
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
    try {
      await postControl({
        action: "set_generation_ms",
        generation_ms: Number.isFinite(generationMs) ? generationMs : 50,
      });
      state.ui.generationDirty = false;
      state.ui.generationEditing = false;
    } catch (err) {
      console.error("Failed to apply generation timing:", err);
    }
  });

  btnTogglePause?.addEventListener("click", async () => {
    try {
      await postControl({
        action: "set_paused",
        paused: !state.paused,
      });
      state.paused = !state.paused;
      btnTogglePause.textContent = state.paused ? "Devam Et" : "Durdur";
    } catch (err) {
      console.error("Failed to toggle pause:", err);
    }
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
