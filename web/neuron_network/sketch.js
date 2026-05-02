/**
 * ---------------------------------------------------------------------------
 * Özet felsefe: Yapay sinir ağı × Hücresel otomat (Conway) × Izhikevich
 * ---------------------------------------------------------------------------
 * Bu sahne, klasik 2B Izgara GoL yerine GERİ BESLEMELİ graf (sinaps mesafesi):
 *   • ANN katmanları: girdi → gizli kümeler → çıktı (ileri besleme metaforu).
 *   • Sürekli zaman: her kare Euler adımı ile Izh dinamiği (conway_izh/izhikevich.py).
 *   • Ayrık "jenerasyon" (duvar saati): CA, gizli katmanda HANGİ nöronların bir
 *     sonraki dilimde matematik (Izh) yapacağını belirler → seyrek aktivasyon.
 *     GoL — iki seçenek (CFG.conwayNeighborMetric): (1) spike: jenerasyon diliminde
 *     atan sinaptik komşular; (2) alive: anlık olarak aktif (canlı) sinaptik komşular —
 *     ızgarasız Conway’e yakın topo versiyonu.
 *   • Girdi/çıktı: dış I ve okuma için CALLOUT her jenerasyonda "canlı" tutulur—
 *     CA sadece GİZLİ ünitede işlem kapısı (compute gating).
 *   • Pasif nöron: maliyet tasarrufu—Izh atlanır, v dinlenmede sabit.
 * ---------------------------------------------------------------------------
 */

const CFG = {
  layeredLayout: true,
  /** Girdi/çıktı katmanı Conway tarafından pasife çekilmesin (ANN arayüzü). */
  ioComputeAlwaysOn: true,
  /** Klasik Conway eşikleri: doğum = tam N spike-komşu; yaşam = [min,max] aralığı. */
  conwayBirthNeighbors: 3,
  conwaySurviveMin: 2,
  conwaySurviveMax: 3,
  /**
   * 'spike': jenerasyonda atan komşu sayısı.
   * 'alive': klasik graf-GoL — sinaptık komşular arasında şu anda aktif olanlar sayılır.
   */
  conwayNeighborMetric: "spike",
  /** Kaç görüntü karesinde bir O(N²) sinaps grafiği yeniden kurulur (≥1); N>1 gecikmiş kenar takası. */
  synapseRebuildEveryFrames: 1,
  inputCount: 7,
  hiddenClusters: 3,
  hiddenPerCluster: 6,
  outputCount: 6,
  maxSpeed: 2.1,
  maxSpeedPassiveMult: 0.55,
  damping: 0.988,
  dampingPassiveMult: 0.997,
  pairAttractStrength: 0.1,
  pairAttractSoft: 6200,
  pairMaxInteract: 240,
  /** Aynı katman içi sinaps çemberi (~kümelenme) */
  linkDist: 205,
  /** Komşu katmanlar arası sinaps menzili (girdi↔gizli, gizli↔çıktı); dikey fark yüzünden büyük olmalı */
  linkDistCrossLayer: 455,
  mouseAttractStrength: 0.35,
  mouseAttractRadius: 280,
  mouseAttractOn: false,
  trailAlpha: 16,
  anchorSpringInput: 0.028,
  anchorSpringHidden: 0.018,
  anchorSpringOutput: 0.026,
  izh: { a: 0.02, b: 0.2, c: -65.0, d: 8.0, dt: 0.1, spikeThr: 30.0 },
  izhBaseI: 2.6,
  izhNoiseAmp: 0.85,
  synapticCoupling: 10.5,
  spikeFlashFrames: 14,
  fireEwmaTau: 0.92,
  fireEwmaBurst: 0.35,
  /** Conway jenerasyon süresi [ms]; sürekli Izh adımları arasında 'adım' sınırı */
  generationPeriodMs: 50,
  neighborSpikeWindowMs: 1000,
  thresholdDrop: 6.5,
  thresholdBonusDurationMs: 650,
  overloadNeighborSpikes: 4,
  pulseSpeedMin: 0.018,
  pulseSpeedMax: 0.032,
  maxPulses: 280,
  livePanelEveryNFrames: 8,
  /** Her girdi, en yakın N gizli ile omurga bağı garanti eder */
  backboneInputToHidden: 5,
  /** Her çıktı, en yakın M gizli ile bağlanır */
  backboneHiddenToOutput: 4,
};

let neurons = [];
let synapticNeighbors = [];
/** @type {{from:number,to:number,u:number,speed:number}[]} */
let synapsePulses = [];

let lastManualINote = "—";
let simStartTimeMs = 0;
/** Tamamlanan jenerasyon sayacı (ilk Conway sonrası 1 artar) */
let generationIndex = 0;
/** Bu jenerasyon zaman diliminin başlangıcı [ms] */
let generationEpochStartMs = 0;
/** Şu anki (henüz bitmemiş) jenerasyonda en az bir kez spike olduysa true */
let spikedDuringGeneration = [];

class Neuron {
  constructor(x, y, opts = {}) {
    this.x = x;
    this.y = y;
    this.vx = 0;
    this.vy = 0;
    this.r =
      opts.r ??
      2.4 + Math.random() * 1.8;
    this.v = -65.0;
    this.u = CFG.izh.b * this.v;
    this.incomingSynapse = 0;
    this.fireEwma = 0;
    this.spikeFlash = 0;
    this.life = opts.life ?? "active";
    this.spikeTimesMs = [];
    this.thresholdBonusUntilMs = 0;
    /** @type {'input'|'hidden'|'output'} */
    this.layer = opts.layer ?? "hidden";
    this.anchorX = opts.anchorX ?? x;
    this.anchorY = opts.anchorY ?? y;
    this.extraManualI = 0;
    this.extraManualIUntilMs = 0;
  }

  effectiveSpikeThr(nowMs) {
    let thr = CFG.izh.spikeThr;
    if (nowMs < this.thresholdBonusUntilMs) thr -= CFG.thresholdDrop;
    return thr;
  }

  pruneSpikeHistory(nowMs) {
    const cut = nowMs - CFG.neighborSpikeWindowMs - 250;
    this.spikeTimesMs = this.spikeTimesMs.filter((t) => t > cut);
  }

  recordSpike(nowMs) {
    this.spikeTimesMs.push(nowMs);
  }

  neighborHadSpikeInWindow(j, nowMs) {
    const o = neurons[j];
    const win = CFG.neighborSpikeWindowMs;
    return o.spikeTimesMs.some((t) => nowMs - t <= win && nowMs >= t);
  }

  manualIDrive(nowMs) {
    if (this.layer !== "input") return 0;
    if (nowMs >= this.extraManualIUntilMs) return 0;
    return this.extraManualI;
  }

  izhDrive(nowMs) {
    if (this.life === "dead") return 0;
    return (
      CFG.izhBaseI +
      (Math.random() - 0.5) * CFG.izhNoiseAmp +
      this.incomingSynapse +
      this.manualIDrive(nowMs)
    );
  }

  anchorSpringK() {
    if (this.layer === "input") return CFG.anchorSpringInput;
    if (this.layer === "output") return CFG.anchorSpringOutput;
    return CFG.anchorSpringHidden;
  }

  integrateMotion(w, h, pmx, pmy) {
    if (this.life === "dead") {
      this.vx *= 0.88;
      this.vy *= 0.88;
      const sp = Math.hypot(this.vx, this.vy);
      const cap = CFG.maxSpeed * 0.12;
      if (sp > cap) {
        const s = cap / sp;
        this.vx *= s;
        this.vy *= s;
      }
      this.x += this.vx;
      this.y += this.vy;
      const pad = this.r + 2;
      this.x = Math.max(pad, Math.min(w - pad, this.x));
      this.y = Math.max(pad, Math.min(h - pad, this.y));
      return;
    }

    const damp =
      this.life === "passive"
        ? CFG.damping * CFG.dampingPassiveMult
        : CFG.damping;
    const vCap =
      this.life === "passive"
        ? CFG.maxSpeed * CFG.maxSpeedPassiveMult
        : CFG.maxSpeed;

    let ax = (Math.random() - 0.5) * 0.022;
    let ay = (Math.random() - 0.5) * 0.022;
    ax *= this.life === "passive" ? 0.45 : 1;
    ay *= this.life === "passive" ? 0.45 : 1;

    const k = this.anchorSpringK();
    ax += (this.anchorX - this.x) * k;
    ay += (this.anchorY - this.y) * k;

    neurons.forEach((o) => {
      if (o === this || o.life === "dead") return;
      let dx = o.x - this.x;
      let dy = o.y - this.y;
      let d = Math.sqrt(dx * dx + dy * dy);
      if (d < 1e-6 || d > CFG.pairMaxInteract) return;
      let inv = CFG.pairAttractStrength / (d * d + CFG.pairAttractSoft);
      ax += dx * inv;
      ay += dy * inv;
    });

    if (CFG.mouseAttractOn) {
      let mx = pmx - this.x;
      let my = pmy - this.y;
      let md = Math.hypot(mx, my);
      if (md > 4 && md < CFG.mouseAttractRadius) {
        let t =
          CFG.mouseAttractStrength *
          (1 - md / CFG.mouseAttractRadius) ** 2;
        ax += (mx / md) * t;
        ay += (my / md) * t;
      }
    }

    this.vx += ax;
    this.vy += ay;
    this.vx *= damp;
    this.vy *= damp;
    const spd = Math.hypot(this.vx, this.vy);
    if (spd > vCap) {
      const s = vCap / spd;
      this.vx *= s;
      this.vy *= s;
    }
    this.x += this.vx;
    this.y += this.vy;
    const pad = this.r + 2;
    if (this.x < pad) {
      this.x = pad;
      this.vx *= -0.5;
    } else if (this.x > w - pad) {
      this.x = w - pad;
      this.vx *= -0.5;
    }
    if (this.y < pad) {
      this.y = pad;
      this.vy *= -0.5;
    } else if (this.y > h - pad) {
      this.y = h - pad;
      this.vy *= -0.5;
    }
  }

  decayFlash() {
    if (this.spikeFlash > 0) this.spikeFlash -= 1;
  }
}

function layerRank(layer) {
  if (layer === "input") return 0;
  if (layer === "hidden") return 1;
  return 2;
}

/** i–j çiftinde sinaps oluşturmak için maksimum Euclidean mesafe */
function maxSynapseReach(i, j) {
  const a = neurons[i].layer;
  const b = neurons[j].layer;
  const ri = layerRank(a);
  const rj = layerRank(b);
  const d = Math.abs(ri - rj);
  if (d === 0) return CFG.linkDist;
  if (d === 1) return CFG.linkDistCrossLayer;
  return Math.min(CFG.linkDist, CFG.linkDistCrossLayer * 0.35);
}

function sortedIndicesByDist(fromIdx, pool) {
  return pool
    .map((idx) => ({ idx, dist: neuronDistSq(fromIdx, idx) }))
    .sort((a, b) => a.dist - b.dist)
    .map((x) => x.idx);
}

function neuronDistSq(i, j) {
  const a = neurons[i];
  const b = neurons[j];
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function addSynapseUndirected(i, j) {
  if (i === j || i < 0 || j < 0) return;
  if (
    synapticNeighbors[i].includes(j)
  )
    return;
  synapticNeighbors[i].push(j);
  synapticNeighbors[j].push(i);
}

/**
 * Grafik yalnızca linkDist yüzünden kopmasın diye girdi→gizli ve gizli→çıktı minimum bağlar.
 */
function ensureFeedforwardBackbone() {
  const n = neurons.length;
  const inIdx = [];
  const hidIdx = [];
  const outIdx = [];
  for (let i = 0; i < n; i++) {
    if (neurons[i].layer === "input") {
      inIdx.push(i);
    } else if (neurons[i].layer === "hidden") {
      hidIdx.push(i);
    } else {
      outIdx.push(i);
    }
  }
  for (const i of inIdx) {
    const near = sortedIndicesByDist(i, hidIdx).slice(
      0,
      CFG.backboneInputToHidden,
    );
    for (const h of near) addSynapseUndirected(i, h);
  }
  for (const o of outIdx) {
    const near = sortedIndicesByDist(o, hidIdx).slice(
      0,
      CFG.backboneHiddenToOutput,
    );
    for (const h of near) addSynapseUndirected(o, h);
  }
}

/** Soldan sağa öncelikli yönsel uçlar [kaynak, hedef] */
function feedforwardDirectedEnds(i, j) {
  const a = neurons[i];
  const b = neurons[j];
  const ra = layerRank(a.layer);
  const rb = layerRank(b.layer);
  if (ra < rb) return [i, j];
  if (rb < ra) return [j, i];
  if (a.x <= b.x) return [i, j];
  return [j, i];
}

function spawnPulsesFromSpike(spikerIndex) {
  const neigh = synapticNeighbors[spikerIndex];
  if (!neigh) return;
  for (const j of neigh) {
    const [src, dst] = feedforwardDirectedEnds(spikerIndex, j);
    if (src !== spikerIndex) continue;
    if (synapsePulses.length >= CFG.maxPulses) return;
    synapsePulses.push({
      from: src,
      to: dst,
      u: 0,
      speed:
        CFG.pulseSpeedMin +
        Math.random() * (CFG.pulseSpeedMax - CFG.pulseSpeedMin),
    });
  }
}

function updateAndDrawPulses(p) {
  for (let k = synapsePulses.length - 1; k >= 0; k--) {
    const pul = synapsePulses[k];
    const a = neurons[pul.from];
    const b = neurons[pul.to];
    if (!a || !b) {
      synapsePulses.splice(k, 1);
      continue;
    }
    pul.u += pul.speed;
    if (pul.u >= 1) {
      synapsePulses.splice(k, 1);
      continue;
    }
    const x = p.lerp(a.x, b.x, pul.u);
    const y = p.lerp(a.y, b.y, pul.u);
    p.noStroke();
    p.fill(255, 252, 210, 230);
    p.circle(x, y, 4.2);
    p.fill(120, 200, 255, 85);
    p.circle(x, y, 10);
    p.fill(255, 180, 60, 55);
    p.circle(x, y, 15);
  }
}

function expectedNeuronCount() {
  return (
    CFG.inputCount +
    CFG.hiddenClusters * CFG.hiddenPerCluster +
    CFG.outputCount
  );
}

function createLayeredNeurons(w, h) {
  neurons = [];
  synapsePulses = [];
  const mx = 36;
  const my = 28;
  const usableW = w - mx * 2;
  const usableH = h - my * 2;

  const inX = mx + usableW * 0.07;
  const outX = mx + usableW * 0.93;
  for (let i = 0; i < CFG.inputCount; i++) {
    const t = (i + 1) / (CFG.inputCount + 1);
    const y = my + t * usableH;
    const jx = (Math.random() - 0.5) * 14;
    const jy = (Math.random() - 0.5) * 18;
    neurons.push(
      new Neuron(inX + jx, y + jy, {
        layer: "input",
        anchorX: inX + jx * 0.3,
        anchorY: y,
        life: "active",
      }),
    );
  }

  for (let c = 0; c < CFG.hiddenClusters; c++) {
    const cx = mx + usableW * (0.32 + c * 0.14);
    const cyBase = my + usableH * 0.5 + (c - 1) * 36;
    for (let k = 0; k < CFG.hiddenPerCluster; k++) {
      const ang = (k / CFG.hiddenPerCluster) * Math.PI * 2 + c * 0.4;
      const rad = 22 + (k % 3) * 8;
      const jx = (Math.random() - 0.5) * 12;
      const jy = (Math.random() - 0.5) * 12;
      const x = cx + Math.cos(ang) * rad + jx;
      const y = cyBase + Math.sin(ang) * rad * 0.85 + jy;
      neurons.push(
        new Neuron(x, y, {
          layer: "hidden",
          anchorX: cx + Math.cos(ang) * rad * 0.85,
          anchorY: cyBase + Math.sin(ang) * rad * 0.7,
          life: Math.random() < 0.78 ? "active" : "passive",
        }),
      );
    }
  }

  for (let i = 0; i < CFG.outputCount; i++) {
    const t = (i + 1) / (CFG.outputCount + 1);
    const y = my + t * usableH;
    const jx = (Math.random() - 0.5) * 14;
    const jy = (Math.random() - 0.5) * 18;
    neurons.push(
      new Neuron(outX + jx, y + jy, {
        layer: "output",
        anchorX: outX + jx * 0.3,
        anchorY: y,
        life: "active",
      }),
    );
  }

  const ex = expectedNeuronCount();
  if (neurons.length !== ex) {
    console.warn(
      "[ayn-sim] Beklenen nöron sayısı",
      ex,
      "fiilî:",
      neurons.length,
    );
  }
}

function rebuildSynapticGraph() {
  const n = neurons.length;
  synapticNeighbors = new Array(n);
  for (let i = 0; i < n; i++) synapticNeighbors[i] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dxy = Math.hypot(neurons[j].x - neurons[i].x, neurons[j].y - neurons[i].y);
      const cap = maxSynapseReach(i, j);
      if (dxy >= cap || dxy < 1e-6) continue;
      synapticNeighbors[i].push(j);
      synapticNeighbors[j].push(i);
    }
  }
  if (CFG.layeredLayout) ensureFeedforwardBackbone();
}

/**
 * CA kapısı — sadece GİZLİ: önceki jenerasyonda spike atmış sinaptik komşu sayısı.
 * ANN felsefesi: I/O her zaman "okuyor/yazıyor"; ara temsil (hidden) seyrek güncellenir.
 *
 * @param {boolean[]} didSpikeLastGen indeks → önceki jenerasyonda ≥1 spike mi?
 */
function applyConwayControlFromSpikes(didSpikeLastGen) {
  const n = neurons.length;
  const birthN = CFG.conwayBirthNeighbors;
  const sLo = CFG.conwaySurviveMin;
  const sHi = CFG.conwaySurviveMax;
  const next = new Array(n);
  for (let i = 0; i < n; i++) {
    const neu = neurons[i];
    if (neu.life === "dead") {
      next[i] = "dead";
      continue;
    }
    if (
      CFG.layeredLayout &&
      CFG.ioComputeAlwaysOn &&
      (neu.layer === "input" || neu.layer === "output")
    ) {
      next[i] = "active";
      continue;
    }
    let spikeNeighbors = 0;
    for (const j of synapticNeighbors[i]) {
      if (didSpikeLastGen[j]) spikeNeighbors++;
    }
    if (neu.life === "passive") {
      next[i] = spikeNeighbors === birthN ? "active" : "passive";
    } else {
      next[i] =
        spikeNeighbors >= sLo && spikeNeighbors <= sHi ? "active" : "passive";
    }
  }
  for (let i = 0; i < n; i++) neurons[i].life = next[i];
}

/**
 * Jenerasyon süresi dolduysa: önceki dilimde biriken spike'lara göre Conway,
 * sonra spike birikimini sıfırla ve zamanı hizala.
 */
function clampGenerationPeriodMs(v) {
  return Math.max(8, Math.min(800, Number(v) || 50));
}

function clampSynapseRebuildEveryFrames(v) {
  return Math.max(1, Math.min(60, Math.round(Number(v) || 1)));
}

/**
 * Graf üzerinde klasik Conway mantığına yakın: sinaptık komşulardan şu anda
 * aktif (‘canlı’) olanlar bir önceki anlık topo durumundan sayılır; pasif/ölü dahil edilmez.
 */
function applyConwayControlFromGraphAlive() {
  const n = neurons.length;
  const birthN = CFG.conwayBirthNeighbors;
  const sLo = CFG.conwaySurviveMin;
  const sHi = CFG.conwaySurviveMax;
  const next = new Array(n);
  for (let i = 0; i < n; i++) {
    const neu = neurons[i];
    if (neu.life === "dead") {
      next[i] = "dead";
      continue;
    }
    if (
      CFG.layeredLayout &&
      CFG.ioComputeAlwaysOn &&
      (neu.layer === "input" || neu.layer === "output")
    ) {
      next[i] = "active";
      continue;
    }
    let aliveNeigh = 0;
    for (const j of synapticNeighbors[i]) {
      if (neurons[j].life === "active") aliveNeigh++;
    }
    if (neu.life === "passive") {
      next[i] = aliveNeigh === birthN ? "active" : "passive";
    } else {
      next[i] =
        aliveNeigh >= sLo && aliveNeigh <= sHi ? "active" : "passive";
    }
  }
  for (let i = 0; i < n; i++) neurons[i].life = next[i];
}

function maybeAdvanceGenerationWallClock(nowMs) {
  if (neurons.length === 0) return;
  const period = clampGenerationPeriodMs(CFG.generationPeriodMs);
  if (nowMs - generationEpochStartMs < period) return;

  if (CFG.conwayNeighborMetric === "alive") applyConwayControlFromGraphAlive();
  else applyConwayControlFromSpikes(spikedDuringGeneration);

  spikedDuringGeneration = new Array(neurons.length).fill(false);
  generationEpochStartMs = nowMs;
  generationIndex++;
}

function applyNeighborSpikeRules(nowMs) {
  const n = neurons.length;
  for (let i = 0; i < n; i++) {
    const neu = neurons[i];
    neu.pruneSpikeHistory(nowMs);
    if (neu.life === "dead") continue;

    const neigh = synapticNeighbors[i];
    let simul = 0;
    let inWindow = 0;
    for (const j of neigh) {
      const o = neurons[j];
      if (o.life === "dead") continue;
      if (o._spikedThisFrame) simul++;
      if (neu.neighborHadSpikeInWindow(j, nowMs)) inWindow++;
    }

    if (simul > CFG.overloadNeighborSpikes) {
      neu.life = "dead";
      neu.v = CFG.izh.c;
      neu.u = CFG.izh.b * neu.v;
      neu.vx *= 0.2;
      neu.vy *= 0.2;
      neu.spikeFlash = 0;
      neu.thresholdBonusUntilMs = 0;
      continue;
    }

    if (inWindow === 3) {
      neu.thresholdBonusUntilMs = Math.max(
        neu.thresholdBonusUntilMs,
        nowMs + CFG.thresholdBonusDurationMs,
      );
    }
  }
}

function stepIzhikevichNetwork(nowMs) {
  const { a, b, c, d, dt } = CFG.izh;
  const n = neurons.length;
  const vNew = new Array(n);
  const uNew = new Array(n);
  const spiked = new Array(n);
  /** Pasif Conway hücreleri için dinlenme; Izh Euler atlanır (maliyet). */
  const vRest = c;
  const uRest = b * c;

  for (let i = 0; i < n; i++) {
    neurons[i]._spikedThisFrame = false;
  }

  for (let i = 0; i < n; i++) {
    const neu = neurons[i];
    if (neu.life === "dead") {
      vNew[i] = neu.v;
      uNew[i] = neu.u;
      spiked[i] = false;
      neu.incomingSynapse = 0;
      continue;
    }

    if (neu.life === "passive") {
      neu.incomingSynapse = 0;
      spiked[i] = false;
      vNew[i] = vRest;
      uNew[i] = uRest;
      continue;
    }

    const I = neu.izhDrive(nowMs);
    neu.incomingSynapse = 0;
    const thr = neu.effectiveSpikeThr(nowMs);
    let v = neu.v;
    let u = neu.u;
    const dv = dt * (0.04 * v * v + 5 * v + 140 - u + I);
    const du = dt * a * (b * v - u);
    let vn = v + dv;
    let un = u + du;
    const sp = vn >= thr;
    spiked[i] = sp;
    if (sp) {
      vn = c;
      un += d;
      neu.fireEwma +=
        CFG.fireEwmaBurst * (1 - Math.min(1, neu.fireEwma + 0.01));
      neu.spikeFlash = CFG.spikeFlashFrames;
      neu.recordSpike(nowMs);
      neu._spikedThisFrame = true;
    }
    vNew[i] = vn;
    uNew[i] = un;
  }

  for (let i = 0; i < n; i++) {
    neurons[i].v = vNew[i];
    neurons[i].u = uNew[i];
  }

  for (let i = 0; i < n; i++) {
    if (spiked[i]) spawnPulsesFromSpike(i);
  }

  for (let i = 0; i < n; i++) neurons[i].fireEwma *= CFG.fireEwmaTau;

  for (let i = 0; i < n; i++) {
    if (!spiked[i]) continue;
    const src = neurons[i];
    if (src.life === "dead" || src.life === "passive") continue;
    for (const j of synapticNeighbors[i]) {
      const tgt = neurons[j];
      if (tgt.life === "dead" || tgt.life === "passive") continue;
      let dist = Math.hypot(tgt.x - src.x, tgt.y - src.y);
      const cap = maxSynapseReach(i, j);
      if (dist >= cap || dist < 1e-6) continue;
      const prox = 1 - dist / cap;
      tgt.incomingSynapse += CFG.synapticCoupling * prox * prox;
    }
  }
}

function lineStrokeFromFirerate(combinedRate, baseAlpha, p) {
  const fr = Math.max(0, Math.min(1, combinedRate));
  const mix = Math.pow(fr, 0.55);
  const r = p.lerp(55, 255, mix);
  const g = p.lerp(155, 90, mix);
  const bl = p.lerp(248, 75, mix);
  p.stroke(r, g, bl, baseAlpha);
}

function drawNeuronBody(p, neu) {
  p.noStroke();
  if (neu.life === "dead") {
    p.fill(72, 76, 82, 160);
    p.circle(neu.x, neu.y, neu.r * 2);
    p.fill(55, 58, 62, 90);
    p.circle(neu.x, neu.y, neu.r * 2.5);
    return;
  }
  const vNorm = p.constrain((neu.v + 75) / 35, 0, 1);
  let coldR = p.lerp(160, 220, vNorm);
  let coldG = p.lerp(195, 235, vNorm);
  let coldB = p.lerp(255, 255, vNorm);
  if (neu.life === "passive") {
    coldR *= 0.55;
    coldG *= 0.62;
    coldB *= 0.85;
  }
  if (neu.layer === "input") {
    coldB = Math.min(255, coldB + 25);
    coldG *= 1.06;
  } else if (neu.layer === "output") {
    coldR = Math.min(255, coldR + 28);
    coldG *= 0.92;
  }
  let fillR = coldR;
  let fillG = coldG;
  let fillB = coldB;
  let haloA = neu.life === "passive" ? 55 : 90;
  if (neu.spikeFlash > 0 && neu.life !== "dead") {
    const pulse = neu.spikeFlash / CFG.spikeFlashFrames;
    fillR = p.lerp(255, coldR, 1 - pulse);
    fillG = p.lerp(220, coldG, 1 - pulse);
    fillB = p.lerp(40, coldB, 1 - pulse);
    haloA = 140;
  }
  const bodyA = neu.life === "passive" ? 155 : 235;
  p.fill(fillR, fillG, fillB, bodyA);
  p.circle(neu.x, neu.y, neu.r * 2 * (neu.life === "passive" ? 0.88 : 1));
  p.fill(fillR, fillG + 40, fillB + 30, haloA * 0.82);
  p.circle(neu.x, neu.y, neu.r * 2.75 * (neu.life === "passive" ? 0.9 : 1));
}

function drawSynapses(p) {
  const n = neurons.length;
  for (let i = 0; i < n; i++) {
    for (const j of synapticNeighbors[i]) {
      if (j < i) continue;
      const a = neurons[i];
      const b = neurons[j];
      let dxy = Math.hypot(b.x - a.x, b.y - a.y);
      const cap = maxSynapseReach(i, j);
      if (dxy < 1e-6) continue;
      let tProximity = 1 - Math.min(dxy / cap, 1 - 1e-6);
      let alpha = 34 + tProximity * tProximity * 198;
      let sw = 0.32 + tProximity * tProximity * 2.4;
      if (a.life === "dead" || b.life === "dead") {
        alpha *= 0.22;
        sw *= 0.45;
      } else if (a.life === "passive" && b.life === "passive") {
        alpha *= 0.55;
      }
      const combined =
        Math.max(a.fireEwma, b.fireEwma) * 0.65 +
        Math.min(a.fireEwma, b.fireEwma) * 0.35;
      lineStrokeFromFirerate(combined, alpha, p);
      p.strokeWeight(sw);
      p.line(a.x, a.y, b.x, b.y);
    }
  }
}

function countEdges() {
  let e = 0;
  for (let i = 0; i < synapticNeighbors.length; i++) e += synapticNeighbors[i].length;
  return e >> 1;
}

/** Kodda olduğu gibi: yalnızca life==='active' için Izh Euler + çoğu sinaptik hedef işlemi yapılır. */
function computeIzhiWorkloadMetrics() {
  const traditional = neurons.length;
  let current = 0;
  for (const n of neurons) {
    if (n.life === "active") current++;
  }
  const savingsPct =
    traditional > 0 ? (1 - current / traditional) * 100 : 0;
  return { traditional, current, savingsPct };
}

function layerStats(layerName) {
  let a = 0,
    pa = 0,
    d = 0;
  let vsum = 0,
    vc = 0;
  for (const n of neurons) {
    if (n.layer !== layerName) continue;
    if (n.life === "active") a++;
    else if (n.life === "passive") pa++;
    else d++;
    if (n.life !== "dead") {
      vsum += n.v;
      vc++;
    }
  }
  const meanV = vc ? vsum / vc : 0;
  return { a, pa, d, meanV };
}

function collectLiveSnapshot(frame, nowMs, spikesLast) {
  const li = layerStats("input");
  const lh = layerStats("hidden");
  const lo = layerStats("output");
  const eff = computeIzhiWorkloadMetrics();
  const caMode =
    CFG.conwayNeighborMetric === "alive" ? "aktif komşu" : "spike";
  return {
    frame,
    gen: generationIndex,
    t: (nowMs - simStartTimeMs) / 1000,
    edges: countEdges(),
    spikesLast,
    in: li,
    hid: lh,
    out: lo,
    eff,
    caMode,
    rebuildEvery: clampSynapseRebuildEveryFrames(
      CFG.synapseRebuildEveryFrames,
    ),
  };
}

function updateLivePanelDom(snap) {
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.textContent = v;
  };
  set("stat-gen", String(snap.gen));
  set("stat-gen-dt", String(clampGenerationPeriodMs(CFG.generationPeriodMs)));
  set("stat-frame", String(snap.frame));
  set("stat-time", snap.t.toFixed(2));
  set("stat-edges", String(snap.edges));
  set("stat-spikes", String(snap.spikesLast));
  set(
    "stat-ca-rebuild-short",
    snap.caMode != null && snap.rebuildEvery != null
      ? `${snap.caMode} · N=${snap.rebuildEvery}`
      : "—",
  );
  const fmt = (x) => `${x.a} / ${x.pa} / ${x.d}`;
  set("stat-in", fmt(snap.in));
  set("stat-hid", fmt(snap.hid));
  set("stat-out", fmt(snap.out));
  set(
    "stat-v",
    `${snap.in.meanV.toFixed(1)} | ${snap.hid.meanV.toFixed(1)} | ${snap.out.meanV.toFixed(1)}`,
  );
  set("stat-last-i", lastManualINote);
  if (snap.eff) {
    set("stat-eff-traditional", String(snap.eff.traditional));
    set("stat-eff-current", String(snap.eff.current));
    const se = document.getElementById("stat-eff-savings");
    if (se) se.textContent = `${snap.eff.savingsPct.toFixed(1)} %`;
  }
}

/**
 * Girdi katmanındaki tüm nöronlara manuel akım I uygular (belirtilen süre).
 * @param {number|number[]} I tek değer veya her girdi indeksi için dizi
 * @param {number} durationMs
 */
function injectInputLayerCurrent(I, durationMs = 900) {
  const now = performance.now();
  const inputs = neurons.filter((n) => n.layer === "input");
  if (inputs.length === 0) return;
  if (Array.isArray(I)) {
    for (let i = 0; i < inputs.length; i++) {
      const val = I[i] ?? I[I.length - 1] ?? 0;
      inputs[i].extraManualI = val;
      inputs[i].extraManualIUntilMs = now + durationMs;
    }
    lastManualINote = `[${I.map((x) => Number(x).toFixed(1)).join(", ")}] (${durationMs}ms)`;
  } else {
    const v = Number(I);
    for (const n of inputs) {
      n.extraManualI = v;
      n.extraManualIUntilMs = now + durationMs;
    }
    lastManualINote = `${v.toFixed(1)} (${durationMs}ms)`;
  }
}

function syncMouseAttractCheckbox() {
  const cb = document.getElementById("mouse-attract-cb");
  if (cb) cb.checked = CFG.mouseAttractOn;
}

function wirePanelControls() {
  const r = document.getElementById("gen-period-range");
  const lbl = document.getElementById("gen-period-label");
  if (r) {
    const apply = () => {
      CFG.generationPeriodMs = clampGenerationPeriodMs(r.value);
      r.value = String(CFG.generationPeriodMs);
      if (lbl) lbl.textContent = String(CFG.generationPeriodMs);
    };
    r.addEventListener("input", apply);
    r.addEventListener("change", apply);
    r.value = String(clampGenerationPeriodMs(CFG.generationPeriodMs));
    apply();
  }
  const cb = document.getElementById("mouse-attract-cb");
  if (cb) {
    cb.checked = CFG.mouseAttractOn;
    cb.addEventListener("change", () => {
      CFG.mouseAttractOn = cb.checked;
    });
  }

  const met = document.getElementById("conway-metric");
  if (met) {
    met.value =
      CFG.conwayNeighborMetric === "alive" ? "alive" : "spike";
    met.addEventListener("change", () => {
      CFG.conwayNeighborMetric =
        met.value === "alive" ? "alive" : "spike";
    });
  }

  const rin = document.getElementById("rebuild-every-input");
  if (rin) {
    const rsync = () => {
      CFG.synapseRebuildEveryFrames = clampSynapseRebuildEveryFrames(
        rin.value,
      );
      rin.value = String(CFG.synapseRebuildEveryFrames);
    };
    rin.addEventListener("change", rsync);
    rin.value = String(
      clampSynapseRebuildEveryFrames(CFG.synapseRebuildEveryFrames),
    );
    rsync();
  }
}

function resetNeurons(w, h) {
  if (!w || !h || !Number.isFinite(w) || !Number.isFinite(h)) return;
  createLayeredNeurons(w, h);
  rebuildSynapticGraph();
  simStartTimeMs = performance.now();
  generationIndex = 0;
  generationEpochStartMs = performance.now();
  spikedDuringGeneration = new Array(neurons.length).fill(false);
  lastManualINote = "—";
}

function sketch(p) {
  p.setup = function () {
    const stage = document.getElementById("p5-root");
    const sw = Math.min(920, Math.max(520, window.innerWidth - 300));
    const sh = Math.min(620, Math.max(380, window.innerHeight - 160));
    p.createCanvas(sw, sh);
    resetNeurons(p.width, p.height);
    const c = document.querySelector("#p5-root canvas");
    if (c) c.setAttribute("tabindex", "0");

    const inj = () => {
      const raw = document.getElementById("manual-I");
      const v = raw ? parseFloat(raw.value) : 24;
      injectInputLayerCurrent(Number.isFinite(v) ? v : 24, 950);
    };
    document.getElementById("btn-inject")?.addEventListener("click", inj);
    document.getElementById("btn-reset")?.addEventListener("click", () => {
      resetNeurons(p.width, p.height);
    });

    window.tezSimulation = {
      injectInputLayerCurrent: injectInputLayerCurrent,
      resetLayout: () => resetNeurons(p.width, p.height),
      getNeurons: () => neurons,
      getConfig: () => CFG,
      setGenerationPeriodMs: (ms) => {
        CFG.generationPeriodMs = clampGenerationPeriodMs(ms);
        const r = document.getElementById("gen-period-range");
        const lbl = document.getElementById("gen-period-label");
        if (r) r.value = String(CFG.generationPeriodMs);
        if (lbl) lbl.textContent = String(CFG.generationPeriodMs);
      },
      setConwayNeighborMetric: (mode) => {
        CFG.conwayNeighborMetric = mode === "alive" ? "alive" : "spike";
        const met = document.getElementById("conway-metric");
        if (met) met.value = CFG.conwayNeighborMetric;
      },
      setSynapseRebuildEveryFrames: (n) => {
        CFG.synapseRebuildEveryFrames = clampSynapseRebuildEveryFrames(n);
        const rin = document.getElementById("rebuild-every-input");
        if (rin) rin.value = String(CFG.synapseRebuildEveryFrames);
      },
    };

    wirePanelControls();
    updateLivePanelDom(collectLiveSnapshot(0, performance.now(), 0));
  };

  p.windowResized = function () {
    const sw = Math.min(920, Math.max(520, window.innerWidth - 300));
    const sh = Math.min(620, Math.max(380, window.innerHeight - 160));
    p.resizeCanvas(sw, sh);
    resetNeurons(p.width, p.height);
  };

  p.draw = function () {
    const nowMs = performance.now();

    p.fill(0, CFG.trailAlpha);
    p.noStroke();
    p.rect(0, 0, p.width, p.height);

    neurons.forEach((neu) => {
      neu.integrateMotion(p.width, p.height, p.mouseX, p.mouseY);
      neu.decayFlash();
    });

    const rebuildN = clampSynapseRebuildEveryFrames(
      CFG.synapseRebuildEveryFrames,
    );
    CFG.synapseRebuildEveryFrames = rebuildN;
    if (rebuildN <= 1 || p.frameCount % rebuildN === 0)
      rebuildSynapticGraph();

    maybeAdvanceGenerationWallClock(nowMs);

    let spikesCount = 0;
    stepIzhikevichNetwork(nowMs);
    for (let i = 0; i < neurons.length; i++) {
      if (neurons[i]._spikedThisFrame) {
        spikesCount++;
        spikedDuringGeneration[i] = true;
      }
    }

    applyNeighborSpikeRules(nowMs);

    drawSynapses(p);
    updateAndDrawPulses(p);
    neurons.forEach((neu) => drawNeuronBody(p, neu));

    if (p.frameCount % CFG.livePanelEveryNFrames === 0) {
      updateLivePanelDom(
        collectLiveSnapshot(p.frameCount, nowMs, spikesCount),
      );
    }
  };

  p.mousePressed = function () {
    if (CFG.layeredLayout) return false;
    return false;
  };

  p.keyPressed = function () {
    const k = p.key?.toLowerCase?.() ?? "";
    if (k === "r") resetNeurons(p.width, p.height);
    else if (k === "g") {
      CFG.mouseAttractOn = !CFG.mouseAttractOn;
      syncMouseAttractCheckbox();
    }
    else if (k === "i") {
      injectInputLayerCurrent(22, 800);
    }
  };
}

new p5(sketch, "p5-root");
