import * as THREE from "https://unpkg.com/three@0.166.1/build/three.module.js";
import { OrbitControls } from "https://unpkg.com/three@0.166.1/examples/jsm/controls/OrbitControls.js";
import { EffectComposer } from "https://unpkg.com/three@0.166.1/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "https://unpkg.com/three@0.166.1/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "https://unpkg.com/three@0.166.1/examples/jsm/postprocessing/UnrealBloomPass.js";

/** Kozmik mor temelli palet · yeşil–altın ateş · turuncu–kırmızı soğuma */
const DEAD_COLOR = new THREE.Color(0x5a4278);
const ALIVE_COLOR = new THREE.Color(0x2af0b8);
const SPIKE_COLOR = new THREE.Color(0xffd84a);
const TRAIL_HOT = new THREE.Color(0xffea70);
const TRAIL_COOL = new THREE.Color(0xd4283c);
const BASE_EDGE_COLOR = new THREE.Color(0x634a82);
const NEON_BURST = new THREE.Color(0x62ffb4);

/**
 * Per-theme scene chrome.
 *
 * IMPORTANT: even in "light" UI mode the WebGL viewport keeps a dark
 * background. Reason: ``UnrealBloomPass`` extracts every fragment whose
 * luminance exceeds ``bloomThreshold`` (~0.29 in the medium preset). A
 * near-white background has luminance ~0.95, so on a light bg the bloom
 * extractor selects the entire frame and blurs it back over the scene –
 * the result is a fully washed-out, all-white image with no particles
 * visible. This matches industry tooling: Blender, Maya and Photoshop
 * all keep a dark canvas regardless of UI theme so HDR content stays
 * legible. The brighter bg + lower exposure preset is kept for slightly
 * softer contrast that pairs better with a light surrounding UI.
 */
const THEME_PRESETS = {
  dark: {
    background: 0x090618,
    fogColor: 0x0c0822,
    fogDensity: 0.014,
    toneExposure: 1.08,
    baseEdgeColor: new THREE.Color(0x634a82),
  },
  light: {
    background: 0x141a2b,
    fogColor: 0x141a2b,
    fogDensity: 0.012,
    toneExposure: 1.04,
    baseEdgeColor: new THREE.Color(0x8c74c0),
  },
};

/** Quality presets: bloom + edges + halo */
const QUALITY_PRESETS = {
  low: {
    pixelRatioCap: 1.0,
    maxActiveSegments: 6000,
    baseEdgeStride: 4,
    haloStrength: 0.5,
    bloomStrength: 0.55,
    bloomRadius: 0.38,
    bloomThreshold: 0.38,
    baseOpacity: 0.2,
  },
  medium: {
    pixelRatioCap: 1.25,
    maxActiveSegments: 12000,
    baseEdgeStride: 1,
    haloStrength: 0.62,
    bloomStrength: 0.82,
    bloomRadius: 0.52,
    bloomThreshold: 0.29,
    baseOpacity: 0.26,
  },
  high: {
    pixelRatioCap: 1.5,
    maxActiveSegments: 24000,
    baseEdgeStride: 1,
    haloStrength: 0.74,
    bloomStrength: 1.08,
    bloomRadius: 0.62,
    bloomThreshold: 0.22,
    baseOpacity: 0.3,
  },
};

const VERT_SHADER = `
attribute float aSize;
attribute float aFlash;
attribute vec3 color;
varying vec3 vColor;
varying float vFlash;
void main() {
  vColor = color;
  vFlash = aFlash;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  float dist = max(0.1, -mvPosition.z);
  float pulse = clamp(aFlash, 0.0, 1.0);
  float s = aSize * (280.0 / dist) * (1.0 + pulse * 1.05);
  gl_PointSize = clamp(s, 2.0, 34.0);
  gl_Position = projectionMatrix * mvPosition;
}
`;

const FRAG_SHADER = `
precision highp float;
uniform float uHaloStrength;
varying vec3 vColor;
varying float vFlash;
void main() {
  vec2 p = gl_PointCoord - vec2(0.5);
  float r = length(p);
  if (r > 0.5) discard;
  float core = smoothstep(0.2, 0.06, r);
  float halo = smoothstep(0.5, 0.12, r) * uHaloStrength;
  float alpha = clamp(core * 0.98 + halo * 0.55, 0.0, 1.0);
  float f = clamp(vFlash, 0.0, 1.0);
  vec3 gold = vec3(1.0, 0.88, 0.32);
  vec3 neon = vec3(0.38, 1.0, 0.62);
  vec3 fire = mix(vColor, mix(gold, neon, f * 0.72), f);
  vec3 c = mix(fire, vec3(1.0), f * f * 0.35);
  float hdr = 1.0 + f * 4.2;
  gl_FragColor = vec4(c * hdr, alpha);
}
`;

function indexKey(a, b) {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

export class HybridVisualizer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.themeName = "dark";
    const themePreset = THEME_PRESETS[this.themeName];
    this.scene.background = new THREE.Color(themePreset.background);
    this.scene.fog = new THREE.FogExp2(themePreset.fogColor, themePreset.fogDensity);
    this._currentBaseEdgeColor = themePreset.baseEdgeColor.clone();
    // Cached scene span set by ``_fitCameraToScene``. Used by
    // ``_applySceneFog`` so the fog density auto-shrinks when many
    // topologies are loaded side-by-side – otherwise the default density
    // (tuned for a single ~80-unit component) absorbs >99% of every
    // particle past the first component and the viewport looks empty.
    this._sceneExtent = 80;
    // Reusable Vector3 for screen projection of component centroids; kept
    // off the per-frame allocator to avoid GC spikes during overlay updates.
    this._projectionVec = new THREE.Vector3();

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.1, 500);
    this._defaultCameraPos = new THREE.Vector3(0, 28, 78);
    this._defaultTarget = new THREE.Vector3(0, 0, 0);
    this.camera.position.copy(this._defaultCameraPos);
    this.camera.lookAt(this._defaultTarget);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = themePreset.toneExposure;
    container.appendChild(this.renderer.domElement);

    this.renderPass = null;
    this.bloomPass = null;
    this.composer = null;
    this._baseOpacity = QUALITY_PRESETS.medium.baseOpacity;
    this._pulseUntil = new Map();
    this._pulseDurationMs = 520;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = true;
    this.controls.minDistance = 18;
    this.controls.maxDistance = 220;
    this.controls.target.copy(this._defaultTarget);
    this.controls.autoRotate = false;
    this.controls.autoRotateSpeed = 0.6;

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 1.2;
    this.mouseNdc = new THREE.Vector2();

    this.qualityLevel = "medium";
    this._applyQualityPreset(this.qualityLevel);

    this.gridW = 0;
    this.gridH = 0;
    this.nodeCount = 0;
    this.topologyMode = false;
    this.topologyVersion = -1;

    this.positions = null;
    this.colors = null;
    this.sizes = null;
    this.flash = null;
    this.alive = null;
    this.spikeMask = null;
    this.graphEdges = [];
    this.graphVersion = -1;
    // Reusable Box3 / Vector3 to avoid per-frame allocations.
    this._boundsBox = new THREE.Box3();
    this._boundsCenter = new THREE.Vector3();
    this._boundsSize = new THREE.Vector3();

    this.points = null;
    this.pointGeometry = null;
    this.pointMaterial = null;

    this.baseLineGeometry = null;
    this.baseLineSegments = null;
    this.baseEdgeCount = 0;

    this.activeLineGeometry = new THREE.BufferGeometry();
    this.activeLineSegments = null;
    this.maxActiveSegments = 12000;
    this.activeLinePositions = null;
    this.activeLineColors = null;
    this.cooldownByEdge = new Map();

    this._fpsRing = new Float32Array(30);
    this._fpsRingIdx = 0;
    this._lastFrameTime = performance.now();
    this._adaptiveActiveCap = 12000;

    this._bindEvents();

    this._initPostprocessing();
    this.resize();
  }

  _initPostprocessing() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.renderPass = new RenderPass(this.scene, this.camera);
    const res = new THREE.Vector2(Math.max(1, w), Math.max(1, h));
    const q = QUALITY_PRESETS[this.qualityLevel] ?? QUALITY_PRESETS.medium;
    this.bloomPass = new UnrealBloomPass(res, q.bloomStrength, q.bloomRadius, q.bloomThreshold);

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(this.renderPass);
    this.composer.addPass(this.bloomPass);

    const pr = Math.min(window.devicePixelRatio || 1, q.pixelRatioCap);
    if (typeof this.composer.setPixelRatio === "function") {
      this.composer.setPixelRatio(pr);
    }
  }

  /**
   * Tıklamada yerel parlama süresi; sonraki kare beklenmeden anında tepki için tampon yazılır.
   */
  pulseNode(index, durationMs) {
    if (!Number.isInteger(index) || index < 0) return;
    const ms = Number.isFinite(durationMs) ? durationMs : this._pulseDurationMs;
    this._pulseUntil.set(index, performance.now() + ms);
    if (!this.flash || index >= this.flash.length) {
      this._boostEdgesAroundIndex(index);
      return;
    }
    this.flash[index] = Math.max(this.flash[index], 1.0);
    const ga = this.pointGeometry?.attributes;
    if (ga?.aFlash) ga.aFlash.needsUpdate = true;
    const base = index * 3;
    const b = SPIKE_COLOR;
    if (this.colors && base + 2 < this.colors.length) {
      const t = 0.55;
      this.colors[base] = THREE.MathUtils.lerp(this.colors[base], b.r, t);
      this.colors[base + 1] = THREE.MathUtils.lerp(this.colors[base + 1], b.g, t);
      this.colors[base + 2] = THREE.MathUtils.lerp(this.colors[base + 2], b.b, t);
      if (ga?.color) ga.color.needsUpdate = true;
    }
    if (this.sizes && index < this.sizes.length) {
      this.sizes[index] = Math.max(this.sizes[index], 10.2);
      if (ga?.aSize) ga.aSize.needsUpdate = true;
    }
    this._boostEdgesAroundIndex(index);
  }

  /** Grafik üzerinden `index`'e bağlı kenarlarda kısa süre parlaklık sürdürümü */
  _boostEdgesAroundIndex(index) {
    if (!this.graphEdges.length) return;
    for (let e = 0; e < this.graphEdges.length; e++) {
      const [i, j] = this.graphEdges[e];
      if (i !== index && j !== index) continue;
      const key = indexKey(i, j);
      this.cooldownByEdge.set(key, Math.max(this.cooldownByEdge.get(key) ?? 0, 1.0));
    }
  }

  _applyQualityPreset(level) {
    const key = QUALITY_PRESETS[level] ? level : "medium";
    this.qualityLevel = key;
    const q = QUALITY_PRESETS[key];
    const pr = Math.min(q.pixelRatioCap, window.devicePixelRatio || 1);
    this.renderer.setPixelRatio(pr);
    this.maxActiveSegments = q.maxActiveSegments;
    this._adaptiveActiveCap = q.maxActiveSegments;
    this._baseEdgeStride = q.baseEdgeStride;
    this._haloStrength = q.haloStrength;
    this._baseOpacity = q.baseOpacity ?? 0.26;
    if (this.pointMaterial && this.pointMaterial.uniforms) {
      this.pointMaterial.uniforms.uHaloStrength.value = this._haloStrength;
    }
    if (this.bloomPass) {
      this.bloomPass.enabled = q.bloomStrength > 0.02;
      this.bloomPass.strength = q.bloomStrength;
      this.bloomPass.radius = q.bloomRadius;
      this.bloomPass.threshold = q.bloomThreshold;
    }
    if (this.baseLineSegments?.material) {
      this.baseLineSegments.material.opacity = this._baseOpacity;
      this.baseLineSegments.material.needsUpdate = true;
    }
    if (this.composer && typeof this.composer.setPixelRatio === "function") {
      this.composer.setPixelRatio(pr);
    }
  }

  setQuality(level) {
    this._applyQualityPreset(level);
    this.resize();
    if (this.graphEdges.length && this.positions) {
      this._rebuildBaseEdgesFromGraph();
    }
    this._reallocActiveBuffers();
  }

  setAutoRotate(enabled) {
    this.controls.autoRotate = !!enabled;
  }

  /**
   * Swap the scene chrome to match the requested UI theme. Cheap: only
   * mutates already-allocated Three.js color objects + a uniform on the
   * tone mapper. The base edge color is rebuilt only when a topology graph
   * is present so we skip a no-op buffer rebuild on the rectangular grid.
   */
  setTheme(themeName) {
    const next = themeName === "light" ? "light" : "dark";
    if (next === this.themeName) return;
    this.themeName = next;
    const preset = THEME_PRESETS[next];
    this.scene.background = new THREE.Color(preset.background);
    if (this.scene.fog) {
      this.scene.fog.color = new THREE.Color(preset.fogColor);
      this._applySceneFog();
    }
    this.renderer.toneMappingExposure = preset.toneExposure;
    this._currentBaseEdgeColor = preset.baseEdgeColor.clone();
    if (this.graphEdges?.length && this.positions) {
      this._rebuildBaseEdgesFromGraph();
    }
  }

  /**
   * Recompute ``scene.fog.density`` so the fog stays informative without
   * eating distant components.
   *
   * Math:
   *     ``FogExp2`` transmittance over distance ``d`` with density
   *     ``rho`` is ``exp(-rho * d)``. We want the FAR EDGE of the scene
   *     bbox (``_sceneExtent``) to keep at least ~30% transmittance so
   *     particles behind a component remain visible. Solving
   *     ``exp(-rho * extent) = 0.30`` gives ``rho = 1.2 / extent``.
   *
   * The result is clamped to the active theme's preset density (so a
   * tiny single-component scene does not get even denser fog than the
   * artist-tuned default) and to a small floor (so a huge multi-topology
   * scene still has *some* depth cue).
   */
  _applySceneFog() {
    if (!this.scene.fog) return;
    const preset = THEME_PRESETS[this.themeName] ?? THEME_PRESETS.dark;
    const extent = Math.max(this._sceneExtent || 1, 1);
    // 1.2 ≈ -ln(0.3); keep ~30% transmittance at the bbox edge so the
    // farthest topology still reads against the background.
    const extentDensity = 1.2 / extent;
    const target = Math.min(preset.fogDensity, extentDensity);
    // Floor stops the scene from looking flat (no atmospheric falloff).
    this.scene.fog.density = Math.max(0.0006, target);
  }

  /**
   * Project a world-space point onto the renderer's container client
   * rectangle. Returns ``null`` when the renderer has zero size or the
   * point falls outside the camera's clip space (so callers can hide an
   * overlay rather than parking it in the corner).
   *
   * Time:  O(1) – one matrix·vec and a couple of mults.
   */
  projectToContainer(point) {
    if (!Array.isArray(point) || point.length < 3) return null;
    const rect = this.renderer.domElement.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    this._projectionVec.set(
      Number(point[0]) || 0,
      Number(point[1]) || 0,
      Number(point[2]) || 0,
    );
    this._projectionVec.project(this.camera);
    const visible =
      this._projectionVec.z > -1
      && this._projectionVec.z < 1
      && this._projectionVec.x > -1.05
      && this._projectionVec.x < 1.05
      && this._projectionVec.y > -1.05
      && this._projectionVec.y < 1.05;
    return {
      x: (this._projectionVec.x * 0.5 + 0.5) * rect.width,
      y: (1 - (this._projectionVec.y * 0.5 + 0.5)) * rect.height,
      visible,
    };
  }

  resetCameraDefault() {
    this.camera.position.copy(this._defaultCameraPos);
    this.controls.target.copy(this._defaultTarget);
    this.controls.update();
  }

  _bindEvents() {
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    if (this.composer && typeof this.composer.setSize === "function") {
      this.composer.setSize(w, h);
    }
    if (this.bloomPass?.resolution) {
      this.bloomPass.resolution.set(Math.max(1, w), Math.max(1, h));
    }
  }

  _reallocActiveBuffers() {
    const cap = Math.max(256, Math.min(this._adaptiveActiveCap, 90000));
    const needLen = cap * 2 * 3;
    if (this.activeLinePositions && this.activeLinePositions.length === needLen) {
      this.maxActiveSegments = cap;
      return;
    }
    this.maxActiveSegments = cap;
    this.activeLinePositions = new Float32Array(cap * 2 * 3);
    this.activeLineColors = new Float32Array(cap * 2 * 3);
    this.activeLineGeometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.activeLinePositions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.activeLineGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(this.activeLineColors, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.activeLineGeometry.setDrawRange(0, 0);
  }

  _createOrganicPositions(width, height) {
    const n = width * height;
    const data = new Float32Array(n * 3);
    let ptr = 0;
    for (let i = 0; i < n; i++) {
      const gx = i % width;
      const gy = (i / width) | 0;
      const u = (gx / Math.max(1, width - 1)) * 2 - 1;
      const v = (gy / Math.max(1, height - 1)) * 2 - 1;
      const r = Math.sqrt(u * u + v * v);
      const theta = Math.atan2(v, u);
      const jitter = (Math.sin(i * 12.9898) * 43758.5453) % 1;
      const radius = 18 + r * 16 + jitter * 1.8;
      const x = Math.cos(theta) * radius + (Math.random() - 0.5) * 1.8;
      const y = Math.sin(theta) * radius + (Math.random() - 0.5) * 1.8;
      const z = (Math.sin(gx * 0.42) + Math.cos(gy * 0.36)) * 3.4 + (Math.random() - 0.5) * 2.4;
      data[ptr++] = x;
      data[ptr++] = y;
      data[ptr++] = z;
    }
    return data;
  }

  /**
   * Tear down every GPU-resident object the visualizer ever allocated.
   *
   * Critical on a 4 GB GTX 1650: switching topology pulls the rug under
   * potentially millions of vertex/edge bytes. Without explicit dispose()
   * Three.js leaks them to the WebGL context, and within a few swaps the
   * driver throws CONTEXT_LOST and the page goes blank.
   *
   * Call this BEFORE allocating new buffers in :meth:`_initPointCloud`.
   */
  _disposeNodeAndEdgeResources() {
    if (this.points) {
      this.scene.remove(this.points);
      this.points = null;
    }
    if (this.pointGeometry) {
      this.pointGeometry.dispose();
      this.pointGeometry = null;
    }
    if (this.pointMaterial) {
      this.pointMaterial.dispose();
      this.pointMaterial = null;
    }
    if (this.baseLineSegments) {
      this.scene.remove(this.baseLineSegments);
      this.baseLineSegments.material?.dispose?.();
      this.baseLineSegments = null;
    }
    if (this.baseLineGeometry) {
      this.baseLineGeometry.dispose();
      this.baseLineGeometry = null;
    }
    if (this.activeLineSegments) {
      this.scene.remove(this.activeLineSegments);
      this.activeLineSegments.material?.dispose?.();
      this.activeLineSegments = null;
    }
    if (this.activeLineGeometry) {
      this.activeLineGeometry.dispose();
    }
    // Drop CPU-side caches too: they'd otherwise pin RAM after a swap.
    this.positions = null;
    this.colors = null;
    this.sizes = null;
    this.flash = null;
    this.alive = null;
    this.spikeMask = null;
    this.graphEdges = [];
    this.graphVersion = -1;
    this.topologyVersion = -1;
    this.cooldownByEdge.clear();
    this._pulseUntil.clear();

    this.activeLineGeometry = new THREE.BufferGeometry();
    this.activeLinePositions = null;
    this.activeLineColors = null;
  }

  /**
   * Allocate every CPU-side Float32Array and BufferGeometry the renderer
   * needs. Reuses the *same* shader material and Points/Lines mesh
   * structure across both rectangular (small-world) and topology modes
   * so the cyan / gold / red visual rules stay consistent.
   */
  _initPointCloud(nodeCount, positions, opts = {}) {
    const { width = 0, height = 0, topology = false } = opts;
    this._disposeNodeAndEdgeResources();

    this.nodeCount = nodeCount;
    this.gridW = width;
    this.gridH = height;
    this.topologyMode = topology;

    this.positions =
      positions instanceof Float32Array
        ? positions
        : new Float32Array(positions);
    this.colors = new Float32Array(this.nodeCount * 3);
    this.sizes = new Float32Array(this.nodeCount);
    this.flash = new Float32Array(this.nodeCount);
    this.alive = new Uint8Array(this.nodeCount);
    this.spikeMask = new Uint8Array(this.nodeCount);

    this.pointGeometry = new THREE.BufferGeometry();
    const posAttr = new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage);
    const colAttr = new THREE.BufferAttribute(this.colors, 3).setUsage(THREE.DynamicDrawUsage);
    const sizeAttr = new THREE.BufferAttribute(this.sizes, 1).setUsage(THREE.DynamicDrawUsage);
    const flashAttr = new THREE.BufferAttribute(this.flash, 1).setUsage(THREE.DynamicDrawUsage);
    this.pointGeometry.setAttribute("position", posAttr);
    this.pointGeometry.setAttribute("color", colAttr);
    this.pointGeometry.setAttribute("aSize", sizeAttr);
    this.pointGeometry.setAttribute("aFlash", flashAttr);
    this.pointGeometry.computeBoundingSphere();

    this.pointMaterial = new THREE.ShaderMaterial({
      vertexShader: VERT_SHADER,
      fragmentShader: FRAG_SHADER,
      transparent: true,
      depthWrite: true,
      blending: THREE.NormalBlending,
      uniforms: {
        uHaloStrength: { value: this._haloStrength ?? 0.55 },
      },
    });

    this.points = new THREE.Points(this.pointGeometry, this.pointMaterial);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    this._reallocActiveBuffers();
    const activeMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.92,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });
    this.activeLineSegments = new THREE.LineSegments(this.activeLineGeometry, activeMat);
    this.activeLineSegments.renderOrder = 2;
    this.scene.add(this.activeLineSegments);

    this.graphVersion = -1;
    this.graphEdges = [];
  }

  /**
   * Initialise the rectangular (small-world / GoL) mode using the
   * existing organic-disk node placement. Kept as a thin wrapper so the
   * legacy frame format keeps working unchanged.
   */
  _initRectangularPointCloud(width, height) {
    const positions = this._createOrganicPositions(width, height);
    this._initPointCloud(width * height, positions, { width, height, topology: false });
  }

  /**
   * Initialise topology mode using the flat ``node_coords`` array sent by
   * the backend (3 floats per node, X/Y/Z). The renderer never allocates
   * per-vertex Vector3/Color objects: we copy straight into the Float32
   * positions attribute and let the GPU consume it.
   */
  _initTopologyPointCloud(nodeCount, flatCoords) {
    const positions = new Float32Array(nodeCount * 3);
    const src = flatCoords;
    const lim = Math.min(positions.length, src.length);
    for (let i = 0; i < lim; i++) positions[i] = src[i];
    this._initPointCloud(nodeCount, positions, { topology: true });
    this._fitCameraToScene();
  }

  /**
   * Compute an enclosing Box3 over current positions and move the camera
   * so the whole topology is visible without manual zoom.
   *
   * Algorithm:
   *   1. Walk the positions Float32Array once to compute min/max per axis.
   *   2. Use the larger of the bbox extents and the FOV to derive the
   *      required Z distance.
   *   3. Push the camera back along its current view direction.
   */
  _fitCameraToScene() {
    if (!this.positions || this.positions.length < 3) return;
    const box = this._boundsBox.makeEmpty();
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < this.positions.length; i += 3) {
      const x = this.positions[i];
      const y = this.positions[i + 1];
      const z = this.positions[i + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    box.set(
      new THREE.Vector3(minX, minY, minZ),
      new THREE.Vector3(maxX, maxY, maxZ),
    );
    box.getCenter(this._boundsCenter);
    box.getSize(this._boundsSize);

    const maxExtent = Math.max(
      this._boundsSize.x,
      this._boundsSize.y,
      this._boundsSize.z,
      1,
    );
    // Cache for fog auto-tune. Use the camera-target diagonal (not the
    // largest single axis) because FogExp2 attenuates along view distance,
    // so the worst-case visibility budget is the bbox diagonal.
    const diag = Math.sqrt(
      this._boundsSize.x * this._boundsSize.x
      + this._boundsSize.y * this._boundsSize.y
      + this._boundsSize.z * this._boundsSize.z,
    );
    this._sceneExtent = Math.max(maxExtent, diag, 1);
    this._applySceneFog();
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    // Standard fit-camera trig: distance so half-extent subtends half-FOV,
    // padded 1.4x to leave breathing room for bloom and labels.
    const distance = (maxExtent * 0.5) / Math.tan(fov * 0.5) * 1.4;

    const target = this._boundsCenter.clone();
    const dir = new THREE.Vector3(0.4, 0.55, 1).normalize();
    const eye = target.clone().add(dir.multiplyScalar(distance));

    this.camera.position.copy(eye);
    this.camera.near = Math.max(0.1, distance * 0.005);
    this.camera.far = distance * 5.0 + 1000;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(target);
    this.controls.minDistance = Math.max(8, distance * 0.05);
    this.controls.maxDistance = distance * 4.0;
    this.controls.update();

    // Persist the topology-fit camera as the new "reset" pose so the
    // "Görüşü sıfırla" button returns to the right view.
    this._defaultCameraPos = eye.clone();
    this._defaultTarget = target.clone();
  }

  _setGraph(graph) {
    if (!graph) return;
    const version = Number(graph.version ?? 0);
    const hasEdges = Array.isArray(graph.edges) && graph.edges.length > 0;
    // Only refresh when a fresh edge set arrives or the version differs.
    if (!hasEdges) {
      // First-time topology swap: backend may send empty edges if no
      // morphology has parents (degenerate). Keep the empty list to
      // avoid leaving stale edges from the previous topology on screen.
      if (version !== this.graphVersion) {
        this.graphVersion = version;
        this.graphEdges = [];
        this._rebuildBaseEdgesFromGraph();
      }
      return;
    }
    if (version === this.graphVersion && this.graphEdges.length) {
      return;
    }
    this.graphVersion = version;
    this.graphEdges = graph.edges
      .map((edge) => [Number(edge[0]), Number(edge[1])])
      .filter(
        (edge) =>
          Number.isInteger(edge[0]) &&
          Number.isInteger(edge[1]) &&
          edge[0] >= 0 &&
          edge[1] >= 0 &&
          edge[0] < this.nodeCount &&
          edge[1] < this.nodeCount &&
          edge[0] !== edge[1],
      );
    this._rebuildBaseEdgesFromGraph();
  }

  _rebuildBaseEdgesFromGraph() {
    if (this.baseLineSegments) {
      this.scene.remove(this.baseLineSegments);
      this.baseLineGeometry?.dispose();
      this.baseLineSegments = null;
      this.baseLineGeometry = null;
    }
    if (!this.graphEdges.length || !this.positions) return;

    const stride = this._baseEdgeStride ?? 1;
    const edges = [];
    for (let e = 0; e < this.graphEdges.length; e += stride) {
      edges.push(this.graphEdges[e]);
    }

    const segCount = edges.length;
    const pos = new Float32Array(segCount * 2 * 3);
    const col = new Float32Array(segCount * 2 * 3);
    const baseEdge = this._currentBaseEdgeColor ?? BASE_EDGE_COLOR;
    const br = baseEdge.r;
    const bg = baseEdge.g;
    const bb = baseEdge.b;
    const p = this.positions;

    for (let s = 0; s < segCount; s++) {
      const [i, j] = edges[s];
      const b = s * 6;
      const ai = i * 3;
      const bi = j * 3;
      pos[b] = p[ai];
      pos[b + 1] = p[ai + 1];
      pos[b + 2] = p[ai + 2];
      pos[b + 3] = p[bi];
      pos[b + 4] = p[bi + 1];
      pos[b + 5] = p[bi + 2];
      for (let k = 0; k < 6; k += 3) {
        col[b + k] = br;
        col[b + k + 1] = bg;
        col[b + k + 2] = bb;
      }
    }

    this.baseLineGeometry = new THREE.BufferGeometry();
    this.baseLineGeometry.setAttribute("position", new THREE.BufferAttribute(pos, 3).setUsage(THREE.StaticDrawUsage));
    this.baseLineGeometry.setAttribute("color", new THREE.BufferAttribute(col, 3).setUsage(THREE.StaticDrawUsage));
    this.baseEdgeCount = segCount;
    const baseMat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: this._baseOpacity ?? 0.26,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });
    this.baseLineSegments = new THREE.LineSegments(this.baseLineGeometry, baseMat);
    this.baseLineSegments.renderOrder = 0;
    this.scene.add(this.baseLineSegments);
  }

  _populateActiveLineSegments() {
    const cap = this.maxActiveSegments;
    let seg = 0;
    const nowKeys = new Set();
    const p = this.positions;
    const lp = this.activeLinePositions;
    const lc = this.activeLineColors;

    const addSegment = (a, b, color, alphaScale) => {
      if (seg >= cap) return;
      const key = indexKey(a, b);
      nowKeys.add(key);
      const base = seg * 6;
      const ai = a * 3;
      const bi = b * 3;
      lp[base] = p[ai];
      lp[base + 1] = p[ai + 1];
      lp[base + 2] = p[ai + 2];
      lp[base + 3] = p[bi];
      lp[base + 4] = p[bi + 1];
      lp[base + 5] = p[bi + 2];
      const r = color.r * alphaScale;
      const g = color.g * alphaScale;
      const bCol = color.b * alphaScale;
      lc[base] = r;
      lc[base + 1] = g;
      lc[base + 2] = bCol;
      lc[base + 3] = r;
      lc[base + 4] = g;
      lc[base + 5] = bCol;
      seg += 1;
    };

    for (let e = 0; e < this.graphEdges.length; e++) {
      const [i, j] = this.graphEdges[e];
      if (!this.alive[i] || !this.alive[j]) continue;
      const active = !!(this.spikeMask[i] || this.spikeMask[j]);
      if (!active) continue;
      const key = indexKey(i, j);
      this.cooldownByEdge.set(key, 1.0);
      addSegment(i, j, TRAIL_HOT, 1.0);
    }

    for (const [key, value] of this.cooldownByEdge.entries()) {
      const next = value - 0.06;
      if (next <= 0) {
        this.cooldownByEdge.delete(key);
        continue;
      }
      this.cooldownByEdge.set(key, next);
      if (nowKeys.has(key)) continue;
      const [aRaw, bRaw] = key.split(":");
      const a = Number(aRaw);
      const b = Number(bRaw);
      const c = TRAIL_HOT.clone().lerp(TRAIL_COOL, 1 - next);
      addSegment(a, b, c, next);
    }

    this.activeLineGeometry.setDrawRange(0, seg * 2);
    this.activeLineGeometry.attributes.position.needsUpdate = true;
    this.activeLineGeometry.attributes.color.needsUpdate = true;
  }

  applyFrame(frame) {
    const {
      width,
      height,
      n_nodes: frameNodes,
      gol_state: golState,
      spikes,
      graph,
      topology,
    } = frame;

    const useTopology = !!topology?.active;
    const desiredVersion = Number(topology?.version ?? -1);

    if (useTopology) {
      const incomingNodes = Number(
        topology?.n_nodes ?? frameNodes ?? (Array.isArray(golState) ? golState.length : 0),
      );
      const versionChanged = desiredVersion !== this.topologyVersion;
      const sizeChanged = !this.points || incomingNodes !== this.nodeCount || !this.topologyMode;
      if (Array.isArray(topology.node_coords) && topology.node_coords.length >= incomingNodes * 3) {
        if (versionChanged || sizeChanged) {
          this._initTopologyPointCloud(incomingNodes, topology.node_coords);
          this.topologyVersion = desiredVersion;
        }
      } else if (sizeChanged) {
        // Coords were not in this frame (cached on backend); keep the old
        // mesh until coords arrive on a heartbeat frame to avoid a flash.
        return;
      }
    } else if (!this.points || width !== this.gridW || height !== this.gridH || this.topologyMode) {
      this._initRectangularPointCloud(width, height);
    }
    this._setGraph(graph);

    const now = performance.now();
    const pulseDur = this._pulseDurationMs;
    for (const [pi, until] of [...this._pulseUntil.entries()]) {
      if (until <= now) this._pulseUntil.delete(pi);
    }

    for (let i = 0; i < this.nodeCount; i++) {
      const alive = golState[i] ? 1 : 0;
      const spk = spikes[i] ? 1 : 0;
      this.alive[i] = alive;
      this.spikeMask[i] = spk;

      let pulseAmp = 0;
      const pUntil = this._pulseUntil.get(i);
      if (pUntil != null && pUntil > now && pulseDur > 0) {
        const tRaw = (pUntil - now) / pulseDur;
        pulseAmp = tRaw * tRaw * (3 - 2 * tRaw);
      }
      let flashVal = Math.max(this.flash[i] * 0.88, spk ? 1.0 : 0);
      flashVal = Math.max(flashVal, pulseAmp * 1.08);
      this.flash[i] = flashVal;

      const base = i * 3;
      const baseC = alive ? ALIVE_COLOR : DEAD_COLOR;
      this.colors[base] = baseC.r;
      this.colors[base + 1] = baseC.g;
      this.colors[base + 2] = baseC.b;
      if (spk && alive) {
        this.colors[base] = SPIKE_COLOR.r;
        this.colors[base + 1] = SPIKE_COLOR.g;
        this.colors[base + 2] = SPIKE_COLOR.b;
      } else if (pulseAmp > 0.05) {
        const mixNk = pulseAmp * 0.82;
        const tw = pulseAmp * 0.72;
        const tgR = THREE.MathUtils.lerp(SPIKE_COLOR.r, NEON_BURST.r, mixNk);
        const tgG = THREE.MathUtils.lerp(SPIKE_COLOR.g, NEON_BURST.g, mixNk);
        const tgB = THREE.MathUtils.lerp(SPIKE_COLOR.b, NEON_BURST.b, mixNk);
        this.colors[base] = THREE.MathUtils.lerp(this.colors[base], tgR, tw);
        this.colors[base + 1] = THREE.MathUtils.lerp(this.colors[base + 1], tgG, tw);
        this.colors[base + 2] = THREE.MathUtils.lerp(this.colors[base + 2], tgB, tw);
      }
      let sz = alive ? (spk ? 9.8 : 5.2) : 3.0;
      if (pulseAmp > 0) sz += 2.8 + pulseAmp * 8.5;
      this.sizes[i] = Math.min(12.0, Math.max(2.4, sz));
    }

    this.pointGeometry.attributes.color.needsUpdate = true;
    this.pointGeometry.attributes.aSize.needsUpdate = true;
    this.pointGeometry.attributes.aFlash.needsUpdate = true;
    this._populateActiveLineSegments();
  }

  getFps() {
    let sum = 0;
    for (let i = 0; i < this._fpsRing.length; i++) sum += this._fpsRing[i];
    const avg = sum / this._fpsRing.length;
    return avg > 0 ? 1000 / avg : 0;
  }

  render() {
    const now = performance.now();
    const dt = now - this._lastFrameTime;
    this._lastFrameTime = now;
    this._fpsRing[this._fpsRingIdx % this._fpsRing.length] = Math.max(1, dt);
    this._fpsRingIdx++;

    const avgDt = Array.from(this._fpsRing).reduce((a, b) => a + b, 0) / this._fpsRing.length;
    const presetMax = QUALITY_PRESETS[this.qualityLevel]?.maxActiveSegments ?? 12000;
    if (avgDt > 22 && this._adaptiveActiveCap > 2000) {
      this._adaptiveActiveCap = Math.max(2000, (this._adaptiveActiveCap * 0.85) | 0);
      this.maxActiveSegments = this._adaptiveActiveCap;
      this._reallocActiveBuffers();
    } else if (avgDt < 14 && this._adaptiveActiveCap < presetMax) {
      this._adaptiveActiveCap = Math.min(presetMax, (this._adaptiveActiveCap * 1.05) | 0);
      this.maxActiveSegments = this._adaptiveActiveCap;
      this._reallocActiveBuffers();
    }

    this.controls.update();
    if (this.composer) {
      this.composer.render();
    } else {
      this.renderer.render(this.scene, this.camera);
    }
  }

  getHoveredIndex(clientX, clientY) {
    if (!this.points) return -1;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouseNdc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    this.mouseNdc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    this.raycaster.setFromCamera(this.mouseNdc, this.camera);
    const hits = this.raycaster.intersectObject(this.points);
    if (!hits.length) return -1;
    return hits[0].index ?? -1;
  }

  dispose() {
    this.controls.dispose();
    if (typeof this.composer?.dispose === "function") this.composer.dispose();
    this.renderPass = null;
    this.bloomPass = null;
    this.composer = null;
    this.renderer.dispose();
    this.pointGeometry?.dispose();
    this.pointMaterial?.dispose();
    this.activeLineSegments?.material?.dispose();
    this.activeLineGeometry?.dispose();
    this.baseLineSegments?.material?.dispose();
    this.baseLineGeometry?.dispose();
  }
}
