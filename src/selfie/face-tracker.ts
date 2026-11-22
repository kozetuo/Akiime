import type { FaceLandmarker, FaceLandmarkerResult } from "@mediapipe/tasks-vision";

export interface FaceRotation {
  /** XYZ Euler angles in radians, relative to the last calibration. */
  x: number;
  y: number;
  z: number;
}

export interface FaceFrame {
  /** MediaPipe/ARKit expression names; left and right refer to the subject. */
  blendshapes: Record<string, number>;
  rotation: FaceRotation;
  /** Smoothed presence indicator, not a detector probability/accuracy score. */
  confidence: number;
  detected: boolean;
  timestamp: number;
}

export interface FaceTrackerOptions {
  /** Keep this WASM version aligned with the installed tasks-vision package. */
  wasmRoot?: string;
  modelUrl?: string;
  maxFps?: number;
  delegate?: "GPU" | "CPU";
}

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const neutralRotation = (): FaceRotation => ({ x: 0, y: 0, z: 0 });
const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));
const angleDifference = (angle: number, origin: number): number => Math.atan2(Math.sin(angle - origin), Math.cos(angle - origin));

/**
 * A camera-independent face tracker. Call load() once, then update(video,
 * performance.now()) from the render loop. null means no new inference frame.
 * Pass the original video, not a mirrored canvas; mirror its UI with CSS.
 * No camera frames leave the browser. The initial load downloads model/WASM.
 */
export class FaceTracker {
  private landmarker: FaceLandmarker | null = null;
  private loading: Promise<void> | null = null;
  private generation = 0;
  private lastVideoTime = -1;
  private lastFreshVideoAt = 0;
  private lastUpdateAt = -Infinity;
  private inferenceClock = -Infinity;
  private blendshapes: Record<string, number> = {};
  private rotation = neutralRotation();
  private rawRotation = neutralRotation();
  private origin = neutralRotation();
  private presence = 0;
  private hasDetectedFace = false;
  private readonly interval: number;

  constructor(private readonly options: FaceTrackerOptions = {}) {
    this.interval = 1000 / clamp(options.maxFps ?? 24, 1, 60);
  }

  get ready(): boolean {
    return this.landmarker !== null;
  }

  async load(): Promise<void> {
    if (this.landmarker) return;
    if (this.loading) return this.loading;
    const generation = this.generation;
    this.loading = this.createLandmarker(generation);
    try {
      await this.loading;
    } finally {
      if (generation === this.generation) this.loading = null;
    }
  }

  private async createLandmarker(generation: number): Promise<void> {
    const { FaceLandmarker: Landmarker, FilesetResolver } = await import("@mediapipe/tasks-vision");
    const fileset = await FilesetResolver.forVisionTasks(this.options.wasmRoot ?? WASM_ROOT);
    if (generation !== this.generation) return;
    const create = (delegate: "GPU" | "CPU") => Landmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: this.options.modelUrl ?? MODEL_URL, delegate },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    let tracker: FaceLandmarker;
    try {
      tracker = await create(this.options.delegate ?? "GPU");
    } catch (error) {
      if (generation !== this.generation) return;
      if (this.options.delegate === "CPU") throw error;
      // Some mobile/WebGL configurations cannot create the GPU delegate.
      // Retry with the same model on CPU, rather than leaving the app unusable.
      try {
        tracker = await create("CPU");
      } catch (fallbackError) {
        const detail = fallbackError instanceof Error ? fallbackError.message : String(fallbackError);
        throw new Error(`Face tracking could not load. Check your connection and try again. ${detail}`);
      }
    }
    if (generation !== this.generation) {
      tracker.close();
      return;
    }
    this.landmarker = tracker;
    this.inferenceClock = -Infinity;
    this.reset();
  }

  update(video: HTMLVideoElement, timestamp: number): FaceFrame | null {
    if (!this.landmarker) return null;
    if (!Number.isFinite(timestamp)) return null;
    const elapsed = timestamp - this.lastUpdateAt;
    if (elapsed >= 0 && elapsed < this.interval) return null;

    const hasVideo = video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
    const freshVideo = hasVideo && video.currentTime !== this.lastVideoTime;
    if (!freshVideo && timestamp - this.lastFreshVideoAt < 300) return null;

    let result: FaceLandmarkerResult | null = null;
    if (freshVideo) {
      this.lastVideoTime = video.currentTime;
      this.lastFreshVideoAt = timestamp;
      // MediaPipe requires monotonic timestamps even across camera restarts.
      this.inferenceClock = Math.max(timestamp, this.inferenceClock + 0.01);
      result = this.landmarker.detectForVideo(video, this.inferenceClock);
    }

    const dt = Number.isFinite(this.lastUpdateAt) ? clamp(elapsed / 1000, 0.001, 0.2) : 1 / 24;
    this.lastUpdateAt = timestamp;
    const detected = Boolean(result?.faceLandmarks[0]?.length);
    this.hasDetectedFace = detected;
    const targets: Record<string, number> = {};
    if (detected && result) {
      for (const category of result.faceBlendshapes[0]?.categories ?? []) {
        if (category.categoryName !== "_neutral") {
          targets[category.categoryName] = clamp(category.score, 0, 1);
        }
      }
      const matrix = result.facialTransformationMatrixes[0];
      if (matrix && matrix.rows === 4 && matrix.columns === 4) {
        this.rawRotation = rotationFromMatrix(matrix.data);
      }
    }

    const values: Record<string, number> = {};
    const names = new Set([...Object.keys(this.blendshapes), ...Object.keys(targets)]);
    for (const name of names) {
      // Preserve fast blinks while smoothing the slower expression channels.
      const timeConstant = detected ? (name.startsWith("eyeBlink") ? 0.025 : 0.065) : 0.15;
      const alpha = 1 - Math.exp(-dt / timeConstant);
      values[name] = clamp((this.blendshapes[name] ?? 0) + ((targets[name] ?? 0) - (this.blendshapes[name] ?? 0)) * alpha, 0, 1);
    }
    this.blendshapes = values;

    const targetRotation = detected ? {
      x: clamp(angleDifference(this.rawRotation.x, this.origin.x), -0.8, 0.8),
      y: clamp(angleDifference(this.rawRotation.y, this.origin.y), -1.05, 1.05),
      z: clamp(angleDifference(this.rawRotation.z, this.origin.z), -0.8, 0.8),
    } : neutralRotation();
    const alpha = 1 - Math.exp(-dt / (detected ? 0.085 : 0.2));
    this.rotation = {
      x: this.rotation.x + (targetRotation.x - this.rotation.x) * alpha,
      y: this.rotation.y + (targetRotation.y - this.rotation.y) * alpha,
      z: this.rotation.z + (targetRotation.z - this.rotation.z) * alpha,
    };
    this.presence += ((detected ? 1 : 0) - this.presence) * alpha;
    return {
      blendshapes: { ...this.blendshapes },
      rotation: { ...this.rotation },
      confidence: this.presence,
      detected,
      timestamp,
    };
  }

  /** Make the current tracked head pose the avatar's neutral direction. */
  calibrate(): boolean {
    if (!this.hasDetectedFace) return false;
    this.origin = { ...this.rawRotation };
    this.rotation = neutralRotation();
    return true;
  }

  /** Clear expression/calibration state without re-downloading the model. */
  reset(): void {
    this.blendshapes = {};
    this.rotation = neutralRotation();
    this.rawRotation = neutralRotation();
    this.origin = neutralRotation();
    this.presence = 0;
    this.hasDetectedFace = false;
    this.lastVideoTime = -1;
    this.lastFreshVideoAt = 0;
    this.lastUpdateAt = -Infinity;
  }

  /** Release model/GPU resources. This does not stop root-owned camera tracks. */
  dispose(): void {
    this.generation += 1;
    this.landmarker?.close();
    this.landmarker = null;
    this.loading = null;
    this.reset();
  }
}

/** MediaPipe exposes the face transform as a flattened column-major matrix. */
function rotationFromMatrix(matrix: number[]): FaceRotation {
  if (matrix.length !== 16 || matrix.some((value) => !Number.isFinite(value))) return neutralRotation();
  // Remove scale from the three basis vectors before extracting XYZ Euler angles.
  const sx = Math.hypot(matrix[0], matrix[1], matrix[2]) || 1;
  const sy = Math.hypot(matrix[4], matrix[5], matrix[6]) || 1;
  const sz = Math.hypot(matrix[8], matrix[9], matrix[10]) || 1;
  const m11 = matrix[0] / sx;
  const m12 = matrix[4] / sy;
  const m13 = matrix[8] / sz;
  const m22 = matrix[5] / sy;
  const m23 = matrix[9] / sz;
  const m32 = matrix[6] / sy;
  const m33 = matrix[10] / sz;
  const y = Math.asin(clamp(m13, -1, 1));
  return Math.abs(m13) < 0.99999 ? {
    x: Math.atan2(-m23, m33),
    y,
    z: Math.atan2(-m12, m11),
  } : { x: Math.atan2(m32, m22), y, z: 0 };
}
