export type MotionStartResult = "active" | "unsupported" | "denied";
export interface MotionRotation { x: number; y: number; z: number; }

type Quaternion = { x: number; y: number; z: number; w: number };
type PermissionOrientationConstructor = typeof DeviceOrientationEvent & {
  requestPermission?: () => Promise<"granted" | "denied">;
};
const identity = (): Quaternion => ({ x: 0, y: 0, z: 0, w: 1 });
const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));
const radians = (degrees: number): number => degrees * Math.PI / 180;

/**
 * Optional, bounded artistic phone control, not positional hand tracking.
 * Call start() directly from a button click so iOS can request permission.
 * The first valid sample establishes neutral. Read rotation every render frame.
 */
export class MotionController {
  private listening = false;
  private generation = 0;
  private sample: { alpha: number; beta: number; gamma: number } | null = null;
  private current: Quaternion | null = null;
  private origin: Quaternion | null = null;
  private smoothed = identity();
  private lastReadAt = 0;
  private timeout: ReturnType<typeof setTimeout> | null = null;
  private resolveStart: ((result: MotionStartResult) => void) | null = null;
  private starting: Promise<MotionStartResult> | null = null;

  get active(): boolean {
    return this.listening && this.current !== null;
  }

  async start(): Promise<MotionStartResult> {
    if (this.active) return "active";
    if (this.starting) return this.starting;
    if (typeof window === "undefined" || !window.isSecureContext || typeof window.DeviceOrientationEvent === "undefined") {
      return "unsupported";
    }
    const generation = this.generation;
    const constructor = window.DeviceOrientationEvent as PermissionOrientationConstructor;
    if (typeof constructor.requestPermission === "function") {
      try {
        if (await constructor.requestPermission() !== "granted") return "denied";
      } catch {
        return "denied";
      }
    }
    if (generation !== this.generation) return "unsupported";
    this.listening = true;
    this.lastReadAt = performance.now();
    this.starting = new Promise<MotionStartResult>((resolve) => {
      this.resolveStart = resolve;
      // A desktop can expose the API while providing no sensor readings.
      this.timeout = setTimeout(() => this.stop(), 2200);
      window.addEventListener("deviceorientation", this.onOrientation);
      window.addEventListener("orientationchange", this.onScreenChange);
      window.screen.orientation?.addEventListener("change", this.onScreenChange);
    });
    return this.starting;
  }

  private onOrientation = (event: DeviceOrientationEvent): void => {
    if (!this.listening || event.beta === null || event.gamma === null || !Number.isFinite(event.beta) || !Number.isFinite(event.gamma)) return;
    this.sample = { alpha: Number.isFinite(event.alpha) ? event.alpha! : 0, beta: event.beta, gamma: event.gamma };
    this.current = this.deviceQuaternion();
    if (!this.origin) this.origin = { ...this.current };
    if (this.timeout !== null) clearTimeout(this.timeout);
    this.timeout = null;
    this.resolveStart?.("active");
    this.resolveStart = null;
    this.starting = null;
  };

  private onScreenChange = (): void => {
    if (!this.sample) return;
    // Reset neutral when portrait/landscape changes to avoid an arm snap.
    this.current = this.deviceQuaternion();
    this.calibrate();
  };

  private deviceQuaternion(): Quaternion {
    if (!this.sample) return identity();
    const { alpha, beta, gamma } = this.sample;
    const device = multiply(multiply(axis("z", radians(alpha)), axis("x", radians(beta))), axis("y", radians(gamma)));
    const legacyAngle = (window as Window & { orientation?: number }).orientation;
    const screenAngle = window.screen.orientation?.angle ?? (typeof legacyAngle === "number" ? legacyAngle : 0);
    return multiply(device, axis("z", -radians(screenAngle)));
  }

  get rotation(): MotionRotation | null {
    if (!this.active || !this.current || !this.origin) return null;
    const now = performance.now();
    const dt = clamp((now - this.lastReadAt) / 1000, 0.001, 0.1);
    this.lastReadAt = now;
    const inverse = { x: -this.origin.x, y: -this.origin.y, z: -this.origin.z, w: this.origin.w };
    const target = multiply(inverse, this.current);
    this.smoothed = slerp(this.smoothed, target, 1 - Math.exp(-dt / 0.12));
    const euler = toEuler(this.smoothed);
    // Deliberately modest ranges protect the selfie grip and baked sleeves.
    return { x: clamp(euler.x, -0.45, 0.45), y: clamp(euler.y, -0.5, 0.5), z: clamp(euler.z, -0.45, 0.45) };
  }

  calibrate(): boolean {
    if (!this.current) return false;
    this.origin = { ...this.current };
    this.smoothed = identity();
    this.lastReadAt = performance.now();
    return true;
  }

  stop(): void {
    this.generation += 1;
    if (typeof window !== "undefined") {
      window.removeEventListener("deviceorientation", this.onOrientation);
      window.removeEventListener("orientationchange", this.onScreenChange);
      window.screen.orientation?.removeEventListener("change", this.onScreenChange);
    }
    if (this.timeout !== null) clearTimeout(this.timeout);
    this.timeout = null;
    this.resolveStart?.("unsupported");
    this.resolveStart = null;
    this.starting = null;
    this.listening = false;
    this.sample = null;
    this.current = null;
    this.origin = null;
    this.smoothed = identity();
  }
}

function axis(name: "x" | "y" | "z", angle: number): Quaternion {
  return { x: name === "x" ? Math.sin(angle / 2) : 0, y: name === "y" ? Math.sin(angle / 2) : 0, z: name === "z" ? Math.sin(angle / 2) : 0, w: Math.cos(angle / 2) };
}

function multiply(a: Quaternion, b: Quaternion): Quaternion {
  return {
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

function slerp(a: Quaternion, b: Quaternion, amount: number): Quaternion {
  let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  // q and -q encode the same rotation; choose the shorter arc.
  const target = dot < 0 ? { x: -b.x, y: -b.y, z: -b.z, w: -b.w } : b;
  dot = Math.abs(dot);
  let from = 1 - amount;
  let to = amount;
  if (dot < 0.9995) {
    const angle = Math.acos(clamp(dot, -1, 1));
    const denominator = Math.sin(angle);
    from = Math.sin((1 - amount) * angle) / denominator;
    to = Math.sin(amount * angle) / denominator;
  }
  const result = { x: from * a.x + to * target.x, y: from * a.y + to * target.y, z: from * a.z + to * target.z, w: from * a.w + to * target.w };
  const length = Math.hypot(result.x, result.y, result.z, result.w) || 1;
  return { x: result.x / length, y: result.y / length, z: result.z / length, w: result.w / length };
}

function toEuler(q: Quaternion): MotionRotation {
  const m13 = 2 * (q.x * q.z + q.y * q.w);
  const y = Math.asin(clamp(m13, -1, 1));
  if (Math.abs(m13) < 0.99999) {
    return { x: Math.atan2(2 * (q.x * q.w - q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)), y, z: Math.atan2(2 * (q.z * q.w - q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)) };
  }
  return { x: Math.atan2(2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.z * q.z)), y, z: 0 };
}
