export interface RecordingResult { blob: Blob; extension: string; }
export interface RecorderOptions { onError?: (error: Error) => void; mirrorCamera?: () => boolean; }

const WIDTH = 1080;
const HEIGHT = 1350;
const MIMES = ["video/mp4;codecs=avc1.42E01E", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];

/** Records a local, silent, two-panel camera/avatar composition. */
export class CompositeRecorder {
  private canvas: HTMLCanvasElement | null = null;
  private context: CanvasRenderingContext2D | null = null;
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private raf = 0;
  private chunks: Blob[] = [];
  private startedAt = 0;
  private failure: Error | null = null;
  private result: RecordingResult | null = null;
  private pendingStop: Promise<RecordingResult> | null = null;
  private resolveStop: ((result: RecordingResult) => void) | null = null;
  private rejectStop: ((error: Error) => void) | null = null;
  private frameCount = 0;

  constructor(private readonly avatar: HTMLCanvasElement, private readonly video: HTMLVideoElement, private readonly options: RecorderOptions = {}) {}

  get recording(): boolean {
    return this.recorder?.state === "recording";
  }

  get error(): Error | null { return this.failure; }

  start(): void {
    if (this.recorder) throw new Error("A recording is already in progress.");
    if (typeof MediaRecorder === "undefined" || typeof HTMLCanvasElement.prototype.captureStream !== "function") {
      throw new Error("Video recording is not supported in this browser. Try a current Safari or Chrome browser.");
    }
    this.failure = null;
    this.result = null;
    this.pendingStop = null;
    this.chunks = [];
    this.frameCount = 0;
    this.canvas = document.createElement("canvas");
    this.canvas.width = WIDTH;
    this.canvas.height = HEIGHT;
    this.context = this.canvas.getContext("2d", { alpha: false });
    if (!this.context) throw new Error("The recording canvas could not be created.");
    this.startedAt = performance.now();
    try {
      this.draw();
      this.stream = this.canvas.captureStream(30);
      let recorder: MediaRecorder | null = null;
      for (const mimeType of MIMES) {
        if (!MediaRecorder.isTypeSupported(mimeType)) continue;
        try {
          recorder = new MediaRecorder(this.stream, { mimeType, videoBitsPerSecond: 5_000_000 });
          break;
        } catch { /* A claimed codec can still be unavailable on this device. */ }
      }
      if (!recorder) throw new Error("This browser cannot encode an MP4 or WebM recording.");
      this.recorder = recorder;
      recorder.ondataavailable = (event: BlobEvent) => { if (event.data.size > 0) this.chunks.push(event.data); };
      recorder.onerror = () => this.fail(new Error("The browser stopped encoding the video. Try a shorter recording."));
      recorder.onstop = () => this.finish(recorder.mimeType);
      recorder.start(250);
      this.raf = requestAnimationFrame(this.drawFrame);
    } catch (error) {
      this.release();
      throw recordingError(error);
    }
  }

  stop(): Promise<RecordingResult> {
    if (this.result) return Promise.resolve(this.result);
    if (this.failure) return Promise.reject(this.failure);
    if (this.pendingStop) return this.pendingStop;
    const recorder = this.recorder;
    if (!recorder) return Promise.reject(new Error("There is no recording to save."));
    this.pendingStop = new Promise<RecordingResult>((resolve, reject) => { this.resolveStop = resolve; this.rejectStop = reject; });
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (recorder.state !== "inactive") {
      try { recorder.stop(); } catch (error) { this.fail(recordingError(error)); }
    }
    return this.pendingStop;
  }

  cancel(): void {
    this.rejectStop?.(new Error("Recording cancelled."));
    this.resolveStop = null;
    this.rejectStop = null;
    this.pendingStop = null;
    this.result = null;
    this.chunks = [];
    this.release();
  }

  dispose(): void { this.cancel(); }

  private drawFrame = (): void => {
    if (!this.recording) return;
    try {
      this.draw();
      this.raf = requestAnimationFrame(this.drawFrame);
    } catch (error) {
      this.fail(recordingError(error));
    }
  };

  private draw(): void {
    const ctx = this.context;
    if (!ctx) return;
    ctx.fillStyle = "#08090b";
    ctx.fillRect(0, 0, WIDTH, HEIGHT);
    ctx.fillStyle = "#f7f7f2";
    ctx.font = "700 42px Arial, sans-serif";
    ctx.fillText("AKIIME", 38, 66);
    ctx.fillStyle = "#a4a5ac";
    ctx.font = "500 18px Arial, sans-serif";
    ctx.fillText("SELFIE STUDIO", 39, 99);
    ctx.fillStyle = "#ff6b4a";
    ctx.beginPath();
    ctx.arc(943, 51, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = "#d8d8df";
    ctx.font = "500 21px Arial, sans-serif";
    const seconds = Math.max(0, Math.floor((performance.now() - this.startedAt) / 1000));
    ctx.fillText(`${Math.floor(seconds / 60).toString().padStart(2, "0")}:${(seconds % 60).toString().padStart(2, "0")}`, 963, 59);
    this.drawPane(32, 151, 496, 1115, "Your camera", true);
    this.drawPane(552, 151, 496, 1115, "Your avatar", false);
    ctx.fillStyle = "#74757f";
    ctx.font = "400 18px Arial, sans-serif";
    ctx.fillText("Made with Akiime", 38, 1311);
    // Catch a canvas becoming tainted after an avatar/outfit change, too.
    if (this.frameCount++ % 30 === 0) ctx.getImageData(0, 0, 1, 1);
  }

  private drawPane(x: number, y: number, width: number, height: number, label: string, camera: boolean): void {
    const ctx = this.context!;
    ctx.fillStyle = "#c6c7ce";
    ctx.font = "500 20px Arial, sans-serif";
    ctx.fillText(label, x + 2, y - 20);
    ctx.save();
    roundedRect(ctx, x, y, width, height, 22);
    ctx.clip();
    ctx.fillStyle = camera ? "#1b2119" : "#e5e7e0";
    ctx.fillRect(x, y, width, height);
    if (camera) {
      if (this.video.readyState >= 2 && this.video.videoWidth > 0 && this.video.videoHeight > 0) {
        const scale = Math.max(width / this.video.videoWidth, height / this.video.videoHeight);
        const drawWidth = this.video.videoWidth * scale;
        const drawHeight = this.video.videoHeight * scale;
        const mirror = this.options.mirrorCamera?.() ?? true;
        ctx.translate(x + (mirror ? width : 0), y);
        if (mirror) ctx.scale(-1, 1);
        ctx.drawImage(this.video, (width - drawWidth) / 2, (height - drawHeight) / 2, drawWidth, drawHeight);
      } else {
        ctx.fillStyle = "#92949e";
        ctx.font = "400 24px Arial, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("Camera off", x + width / 2, y + height / 2);
        ctx.textAlign = "left";
      }
    } else if (this.avatar.width > 0 && this.avatar.height > 0) {
      const scale = Math.min(width / this.avatar.width, height / this.avatar.height);
      const drawWidth = this.avatar.width * scale;
      const drawHeight = this.avatar.height * scale;
      ctx.drawImage(this.avatar, x + (width - drawWidth) / 2, y + (height - drawHeight) / 2, drawWidth, drawHeight);
    }
    ctx.restore();
  }

  private finish(mime: string): void {
    if (!this.failure) {
      const type = mime || this.chunks[0]?.type || "video/webm";
      const blob = new Blob(this.chunks, { type });
      if (blob.size === 0) {
        this.fail(new Error("The recording was empty. Record for a moment, then try saving again."));
        return;
      }
      this.result = { blob, extension: type.includes("mp4") ? "mp4" : "webm" };
      this.resolveStop?.(this.result);
    }
    this.resolveStop = null;
    this.rejectStop = null;
    this.chunks = [];
    this.release();
  }

  private fail(error: Error): void {
    if (this.failure) return;
    this.failure = error;
    this.rejectStop?.(error);
    this.resolveStop = null;
    this.rejectStop = null;
    this.chunks = [];
    this.release();
    this.options.onError?.(error);
  }

  private release(): void {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    if (this.recorder) {
      this.recorder.ondataavailable = null;
      this.recorder.onstop = null;
      this.recorder.onerror = null;
      if (this.recorder.state !== "inactive") {
        try { this.recorder.stop(); } catch { /* Already stopped by the browser. */ }
      }
    }
    this.recorder = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.context = null;
    this.canvas = null;
  }
}

function roundedRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number): void {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.lineTo(x + width - radius, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
  ctx.lineTo(x + width, y + height - radius);
  ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
  ctx.lineTo(x + radius, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
  ctx.lineTo(x, y + radius);
  ctx.quadraticCurveTo(x, y, x + radius, y);
  ctx.closePath();
}

function recordingError(error: unknown): Error {
  if (error instanceof Error && error.name === "SecurityError") {
    return new Error("This avatar contains an image that cannot be recorded across origins. Use same-origin assets or enable CORS on the image host.");
  }
  return error instanceof Error ? error : new Error("The video could not be recorded. Please try again.");
}
