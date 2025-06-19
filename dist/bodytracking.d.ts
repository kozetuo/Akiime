import * as tf from '@tensorflow/tfjs-core';
import { GraphModel } from '@tensorflow/tfjs-converter/dist/executor/graph_model';
import { ImageSegmenter } from '@mediapipe/tasks-vision';

interface Point2D {
    x: number;
    y: number;
}
interface Point3D {
    x: number;
    y: number;
    z: number;
}
type Coord2D = [number, number];
type Coord3D = [number, number, number];
type Matrix3x3 = [Coord3D, Coord3D, Coord3D];
type Quaternion = [number, number, number, number];
interface Size {
    width: number;
    height: number;
}
interface Rect {
    xy: Point2D;
    size: Size;
}
type Box = [Coord2D, Coord2D];
type ImageInput = ImageData | ImageBitmap | HTMLCanvasElement;
declare function rectIoU(r0: Rect, r1: Rect): number;
declare function boxIoU(b0: Box, b1: Box): number;
declare function rectFromBox(b: Box): Rect;
declare function boxFromRect(r: Rect): Box;

interface BodyDetection {
    points: Coord2D[];
    box: Box;
    score: number;
}
interface BodyCircle {
    center: Coord2D;
    top: Coord2D;
}
declare class BodyDetector {
    private model;
    private modelSize;
    private modelRatio;
    private anchorsX;
    private anchorsY;
    private anchorsData;
    private posesMax;
    private iouThresh;
    private scoreThresh;
    constructor(model: GraphModel);
    process(image: tf.Tensor4D): Promise<BodyDetection[]>;
    decodeBoxes(pred: tf.Tensor2D, anchors: [tf.Tensor1D, tf.Tensor1D], size: Size): tf.Tensor2D;
    buildAnchors(size: Size): Point2D[];
    prepare(): Promise<void>;
    dispose(): void;
}

interface BodyMask {
    buffer: Float32Array | Uint8Array;
    size: Size;
    box: Box;
}
type MaskParam = boolean | {
    smooth?: boolean;
};
declare class BodySegmenter {
    private model;
    private modelSize;
    private modelRatio;
    private maskFilter?;
    constructor(model: GraphModel, smooth?: boolean);
    process(image: tf.Tensor4D, box: Box): BodyMask;
    resize(size: Size): void;
    size(): Size;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

interface PosePoint {
    pixel: Coord3D;
    metric: Coord3D;
    norm: Coord3D;
    score: number;
    visibility: number;
}
interface PoseDetection {
    keypoints: PosePoint[];
    score: number;
    center: Coord2D;
    top: Coord2D;
    mask?: BodyMask;
    debug?: {
        center: Coord2D;
        top: Coord2D;
        box: Box;
        radius: number;
        angle: number;
    };
}
declare class PoseDetector {
    private model;
    private mask;
    private modelSize;
    private sizeFactor;
    private maskFilter?;
    constructor(model: GraphModel, mask?: MaskParam);
    process(image: tf.Tensor4D, detections: BodyCircle[]): PoseDetection[];
    private refinePoints;
    private rotatedRect;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): Promise<void>;
}

interface FilterParams$1 {
    minCutOff: number;
    minCutOffD: number;
    beta: number;
}
declare class PoseFilter {
    protected freq: number;
    readonly pixelParams: FilterParams$1;
    readonly metricParams: FilterParams$1;
    readonly boxParams: FilterParams$1;
    readonly scoreCutOff = 1;
    readonly visibilityCutOff = 1;
    protected raw?: PoseDetection;
    protected smooth?: PoseDetection;
    protected der?: PoseDetection;
    protected time: number;
    filter(val: PoseDetection, time: number, scale?: number): PoseDetection;
    protected filterKeypoints(val: PosePoint[], raw: PosePoint[], der: PosePoint[], smooth: PosePoint[], scale: number): void;
    protected filterCoord3D(val: Coord3D, raw: Coord3D, der: Coord3D, smooth: Coord3D, scale: number, params: FilterParams$1): void;
    protected filterCoord2D(val: Coord2D, raw: Coord2D, der: Coord2D, smooth: Coord2D, scale: number, params: FilterParams$1): void;
    protected reset(): void;
    protected alpha(cutOff: number): number;
    protected clonePose(v: PoseDetection): PoseDetection;
}

declare class PoseTracker {
    protected bodyDetector?: BodyDetector;
    protected poseDetector?: PoseDetector;
    protected poseModule?: any;
    protected poseAligner?: any;
    protected bodyTracks: BodyCircle[];
    protected poseFilters: PoseFilter[];
    protected angle: number;
    protected ratio: number;
    protected near: number;
    readonly poseScore = 0.6;
    readonly alignScore = 0.9;
    readonly alignVisibility = 0.9;
    protected skipCount: number;
    readonly skipMax = 2;
    process(input: ImageInput, timestamp?: number): Promise<PoseDetection[]>;
    setCamera(angle: number, ratio: number, near?: number): void;
    init(token: string, root?: string, cache?: boolean, mask?: MaskParam, backend?: "webgl" | "cpu"): Promise<void>;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}
declare function estimateTranslation(points: {
    world: Coord3D;
    pixel: Coord3D;
}[], camera: {
    angle: number;
    ratio: number;
}): Coord3D;

interface FaceDetection {
    box: Box;
    score: number;
    keypoints?: Point2D[];
}
declare class FaceDetector {
    private model;
    private modelSize;
    private modelRatio;
    private anchors;
    private anchorsData;
    private size;
    private facesMax;
    private iouThresh;
    private scoreThresh;
    private keypointCount;
    constructor(model: GraphModel);
    process(image: tf.Tensor4D, keypoints?: boolean): Promise<FaceDetection[]>;
    decodeBoxes(pred: tf.Tensor2D, anchors: tf.Tensor2D, size: tf.Tensor1D): tf.Tensor2D;
    buildAnchors(size: Size): [number, number][];
    prepare(): Promise<void>;
    dispose(): void;
}
declare enum FacePoints {
    EyeR = 0,
    EyeL = 1,
    Nose = 2,
    Mouth = 3,
    EarR = 4,
    EarL = 5
}

interface MeshDetection {
    keypoints: Coord3D[];
    box: Box;
    score: number;
    mask?: BodyMask;
}
interface FaceRect {
    box: Box;
    symmetry: [Point2D, Point2D];
}
declare class MeshDetector {
    private model;
    private modelSize;
    private modelHighP;
    private boxFactor;
    symmetryPoints: number[];
    constructor(model: GraphModel);
    process(image: tf.Tensor4D, detections: FaceRect[]): MeshDetection[];
    private rotatedRect;
    prepare(): Promise<void>;
    dispose(): void;
}
declare function scaleMesh(mesh: MeshDetection, factor: number): MeshDetection;

interface FacePose {
    rotation: Quaternion;
    translation: Coord3D;
    scale: number;
    shapeScale: Coord3D;
}
declare class FaceTracker {
    protected faceDetector?: FaceDetector;
    protected meshDetector?: MeshDetector;
    protected bodySegmenter?: BodySegmenter;
    protected faceModule?: any;
    protected faceAligner?: any;
    protected faceTracks: FaceRect[];
    protected faceFilters: any[];
    readonly meshScore = 0.9;
    protected maskSize: number;
    protected maskExt?: number;
    process(input: ImageInput, timestamp?: number): Promise<MeshDetection[]>;
    align(model: Coord3D[]): FacePose | undefined;
    alignTransform(): FacePose | undefined;
    metricPoints(): Coord3D[] | undefined;
    referencePoints(): Coord3D[] | undefined;
    backprojPoints(): Coord3D[] | undefined;
    setCamera(angle: number, ratio: number, near: number): void;
    init(token: string, root?: string, cache?: boolean, highP?: boolean, mask?: MaskParam, maskExt?: number, maskSize?: 256 | 192 | 128, backend?: "webgl" | "cpu"): Promise<void>;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

interface PalmDetection {
    points: Coord2D[];
    box: Box;
    score: number;
}
interface PalmBox {
    box: Box;
    points: Coord2D[];
    start: Coord2D;
    end: Coord2D;
}
declare class PalmDetector {
    private model;
    private modelSize;
    private modelRatio;
    private anchorsX;
    private anchorsY;
    private anchorsData;
    private handsMax;
    private iouThresh;
    private scoreThresh;
    constructor(model: GraphModel);
    process(image: tf.Tensor4D): Promise<PalmDetection[]>;
    decodeBoxes(pred: tf.Tensor2D, anchors: [tf.Tensor1D, tf.Tensor1D], size: Size): tf.Tensor2D;
    buildAnchors(size: Size): Point2D[];
    prepare(): Promise<void>;
    dispose(): void;
}

interface HandPoint {
    pixel: Coord3D;
    metric: Coord3D;
}
interface PhalanxDetection {
    center: Coord3D;
    edges: [Coord3D, Coord3D];
}
interface WristLine {
    point: Coord2D;
    vector: Coord2D;
}
interface WristDetection {
    lines: WristLine[];
}
interface HandDetection {
    keypoints: HandPoint[];
    score: number;
    handedness: number;
    phalanxes: PhalanxDetection[];
    wrist?: WristDetection;
    debug?: any;
}
declare class HandDetector {
    private model;
    private wrist;
    private modelSize;
    private modelRatio;
    private backend?;
    private phalanxProg?;
    private wristDirProg?;
    private wristEdgeProg?;
    constructor(model: GraphModel, wrist?: boolean);
    process(image: tf.Tensor4D, box: PalmBox): HandDetection;
    private processPhalanxes;
    private processWrist;
    private fitLine;
    private fitLineRobust;
    private normalizeLines;
    private rotatedRect;
    prepare(): Promise<void>;
    dispose(): Promise<void>;
}

interface FilterParams {
    minCutOff: number;
    minCutOffD: number;
    beta: number;
}
declare class HandFilter {
    protected freq: number;
    readonly pixelParams: FilterParams;
    readonly metricParams: FilterParams;
    readonly phalanxParams: FilterParams;
    readonly scoreCutOff = 1;
    readonly handednessCutOff = 0.1;
    readonly visibilityCutOff = 1;
    protected raw?: HandDetection;
    protected smooth?: HandDetection;
    protected der?: HandDetection;
    protected time: number;
    filter(val: HandDetection, time: number, scale?: number): HandDetection;
    protected filterKeypoints(val: HandPoint[], raw: HandPoint[], der: HandPoint[], smooth: HandPoint[], scale: number): void;
    protected filterPhalanxes(val: PhalanxDetection[], raw: PhalanxDetection[], der: PhalanxDetection[], smooth: PhalanxDetection[], scale: number): void;
    protected filterWrist(val: HandDetection, raw: HandDetection, der: HandDetection, smooth: HandDetection, pixelD: Coord2D, scale: number): void;
    protected filterCoord3D(val: Coord3D, raw: Coord3D, der: Coord3D, smooth: Coord3D, scale: number, params: FilterParams): void;
    protected filterCoord2D(val: Coord2D, raw: Coord2D, der: Coord2D, smooth: Coord2D, scale: number, params: FilterParams): void;
    protected reset(): void;
    protected alpha(cutOff: number): number;
    protected clonePose(v: HandDetection): HandDetection;
}

declare class HandTracker {
    protected palmDetector?: PalmDetector;
    protected handDetector?: HandDetector;
    protected handTracks: PalmBox[];
    protected handFilters: HandFilter[];
    protected angle: number;
    protected ratio: number;
    protected near: number;
    readonly handScore = 0.55;
    process(input: ImageInput, timestamp?: number): Promise<HandDetection[]>;
    protected align(keypoints: HandPoint[]): void;
    protected correct(keypoints: HandPoint[]): void;
    setCamera(angle: number, ratio: number, near?: number): void;
    init(token: string, root?: string, cache?: boolean, wrist?: boolean, backend?: "webgl" | "cpu"): Promise<void>;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

declare class MaskTracker {
    protected segmenter?: BodySegmenter;
    process(input: ImageInput, timestamp?: number): Promise<BodyMask[]>;
    init(token: string, root?: string, cache?: boolean, smooth?: boolean, masksm?: boolean, backend?: "webgl" | "cpu"): Promise<void>;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

declare class HairSegmenter {
    private model;
    protected smooth: boolean;
    private modelSize;
    private timestamp;
    constructor(model: ImageSegmenter, smooth?: boolean);
    process(image: ImageInput, timestamp?: number): BodyMask | undefined;
    size(): Size;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

declare class HairTracker {
    protected segmenter?: HairSegmenter;
    process(input: ImageInput, timestamp?: number): Promise<BodyMask[]>;
    init(token: string, root?: string, cache?: boolean, smooth?: boolean): Promise<void>;
    reset(): void;
    prepare(): Promise<void>;
    dispose(): void;
}

declare const meshDesc: {
    [key: string]: number[];
};
declare const meshTriangles: number[];
declare const meshUV: Coord2D[];
declare const meshReference: Coord3D[];

declare class MeanColor {
    canvas?: HTMLCanvasElement;
    protected context: CanvasRenderingContext2D | null;
    constructor();
    mean(image: HTMLCanvasElement): number[] | undefined;
    brightness(image: HTMLCanvasElement): number | undefined;
    dispose(): void;
}

declare namespace LinearAlgebra {
    const add: (v0: Coord3D, v1: Coord3D) => Coord3D;
    const sub: (v0: Coord3D, v1: Coord3D) => Coord3D;
    const cross: (v0: Coord3D, v1: Coord3D) => Coord3D;
    const dot: (v0: Coord3D, v1: Coord3D) => number;
    const lerp: (v0: Coord3D, v1: Coord3D, k: number) => Coord3D;
    const scale: (v: Coord3D, s: number) => Coord3D;
    const scaleInPlace: (v: Coord3D, s: number) => Coord3D;
    const negate: (v: Coord3D) => Coord3D;
    const negateInPlace: (v: Coord3D) => Coord3D;
    const normalize: (v: Coord3D) => Coord3D;
    const normalizeInPlace: (v: Coord3D) => Coord3D;
    const normalizeToLen: (v: Coord3D, l: number) => Coord3D;
    const normalizeToLenInPlace: (v: Coord3D, l: number) => Coord3D;
    const lengthSqr: (v: Coord3D) => number;
    const length: (v: Coord3D) => number;
}

export { type BodyCircle, type BodyDetection, BodyDetector, type BodyMask, BodySegmenter, type Box, type Coord2D, type Coord3D, type FaceDetection, FaceDetector, FacePoints, type FacePose, type FaceRect, FaceTracker, HairTracker, type HandDetection, HandDetector, type HandPoint, HandTracker, type ImageInput, LinearAlgebra, type MaskParam, MaskTracker, type Matrix3x3, MeanColor, type MeshDetection, MeshDetector, type PalmBox, type PalmDetection, PalmDetector, type PhalanxDetection, type Point2D, type Point3D, type PoseDetection, PoseDetector, type PosePoint, PoseTracker, type Quaternion, type Rect, type Size, type WristDetection, type WristLine, boxFromRect, boxIoU, estimateTranslation, meshDesc, meshReference, meshTriangles, meshUV, rectFromBox, rectIoU, scaleMesh };
