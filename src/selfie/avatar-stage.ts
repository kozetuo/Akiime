import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import type { FaceFrame } from "./face-tracker";

export type OutfitColor = "ink" | "chalk" | "oxide";
export interface GyroRotation { x: number; y: number; z: number }

const clamp = THREE.MathUtils.clamp;
const Y_AXIS = new THREE.Vector3(0, 1, 0);
const OUTFITS: Record<OutfitColor, { fabric: number; rib: number; pants: number; print: string }> = {
  ink: { fabric: 0x252932, rib: 0x1a1d24, pants: 0x42464c, print: "#e6e8d9" },
  chalk: { fabric: 0xd9d7cd, rib: 0xbcbab0, pants: 0x797d77, print: "#343b35" },
  oxide: { fabric: 0x9d493a, rib: 0x79372e, pants: 0x302f34, print: "#f3dfc4" },
};

/**
 * An original, procedural preview character. It needs no downloaded character,
 * does not claim to contain a Daz asset or baked cloth, and remains usable while
 * the camera/face model is loading. Root owns the render loop and camera stream.
 */
export class AvatarStage {
  readonly renderer: THREE.WebGLRenderer;
  readonly canvas: HTMLCanvasElement;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(33, 1, 0.02, 30);

  private readonly controls: OrbitControls;
  private readonly resizeObserver: ResizeObserver;
  private readonly character = new THREE.Group();
  private readonly torso = new THREE.Group();
  private readonly head = new THREE.Group();
  private readonly hips = new THREE.Group();
  private readonly phoneHand = new THREE.Group();
  private readonly leftArm = new THREE.Group();
  private readonly rightArm = new THREE.Group();
  private readonly rightUpper: THREE.Mesh;
  private readonly rightLower: THREE.Mesh;
  private readonly rightElbow: THREE.Mesh;
  private readonly rightCuff: THREE.Mesh;
  private readonly shirt: THREE.Mesh<THREE.LatheGeometry, THREE.MeshStandardMaterial>;
  private readonly shirtRest: Float32Array;
  private readonly leftEye = new THREE.Group();
  private readonly rightEye = new THREE.Group();
  private readonly leftPupil: THREE.Mesh;
  private readonly rightPupil: THREE.Mesh;
  private readonly leftBrow = new THREE.Group();
  private readonly rightBrow = new THREE.Group();
  private readonly mouthInterior: THREE.Mesh;
  private readonly lips: THREE.Mesh;
  private readonly teeth: THREE.Mesh;
  private readonly tongue: THREE.Mesh;
  private readonly chestLabel: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial>;
  private labelTexture: THREE.CanvasTexture;
  private readonly fabric = new THREE.MeshStandardMaterial({ color: OUTFITS.ink.fabric, roughness: 0.93 });
  private readonly rib = new THREE.MeshStandardMaterial({ color: OUTFITS.ink.rib, roughness: 0.97 });
  private readonly pants = new THREE.MeshStandardMaterial({ color: OUTFITS.ink.pants, roughness: 0.94 });
  private readonly skin = new THREE.MeshStandardMaterial({ color: 0xa86e50, roughness: 0.7 });
  private readonly shoe = new THREE.MeshStandardMaterial({ color: 0xe5e1d6, roughness: 0.7 });
  private readonly shoeSole = new THREE.MeshStandardMaterial({ color: 0xb9b8ac, roughness: 0.86 });
  private readonly dark = new THREE.MeshStandardMaterial({ color: 0x171b20, roughness: 0.68 });
  private readonly hair = new THREE.MeshStandardMaterial({ color: 0x201b18, roughness: 0.96 });
  private readonly mouthMaterial = new THREE.MeshStandardMaterial({ color: 0x331919, roughness: 0.88 });
  private readonly lipMaterial = new THREE.MeshStandardMaterial({ color: 0x815048, roughness: 0.75 });
  private gyroTarget = new THREE.Vector3();
  private gyro = new THREE.Vector3();
  private readonly shoulder = new THREE.Vector3(-0.34, 1.49, 0.015);
  private readonly elbow = new THREE.Vector3();
  private readonly wrist = new THREE.Vector3();
  private readonly temp = new THREE.Vector3();
  private readonly headTarget = new THREE.Vector3();
  private motionEnabled = true;
  private view: "full" | "portrait" = "full";
  private disposed = false;
  private faceValues: Record<string, number> = {};
  private lastFace: FaceFrame | null = null;
  private lastFaceAt = -Infinity;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    this.canvas = this.renderer.domElement;
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", "Animated full-body character holding a phone. Drag to turn the view; pinch or scroll to zoom.");
    this.canvas.style.display = "block";
    this.canvas.style.width = "100%";
    this.canvas.style.height = "100%";
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.16;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(this.canvas);

    const ambient = new THREE.HemisphereLight(0xf3f3ff, 0xa29c8c, 2.65);
    this.scene.add(ambient);
    const key = new THREE.DirectionalLight(0xfff1df, 3.7);
    key.position.set(-2.5, 4.8, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.left = -1.5;
    key.shadow.camera.right = 1.5;
    key.shadow.camera.top = 2.8;
    key.shadow.camera.bottom = -0.5;
    key.shadow.normalBias = 0.015;
    key.shadow.bias = -0.0002;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xe1e6ff, 2.1);
    rim.position.set(2.8, 3.2, -2.2);
    this.scene.add(rim);

    const platform = this.mesh(new THREE.CylinderGeometry(0.64, 0.67, 0.035, 72), new THREE.MeshStandardMaterial({ color: 0xd0d1cb, roughness: 0.9 }));
    platform.position.y = -0.025;
    platform.receiveShadow = true;
    this.scene.add(platform);
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.ShadowMaterial({ color: 0x17181c, opacity: 0.14 }));
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.006;
    floor.receiveShadow = true;
    this.scene.add(floor);

    this.scene.add(this.character);
    this.character.add(this.hips, this.torso, this.head, this.leftArm, this.rightArm);
    this.hips.position.y = 0;

    // A slightly tapered, oversized garment gives the mannequin an actual
    // clothing silhouette rather than a stack of exposed anatomy capsules.
    const points = [
      [0.255, 0.94], [0.29, 0.96], [0.303, 1.005], [0.326, 1.16],
      [0.34, 1.35], [0.328, 1.47], [0.285, 1.53], [0.17, 1.58], [0.125, 1.60],
    ].map(([r, y]) => new THREE.Vector2(r, y));
    this.shirt = this.mesh(new THREE.LatheGeometry(points, 40), this.fabric);
    this.shirt.scale.z = 0.67;
    this.shirtRest = new Float32Array(this.shirt.geometry.attributes.position.array);
    this.torso.add(this.shirt);
    const waist = this.mesh(new THREE.CylinderGeometry(0.274, 0.269, 0.055, 40), this.rib);
    waist.scale.z = 0.69;
    waist.position.y = 0.968;
    this.torso.add(waist);

    // Hood behind the neck, with its rim visible around the jaw.
    const hood = this.ellipsoid(this.fabric, 0.245, 0.27, 0.17, 0, 1.69, -0.075);
    this.torso.add(hood);
    const collar = this.mesh(new THREE.TorusGeometry(0.132, 0.04, 12, 36), this.rib);
    collar.rotation.x = Math.PI / 2;
    collar.scale.y = 0.75;
    collar.position.set(0, 1.59, 0.035);
    this.torso.add(collar);
    const pocket = this.mesh(new RoundedBoxGeometry(0.36, 0.17, 0.035, 3, 0.035), this.fabric);
    pocket.position.set(0, 1.12, 0.207);
    this.torso.add(pocket);
    const pocketSeam = this.curve([
      new THREE.Vector3(-0.16, 1.19, 0.229), new THREE.Vector3(-0.12, 1.13, 0.241),
      new THREE.Vector3(0.12, 1.13, 0.241), new THREE.Vector3(0.16, 1.19, 0.229),
    ], 0.0025, this.rib);
    this.torso.add(pocketSeam);
    for (const x of [-0.047, 0.047]) {
      const string = this.curve([
        new THREE.Vector3(x, 1.565, 0.15), new THREE.Vector3(x * 1.15, 1.49, 0.215), new THREE.Vector3(x * 1.4, 1.40, 0.232),
      ], 0.004, this.rib);
      this.torso.add(string);
      const tip = this.mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.026, 8), this.shoeSole);
      tip.position.set(x * 1.4, 1.394, 0.233);
      this.torso.add(tip);
    }
    this.labelTexture = this.createLabel(OUTFITS.ink.print);
    this.chestLabel = new THREE.Mesh(new THREE.PlaneGeometry(0.20, 0.052), new THREE.MeshBasicMaterial({ map: this.labelTexture, transparent: true, depthWrite: false, toneMapped: false }));
    this.chestLabel.position.set(0.005, 1.375, 0.227);
    this.torso.add(this.chestLabel);

    this.createLegs();
    this.createRestingArm();
    this.rightUpper = this.mesh(new THREE.CapsuleGeometry(0.107, 1, 6, 16), this.fabric);
    this.rightLower = this.mesh(new THREE.CapsuleGeometry(0.09, 1, 6, 16), this.fabric);
    this.rightElbow = this.ellipsoid(this.fabric, 0.111, 0.117, 0.107, 0, 0, 0);
    this.rightCuff = this.mesh(new THREE.CylinderGeometry(0.073, 0.08, 0.085, 20), this.rib);
    this.rightArm.add(this.rightUpper, this.rightLower, this.rightElbow, this.rightCuff, this.phoneHand);
    this.createPhoneHand();

    // Skin and facial components are individually deformable, so the bundled
    // preview supports expressions without a proprietary character/morph set.
    this.head.position.set(0, 1.825, 0.012);
    this.head.add(this.ellipsoid(this.skin, 0.18, 0.226, 0.164, 0, 0.012, 0.012));
    const neck = this.ellipsoid(this.skin, 0.077, 0.105, 0.073, 0, -0.22, -0.002);
    this.head.add(neck);
    for (const side of [-1, 1]) {
      this.head.add(this.ellipsoid(this.skin, 0.035, 0.058, 0.029, side * 0.178, -0.012, -0.001));
      const earring = this.mesh(new THREE.TorusGeometry(0.013, 0.003, 6, 16), new THREE.MeshStandardMaterial({ color: 0xb4ae95, metalness: 0.85, roughness: 0.27 }));
      earring.position.set(side * 0.19, -0.047, 0.019);
      this.head.add(earring);
    }
    const hairCap = this.mesh(new THREE.SphereGeometry(1, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2), this.hair);
    hairCap.scale.set(0.188, 0.153, 0.171);
    hairCap.position.set(0, 0.085, 0.004);
    this.head.add(hairCap);
    for (let i = 0; i < 15; i += 1) {
      const angle = (i / 14) * Math.PI;
      const curl = this.ellipsoid(this.hair, 0.035, 0.032 + (i % 3) * 0.003, 0.034,
        Math.cos(angle) * 0.149, 0.102 + Math.sin(angle) * 0.018, Math.sin(angle) * 0.103 + 0.058);
      this.head.add(curl);
    }
    const white = new THREE.MeshStandardMaterial({ color: 0xe2d9c8, roughness: 0.48 });
    this.leftPupil = this.createEye(this.leftEye, -0.067, white);
    this.rightPupil = this.createEye(this.rightEye, 0.067, white);
    this.head.add(this.leftEye, this.rightEye);
    this.createBrow(this.leftBrow, -0.067);
    this.createBrow(this.rightBrow, 0.067);
    this.head.add(this.leftBrow, this.rightBrow);
    this.head.add(this.ellipsoid(this.skin, 0.025, 0.045, 0.035, 0, -0.018, 0.166));
    this.head.add(this.ellipsoid(this.skin, 0.034, 0.018, 0.025, 0, -0.043, 0.182));
    for (const x of [-0.018, 0.018]) {
      const nostril = this.ellipsoid(this.lipMaterial, 0.006, 0.003, 0.003, x, -0.049, 0.199);
      this.head.add(nostril);
    }
    this.mouthInterior = this.ellipsoid(this.mouthMaterial, 0.045, 0.006, 0.009, 0, -0.098, 0.171);
    this.lips = this.mesh(new THREE.TorusGeometry(1, 0.09, 6, 40), this.lipMaterial);
    this.lips.scale.set(0.045, 0.007, 0.012);
    this.lips.position.copy(this.mouthInterior.position).add(new THREE.Vector3(0, 0, 0.006));
    this.teeth = this.mesh(new RoundedBoxGeometry(0.06, 0.013, 0.008, 2, 0.003), white);
    this.teeth.position.set(0, -0.09, 0.181);
    this.tongue = this.ellipsoid(new THREE.MeshStandardMaterial({ color: 0xa75f61, roughness: 0.65 }), 0.024, 0.007, 0.006, 0, -0.121, 0.181);
    this.head.add(this.mouthInterior, this.lips, this.teeth, this.tongue);

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.075;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 7;
    this.controls.minPolarAngle = Math.PI * 0.22;
    this.controls.maxPolarAngle = Math.PI * 0.6;
    this.controls.rotateSpeed = 0.55;
    this.controls.zoomSpeed = 0.7;
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
    this.update(0, 1 / 60, null);
  }

  setOutfit(color: OutfitColor): void {
    const palette = OUTFITS[color] ?? OUTFITS.ink;
    this.fabric.color.setHex(palette.fabric);
    this.rib.color.setHex(palette.rib);
    this.pants.color.setHex(palette.pants);
    const old = this.labelTexture;
    this.labelTexture = this.createLabel(palette.print);
    this.chestLabel.material.map = this.labelTexture;
    this.chestLabel.material.needsUpdate = true;
    old.dispose();
  }

  setMotionEnabled(enabled: boolean): void { this.motionEnabled = enabled; }

  /** Calibrated relative orientation in radians; no position is inferred. */
  setGyro(rotation: GyroRotation | null): void {
    this.gyroTarget.set(
      clamp(rotation?.x || 0, -0.6, 0.6),
      clamp(rotation?.y || 0, -0.6, 0.6),
      clamp(rotation?.z || 0, -0.6, 0.6),
    );
  }

  setView(view: "full" | "portrait"): void {
    this.view = view;
    this.fitCamera();
  }

  resetView(): void { this.fitCamera(); }

  update(elapsedSeconds: number, deltaSeconds: number, face: FaceFrame | null): void {
    if (this.disposed) return;
    const dt = clamp(deltaSeconds || 1 / 60, 0.001, 0.08);
    const t = Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0;
    const movement = this.motionEnabled ? 1 : 0;
    const breathe = Math.sin(t * 1.7);
    this.character.position.y = breathe * 0.004 * movement;
    this.character.rotation.y = (0.045 + Math.sin(t * 0.42) * 0.026) * movement;
    this.torso.rotation.z = Math.sin(t * 0.55) * 0.009 * movement;
    this.torso.scale.set(1 + breathe * 0.007 * movement, 1, 1 + breathe * 0.008 * movement);
    this.hips.rotation.z = Math.sin(t * 0.55) * -0.006 * movement;
    this.leftArm.rotation.z = Math.sin(t * 0.8) * 0.008 * movement;

    this.gyro.lerp(this.gyroTarget, 1 - Math.exp(-dt / 0.14));
    this.elbow.set(-0.535 - this.gyro.y * 0.045, 1.275 + this.gyro.x * 0.035, 0.245);
    this.wrist.set(-0.535 - this.gyro.y * 0.095, 1.715 - this.gyro.x * 0.075 + breathe * 0.003 * movement, 0.55 + this.gyro.x * 0.055);
    this.segment(this.rightUpper, this.shoulder, this.elbow, 0.107);
    this.segment(this.rightLower, this.elbow, this.wrist, 0.09);
    this.rightElbow.position.copy(this.elbow);
    this.temp.copy(this.wrist).sub(this.elbow).normalize();
    this.rightCuff.position.copy(this.wrist).addScaledVector(this.temp, -0.018);
    this.rightCuff.quaternion.setFromUnitVectors(Y_AXIS, this.temp);
    this.phoneHand.position.copy(this.wrist);
    this.phoneHand.rotation.set(-0.09 + this.gyro.x * 0.5, -0.12 + this.gyro.y * 0.65, -0.08 - this.gyro.z * 0.55);

    // Subtle live secondary motion at the hoodie hem. This is deliberately
    // procedural deformation, not a precomputed or physically simulated bake.
    const position = this.shirt.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < position.count; i += 1) {
      const x = this.shirtRest[i * 3];
      const y = this.shirtRest[i * 3 + 1];
      const z = this.shirtRest[i * 3 + 2];
      const weight = clamp((1.16 - y) / 0.22, 0, 1);
      const angle = Math.atan2(z, x);
      const wave = Math.sin(angle * 3 + t * 1.35) * 0.003 * weight * movement;
      position.setXYZ(i, x + Math.cos(angle) * wave, y + Math.sin(angle * 2 - t * 1.1) * 0.0018 * weight * movement, z + Math.sin(angle) * wave);
    }
    position.needsUpdate = true;

    if (face) {
      this.lastFace = face;
      this.lastFaceAt = t;
    }
    const activeFace = this.lastFace && t - this.lastFaceAt < 0.7 ? this.lastFace : null;
    const expressions = activeFace?.blendshapes ?? {};
    const expressionAlpha = 1 - Math.exp(-dt / 0.045);
    const names = ["jawOpen", "mouthSmileLeft", "mouthSmileRight", "mouthPucker", "mouthFunnel", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight", "eyeLookInLeft", "eyeLookOutLeft", "eyeLookInRight", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeLookDownLeft", "eyeLookDownRight"];
    for (const name of names) {
      const target = clamp(expressions[name] || 0, 0, 1);
      this.faceValues[name] = THREE.MathUtils.lerp(this.faceValues[name] || 0, target, expressionAlpha);
    }
    const expression = (name: string) => this.faceValues[name] || 0;
    const autoBlink = !activeFace?.detected && movement ? Math.pow(Math.max(0, Math.sin(t * 1.23)), 48) : 0;
    const smile = (expression("mouthSmileLeft") + expression("mouthSmileRight")) * 0.5;
    const blinkL = Math.max(autoBlink, expression("eyeBlinkLeft"));
    const blinkR = Math.max(autoBlink, expression("eyeBlinkRight"));
    this.leftEye.scale.y = Math.max(0.07, 1 - blinkL * 0.95 - expression("eyeSquintLeft") * 0.3);
    this.rightEye.scale.y = Math.max(0.07, 1 - blinkR * 0.95 - expression("eyeSquintRight") * 0.3);
    this.leftPupil.position.x = (expression("eyeLookInLeft") - expression("eyeLookOutLeft")) * 0.008;
    this.rightPupil.position.x = (expression("eyeLookOutRight") - expression("eyeLookInRight")) * 0.008;
    this.leftPupil.position.y = (expression("eyeLookUpLeft") - expression("eyeLookDownLeft")) * 0.005;
    this.rightPupil.position.y = (expression("eyeLookUpRight") - expression("eyeLookDownRight")) * 0.005;
    this.leftBrow.position.y = 0.073 + expression("browInnerUp") * 0.018 + expression("browOuterUpLeft") * 0.012 - expression("browDownLeft") * 0.012;
    this.rightBrow.position.y = 0.073 + expression("browInnerUp") * 0.018 + expression("browOuterUpRight") * 0.012 - expression("browDownRight") * 0.012;
    this.leftBrow.rotation.z = -expression("browDownLeft") * 0.16;
    this.rightBrow.rotation.z = expression("browDownRight") * 0.16;
    const open = expression("jawOpen");
    const pucker = Math.max(expression("mouthPucker"), expression("mouthFunnel"));
    const width = 0.043 + smile * 0.018 - pucker * 0.018;
    const height = 0.0045 + open * 0.042 + pucker * 0.012;
    this.mouthInterior.scale.set(width, height, 0.009);
    this.mouthInterior.position.y = -0.098 - open * 0.012;
    this.lips.scale.set(width, height, 0.012);
    this.lips.position.y = this.mouthInterior.position.y;
    this.teeth.visible = open > 0.16;
    this.teeth.scale.x = Math.max(0.45, 1 - pucker * 0.65);
    this.teeth.position.y = this.mouthInterior.position.y + height * 0.56;
    this.tongue.visible = open > 0.32;
    this.tongue.position.y = this.mouthInterior.position.y - height * 0.56;

    const rotation = activeFace?.rotation;
    this.headTarget.set(
      clamp(rotation?.x || 0, -0.65, 0.65) * 0.7,
      -clamp(rotation?.y || 0, -0.85, 0.85) * 0.8,
      -clamp(rotation?.z || 0, -0.55, 0.55) * 0.75 + Math.sin(t * 0.7) * 0.009 * movement,
    );
    const headAlpha = 1 - Math.exp(-dt / 0.075);
    this.head.rotation.x = THREE.MathUtils.lerp(this.head.rotation.x, this.headTarget.x, headAlpha);
    this.head.rotation.y = THREE.MathUtils.lerp(this.head.rotation.y, this.headTarget.y, headAlpha);
    this.head.rotation.z = THREE.MathUtils.lerp(this.head.rotation.z, this.headTarget.z, headAlpha);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.resizeObserver.disconnect();
    this.controls.dispose();
    const geometries = new Set<THREE.BufferGeometry>();
    const materials = new Set<THREE.Material>();
    const textures = new Set<THREE.Texture>();
    this.scene.traverse((object) => {
      if (!(object instanceof THREE.Mesh)) return;
      geometries.add(object.geometry);
      for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
        materials.add(material);
        const map = (material as THREE.MeshStandardMaterial).map;
        if (map) textures.add(map);
      }
    });
    geometries.forEach((geometry) => geometry.dispose());
    materials.forEach((material) => material.dispose());
    textures.forEach((texture) => texture.dispose());
    this.renderer.dispose();
    this.renderer.forceContextLoss();
    this.canvas.remove();
  }

  private resize(): void {
    if (this.disposed) return;
    const { width, height } = this.container.getBoundingClientRect();
    const w = Math.max(1, width);
    const h = Math.max(1, height);
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.fitCamera();
  }

  private fitCamera(): void {
    const portrait = this.view === "portrait";
    const targetY = portrait ? 1.63 : 1.04;
    const frameHeight = portrait ? 1.12 : 2.31;
    const frameWidth = portrait ? 1.22 : 1.45;
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov * 0.5));
    const distance = Math.max(frameHeight / (2 * tangent), frameWidth / (2 * tangent * this.camera.aspect)) * 1.05;
    this.camera.position.set(0.12, targetY + 0.12, distance);
    this.camera.lookAt(0, targetY, 0);
    if (this.controls) {
      this.controls.target.set(0, targetY, 0);
      this.controls.update();
      this.controls.saveState();
    }
  }

  private createLegs(): void {
    this.hips.add(this.ellipsoid(this.pants, 0.27, 0.16, 0.18, 0, 0.94, 0));
    for (const side of [-1, 1]) {
      const leg = new THREE.Group();
      leg.position.x = side * 0.143;
      const upper = this.mesh(new THREE.CylinderGeometry(0.145, 0.12, 0.43, 24), this.pants);
      upper.scale.z = 0.88;
      upper.position.y = 0.74;
      const lower = this.mesh(new THREE.CylinderGeometry(0.123, 0.097, 0.43, 24), this.pants);
      lower.scale.z = 0.9;
      lower.position.set(side * 0.012, 0.345, 0.012);
      const knee = this.ellipsoid(this.pants, 0.122, 0.1, 0.11, side * 0.006, 0.53, 0.005);
      const trouserCuff = this.mesh(new THREE.CylinderGeometry(0.099, 0.101, 0.045, 24), this.pants);
      trouserCuff.position.set(side * 0.012, 0.15, 0.012);
      leg.add(upper, lower, knee, trouserCuff);
      const sole = this.mesh(new RoundedBoxGeometry(0.218, 0.065, 0.365, 3, 0.028), this.shoeSole);
      sole.position.set(side * 0.012, 0.038, 0.067);
      const sneaker = this.mesh(new RoundedBoxGeometry(0.205, 0.113, 0.335, 5, 0.055), this.shoe);
      sneaker.position.set(side * 0.012, 0.098, 0.057);
      const heel = this.mesh(new RoundedBoxGeometry(0.13, 0.07, 0.018, 2, 0.008), this.dark);
      heel.position.set(side * 0.012, 0.105, -0.113);
      leg.add(sole, sneaker, heel);
      for (let i = 0; i < 3; i += 1) {
        const lace = this.mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.087, 8), this.shoe);
        lace.rotation.z = Math.PI / 2;
        lace.position.set(side * 0.012, 0.158, 0.03 + i * 0.035);
        leg.add(lace);
      }
      this.hips.add(leg);
    }
  }

  private createRestingArm(): void {
    const shoulder = new THREE.Vector3(0.335, 1.49, 0.01);
    const elbow = new THREE.Vector3(0.385, 1.175, 0.045);
    const wrist = new THREE.Vector3(0.35, 0.91, 0.12);
    const upper = this.mesh(new THREE.CapsuleGeometry(0.11, 1, 6, 16), this.fabric);
    const lower = this.mesh(new THREE.CapsuleGeometry(0.089, 1, 6, 16), this.fabric);
    this.segment(upper, shoulder, elbow, 0.11);
    this.segment(lower, elbow, wrist, 0.089);
    const elbowMesh = this.ellipsoid(this.fabric, 0.112, 0.106, 0.1, elbow.x, elbow.y, elbow.z);
    const cuff = this.mesh(new THREE.CylinderGeometry(0.076, 0.081, 0.075, 20), this.rib);
    cuff.position.copy(wrist);
    cuff.quaternion.setFromUnitVectors(Y_AXIS, new THREE.Vector3().subVectors(wrist, elbow).normalize());
    const hand = new THREE.Group();
    hand.position.copy(wrist).add(new THREE.Vector3(-0.005, -0.071, 0.01));
    hand.rotation.z = -0.06;
    hand.add(this.ellipsoid(this.skin, 0.054, 0.086, 0.035, 0, 0, 0));
    for (let i = 0; i < 4; i += 1) {
      hand.add(this.ellipsoid(this.skin, 0.012, 0.047 - Math.abs(i - 1.5) * 0.007, 0.014, -0.031 + i * 0.02, -0.058, 0.005));
    }
    hand.add(this.ellipsoid(this.skin, 0.023, 0.04, 0.022, -0.05, -0.006, 0.015));
    this.leftArm.add(upper, lower, elbowMesh, cuff, hand);
  }

  private createPhoneHand(): void {
    const phone = new THREE.Group();
    phone.position.set(0, 0.12, 0.015);
    const body = this.mesh(new RoundedBoxGeometry(0.145, 0.282, 0.023, 5, 0.018), this.dark);
    const glass = this.mesh(new RoundedBoxGeometry(0.131, 0.266, 0.003, 4, 0.015), new THREE.MeshPhysicalMaterial({ color: 0x4d5260, metalness: 0.45, roughness: 0.22 }));
    glass.position.z = 0.014;
    phone.add(body, glass);
    const bump = this.mesh(new RoundedBoxGeometry(0.061, 0.064, 0.012, 3, 0.011), this.dark);
    bump.position.set(-0.032, 0.096, 0.022);
    phone.add(bump);
    for (const [x, y] of [[-0.045, 0.11], [-0.022, 0.085]]) {
      const lens = this.mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.009, 20), new THREE.MeshStandardMaterial({ color: 0x10151e, metalness: 0.8, roughness: 0.18 }));
      lens.rotation.x = Math.PI / 2;
      lens.position.set(x, y, 0.032);
      phone.add(lens);
      const glint = this.ellipsoid(new THREE.MeshBasicMaterial({ color: 0x738091 }), 0.003, 0.003, 0.001, x - 0.003, y + 0.003, 0.037);
      phone.add(glint);
    }
    const flash = this.ellipsoid(new THREE.MeshBasicMaterial({ color: 0xd5c9a3 }), 0.004, 0.004, 0.001, -0.018, 0.113, 0.029);
    phone.add(flash);
    const screen = this.mesh(new RoundedBoxGeometry(0.127, 0.254, 0.002, 3, 0.013), new THREE.MeshBasicMaterial({ color: 0x54636b }));
    screen.position.z = -0.013;
    phone.add(screen);
    this.phoneHand.add(phone);
    // Fingers close around the lower edges, with the palm behind the phone.
    this.phoneHand.add(this.ellipsoid(this.skin, 0.05, 0.074, 0.034, 0.023, 0.027, -0.026));
    for (let i = 0; i < 4; i += 1) {
      const finger = this.ellipsoid(this.skin, 0.027, 0.012, 0.019, -0.056, 0.018 + i * 0.026, 0.014);
      finger.rotation.z = -0.12;
      this.phoneHand.add(finger);
    }
    const thumb = this.ellipsoid(this.skin, 0.017, 0.043, 0.018, 0.06, 0.052, -0.008);
    thumb.rotation.z = -0.25;
    this.phoneHand.add(thumb);
  }

  private createEye(group: THREE.Group, x: number, white: THREE.Material): THREE.Mesh {
    group.position.set(x, 0.024, 0.162);
    const eye = this.ellipsoid(white, 0.041, 0.018, 0.011, 0, 0, 0);
    const iris = this.ellipsoid(this.dark, 0.013, 0.014, 0.004, 0, 0, 0.011);
    const glint = this.ellipsoid(new THREE.MeshBasicMaterial({ color: 0xf1eee5 }), 0.0033, 0.0033, 0.001, -0.003, 0.004, 0.015);
    iris.add(glint);
    group.add(eye, iris);
    return iris;
  }

  private createBrow(group: THREE.Group, x: number): void {
    group.position.set(x, 0.073, 0.155);
    group.add(this.curve([
      new THREE.Vector3(-0.037, -0.004, 0), new THREE.Vector3(0, 0.004, 0.008), new THREE.Vector3(0.035, -0.002, 0),
    ], 0.006, this.hair));
  }

  private segment(mesh: THREE.Mesh, from: THREE.Vector3, to: THREE.Vector3, radius: number): void {
    this.temp.subVectors(to, from);
    const length = this.temp.length();
    mesh.position.copy(from).addScaledVector(this.temp, 0.5);
    mesh.quaternion.setFromUnitVectors(Y_AXIS, this.temp.normalize());
    // CapsuleGeometry has a unit cylinder plus the two hemisphere radii.
    mesh.scale.y = length / (1 + radius * 2);
  }

  private mesh<G extends THREE.BufferGeometry, M extends THREE.Material>(geometry: G, material: M): THREE.Mesh<G, M> {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  private ellipsoid(material: THREE.Material, x: number, y: number, z: number, px: number, py: number, pz: number): THREE.Mesh {
    const mesh = this.mesh(new THREE.SphereGeometry(1, 24, 16), material);
    mesh.scale.set(x, y, z);
    mesh.position.set(px, py, pz);
    return mesh;
  }

  private curve(points: THREE.Vector3[], radius: number, material: THREE.Material): THREE.Mesh {
    return this.mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 18, radius, 6, false), material);
  }

  private createLabel(color: string): THREE.CanvasTexture {
    const canvas = document.createElement("canvas");
    canvas.width = 768;
    canvas.height = 192;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = color;
      ctx.font = "600 58px Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("IYLMIBS", 384, 75);
      ctx.font = "24px monospace";
      ctx.fillText("VIRTUAL WARDROBE", 384, 137);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    return texture;
  }
}
