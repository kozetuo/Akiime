# Akiime Selfie Studio

A browser prototype for Iylmibs: a live camera panel beside a full-body character
holding a phone. The face and head follow the camera; the body plays a designed
selfie idle. This is face capture with an animated body, not full-body mocap.

## What is implemented

- A procedural, stylized Three.js character with a phone and three outfit
  colorways: Ink, Chalk, and Oxide.
- MediaPipe FaceLandmarker expressions and head rotation, with smoothing,
  recentering, and a neutral fallback when no face is detected.
- A procedural breathing/weight-shift idle and subtle hoodie-hem sway. These are
  authored motion effects, **not baked cloth or a cloth physics simulation**.
- Optional relative phone-orientation control with calibration and bounded arm
  movement. It measures rotation, not the position of the user's hand.
- Camera switching, full-body/close-up views, and an interactive avatar view.
- Local, silent split-screen video recording, capped at 60 seconds. The browser
  chooses an available MP4 or WebM encoder.
- An embed-code dialog and a compact `?embed=1` layout.

## Run locally

Use a Node.js version supported by pnpm and **pnpm 11.25.0**, as pinned in
`package.json`.

```sh
pnpm install
pnpm start
pnpm check
pnpm build
```

The development server uses port 3000. `pnpm check` checks TypeScript;
`pnpm build` checks types and creates the static `dist/` output.
Use `pnpm serve` to preview that build.

Camera access requires HTTPS or localhost and browser permission. A phone
accessing a computer's plain HTTP LAN address does not get the localhost
exception; use an HTTPS development URL for phone testing.

The first camera start downloads MediaPipe model/WASM resources from Google
Storage and jsDelivr. Tracking runs in the browser: camera frames and saved
recordings are never sent or uploaded by this app. Downloads remain on the
user's device. Model/WASM URLs can be overridden in `FaceTrackerOptions` for
self-hosting.

## Embed in Iylmibs

Publish `dist/` on a host that permits embedding, then use the actual published
path or URL. This same-origin example assumes it is served at `/akiime/`:

```html
<iframe
  src="/akiime/?embed=1"
  title="Akiime Selfie Studio"
  allow="camera; accelerometer; gyroscope"
  style="width:100%;height:780px;border:0;border-radius:16px"
></iframe>
```

For a separate origin, replace `src` with its published HTTPS URL and configure
the parent site's `Permissions-Policy` to allow that origin for `camera`,
`accelerometer`, and `gyroscope`. A restrictive parent policy cannot be relaxed
by the iframe's `allow` attribute. The studio host must also permit the parent
to frame it through its CSP/frame headers. Browser permission is still required.
An embed snippet does not make a private or protected preview publicly accessible.

## Replace the preview with a production character

There is **no Daz/GLB importer in this prototype yet**. The next asset pipeline is:

1. Prepare the character, clothes, and phone in Daz/Blender; optimize geometry,
   textures, materials, and the skeleton for mobile rendering.
2. Preserve expression morph targets and map MediaPipe names such as `jawOpen`,
   `eyeBlinkLeft`, `mouthSmileLeft`, and `browInnerUp` to the character's shapes.
   Match subject-left/right consistently; the camera preview is mirrored in UI.
3. Author a seamless selfie idle with a stable phone grip. Add a GLB loader,
   animation mixer, facial morph mapping, and arm controls to `avatar-stage.ts`.
4. Simulate garments against that exact idle offline. Convert the result to
   supported playback data, such as animated morph targets or a suitable bone
   approximation. A raw Daz/Blender/Alembic cloth cache is not automatically a
   playable GLB animation; dense caches may need custom playback or offline render.
5. Check the character, hair, and garment licenses for the intended interactive
   web distribution before publishing them.

Keep gyro-driven arm changes small when using baked sleeves: a baked simulation
matches its authored motion, not arbitrary new poses. Larger arm motion needs
garment corrections or a different live simulation approach.

## Current limits and verification

The procedural preview does not establish Daz quality or production cloth
behavior. No motion-data export, imported character, microphone recording, or
upload backend is implemented. Camera quality, face tracking, motion permission,
sensor axes, iframe access, and video encoding need testing on real iOS and
Android phones. Desktop orientation APIs may exist without providing readings;
the idle continues if motion is unavailable or permission is declined.
