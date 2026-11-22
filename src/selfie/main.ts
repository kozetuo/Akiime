import './studio.css';
import { AvatarStage, type OutfitColor } from './avatar-stage';
import { FaceTracker, type FaceFrame } from './face-tracker';
import { MotionController } from './motion-control';
import { CompositeRecorder } from './recorder';

const icons = {
 camera:'<rect x="3" y="6" width="18" height="14" rx="3"/><path d="m8 6 1-3h6l1 3"/><circle cx="12" cy="13" r="4"/>',
 lock:'<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
 code:'<path d="m8 7-5 5 5 5m8-10 5 5-5 5m-3-14-2 18"/>',
 flip:'<path d="M4 9a8 8 0 0 1 14-3l2 2m0-5v5h-5M20 15a8 8 0 0 1-14 3l-2-2m0 5v-5h5"/>',
 center:'<path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="12" r="4"/><path d="M12 10v4m-2-2h4"/>',
 motion:'<rect x="7" y="3" width="10" height="18" rx="2"/><path d="M11 18h2M3 8l-2 4 2 4m18-8 2 4-2 4"/>',
 close:'<path d="m6 6 12 12M6 18 18 6"/>',
 download:'<path d="M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5"/>',
 orbit:'<path d="M19 8c2 1 3 2 3 4 0 3-4 5-10 5S2 15 2 12c0-2 1-3 3-4m13 6 4-2-1 5"/><circle cx="12" cy="8" r="3"/>',
 stop:'<rect x="6" y="6" width="12" height="12" rx="2"/>',
 check:'<path d="m5 12 4 4L19 6"/>',
};
const icon=(name:keyof typeof icons)=>`<svg viewBox="0 0 24 24" aria-hidden="true">${icons[name]}</svg>`;
const app=document.querySelector<HTMLDivElement>('#app')!;
if(new URLSearchParams(location.search).has('embed')) document.body.classList.add('embedded');

app.innerHTML=`
 <header class="topbar">
  <a class="brand" href="./" aria-label="Akiime Selfie Studio"><span class="brand-mark">a</span><span class="wordmark">akiime</span><span class="brand-divider"></span><span class="byline">BY IYLMIBS</span></a>
  <div class="top-meta"><span class="eyebrow">Selfie studio / 001</span><span class="privacy">${icon('lock')} Your camera stays yours</span><button class="embed-button" id="embed-open">${icon('code')} Embed studio</button></div>
 </header>
 <main class="workspace">
  <div class="intro"><div><h1>Your expression. <em>A new form.</em></h1><p>A little you. A little digital. Make your other self come alive.</p></div><span class="session-label" id="session-state">Preview mode</span></div>
  <div class="capture-grid">
   <section class="pane camera-pane" aria-label="Your camera preview">
    <video id="camera-video" class="camera-video" autoplay muted playsinline></video>
    <div class="pane-head"><div class="pane-title"><span>01 /</span> Your camera</div><span class="mode-tag">${icon('lock')} ON DEVICE</span></div>
    <div class="camera-placeholder" id="camera-placeholder">
     <div class="face-frame"><svg class="face-illustration" viewBox="0 0 100 135" aria-hidden="true"><path d="M29 29C29 4 73 4 73 29v32c0 19-10 34-22 34S29 80 29 61Z"/><path d="M28 40c-10-3-12 17 1 20m44-20c10-3 12 17-1 20M34 39h10m14 0h10M40 46v5m23-5v5M51 48l-3 16h7M41 76q10 8 20 0M40 90v18m23-18v18M40 105c-20 3-32 11-36 26m59-26c20 3 32 11 36 26M29 27c10 2 23-9 28-13 3 10 9 14 16 14"/></svg><span class="scan-line"></span></div>
     <h2>Step into the frame.</h2><p>Your face brings your<br>other self to life.</p>
     <button class="primary-button" id="camera-start">${icon('camera')} <span>Start camera</span></button><p class="privacy-note">Nothing is uploaded. Just you, here.</p>
    </div>
    <div class="pane-foot"><span class="camera-message"><span class="tracking-dot" id="tracking-dot"></span><span id="camera-status" role="status">Camera is off</span></span><div class="camera-tools"><button id="camera-flip" class="icon-button hidden" aria-label="Switch front and rear camera" title="Switch camera">${icon('flip')}</button><button id="camera-stop" class="icon-button hidden" aria-label="Turn camera off" title="Camera off">${icon('stop')}</button></div></div>
   </section>
   <section class="pane avatar-pane" aria-label="Your interactive 3D avatar">
    <div id="avatar-stage" class="avatar-canvas" role="img" aria-label="A dressed full-body avatar holding a phone. Drag to rotate; use full body and close-up buttons to change framing."></div>
    <div class="pane-head"><div class="pane-title"><span>02 /</span> Your other self</div><span class="mode-tag">STUDIO AVATAR</span></div>
    <div class="pane-foot"><span class="avatar-foot-note">${icon('orbit')} Drag to explore</span><div class="view-controls" aria-label="Avatar framing"><button id="view-full" aria-pressed="true">Full body</button><button id="view-portrait" aria-pressed="false">Close-up</button></div></div>
   </section>
  </div>
  <div class="studio-controls">
   <div class="wardrobe"><span class="wardrobe-label">The look</span><div class="swatches" role="group" aria-label="Outfit color"><button class="swatch" data-outfit="ink" style="--swatch:#41483a" aria-label="Ink outfit" aria-pressed="true"></button><button class="swatch" data-outfit="chalk" style="--swatch:#dedccf" aria-label="Chalk outfit" aria-pressed="false"></button><button class="swatch" data-outfit="oxide" style="--swatch:#ad6349" aria-label="Oxide outfit" aria-pressed="false"></button></div><span class="look-name" id="look-name">01 / Ink</span></div>
   <div class="main-controls"><button class="secondary-button center-button" id="calibrate" title="Center your face and phone">${icon('center')}<span class="button-label">Recenter</span></button><button class="secondary-button record-button" id="record"><span class="record-dot"></span><span id="record-label">Record</span></button></div>
   <div class="right-controls"><button class="toggle-button" id="idle" aria-pressed="true"><span class="toggle"></span>Idle movement</button><button class="secondary-button" id="motion" aria-pressed="false">${icon('motion')}<span id="motion-label">Phone motion</span></button></div>
  </div>
  <footer class="footer"><span class="footer-left">${icon('lock')} Face tracking runs on your device. No uploads.</span><a class="download-link hidden" id="download">${icon('download')} Download clip</a><span class="footer-right">Dress the moment. Be the muse.</span></footer>
 </main>
 <dialog id="embed-dialog"><div class="dialog-header"><h2>Bring Akiime with you.</h2><button id="embed-close" class="icon-button" aria-label="Close embed dialog">${icon('close')}</button></div><p>Add the studio to Iylmibs with this responsive embed.</p><code id="embed-code"></code><button id="embed-copy" class="primary-button">${icon('code')} Copy embed code</button><p class="embed-note">Your hosting must allow embedding and camera / motion permissions. Private previews require access; publish on your site’s origin for a seamless public experience.</p></dialog>
 <div id="toast" class="toast hidden" role="status" aria-live="polite"></div>`;

const $=<T extends HTMLElement>(id:string)=>document.getElementById(id) as T;
const video=$<HTMLVideoElement>('camera-video');
const startButton=$<HTMLButtonElement>('camera-start');
const placeholder=$('camera-placeholder');
const cameraStatus=$('camera-status');
const dot=$('tracking-dot');
const recordButton=$<HTMLButtonElement>('record');
const motionButton=$<HTMLButtonElement>('motion');
const idleButton=$<HTMLButtonElement>('idle');
const tracker=new FaceTracker();
const motion=new MotionController();
let stage:AvatarStage;
let recorder:CompositeRecorder;
let stream:MediaStream|null=null;
let cameraGeneration=0;
let cameraStarting=false;
let cameraActive=false;
let trackingActive=false;
let rear=false;
let face:FaceFrame|null=null;
let idle=!matchMedia('(prefers-reduced-motion: reduce)').matches;
let raf=0,toastTimeout=0;
let clipUrl:string|null=null;
let recordingStarted=0;
let destroyed=false;

function toast(message:string,error=false){
 const el=$('toast');el.textContent=message;el.classList.toggle('error',error);el.classList.remove('hidden');
 window.clearTimeout(toastTimeout);toastTimeout=window.setTimeout(()=>el.classList.add('hidden'),error?7000:3800);
}
function cameraError(error:unknown):string{
 const name=error instanceof DOMException?error.name:'';
 if(name==='NotAllowedError'||name==='SecurityError')return 'Camera access was blocked. Allow the camera in your browser, then try again.';
 if(name==='NotFoundError')return 'No camera was found. You can still explore and record the avatar.';
 if(name==='NotReadableError')return 'Your camera is busy. Close other apps using it, then try again.';
 return 'Could not start the camera. Check your connection and camera permissions, then retry.';
}
function setCameraUI(){
 const visible=cameraActive;
 video.classList.toggle('visible',visible);video.classList.toggle('rear',rear);
 placeholder.classList.toggle('hidden',visible);
 $('camera-stop').classList.toggle('hidden',!visible&&!cameraStarting);
 $('camera-flip').classList.toggle('hidden',!visible);
 startButton.disabled=cameraStarting;
 startButton.querySelector('span')!.textContent=cameraStarting?'Connecting…':'Start camera';
 $('session-state').textContent=visible?'Selfie session':'Preview mode';
 if(!visible) {cameraStatus.textContent=cameraStarting?'Waiting for camera…':'Camera is off';dot.classList.remove('live');}
}
function stopCamera(){
 cameraGeneration++;cameraStarting=false;cameraActive=false;trackingActive=false;
 stream?.getTracks().forEach(t=>t.stop());stream=null;video.pause();video.srcObject=null;
 face=null;tracker.reset();setCameraUI();
}
async function startCamera(){
 const generation=++cameraGeneration;
 stream?.getTracks().forEach(t=>t.stop());stream=null;cameraActive=false;trackingActive=false;face=null;tracker.reset();
 cameraStarting=true;setCameraUI();
 if(!window.isSecureContext||!navigator.mediaDevices?.getUserMedia){cameraStarting=false;setCameraUI();toast('Camera access needs HTTPS or localhost in a supported browser.',true);return;}
 let acquired:MediaStream|null=null;
 try{
  acquired=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:rear?'environment':'user'},width:{ideal:1280},height:{ideal:720}},audio:false});
  if(generation!==cameraGeneration||destroyed){acquired.getTracks().forEach(t=>t.stop());return;}
  stream=acquired;video.srcObject=stream;await video.play();
  if(generation!==cameraGeneration||destroyed){acquired.getTracks().forEach(t=>t.stop());return;}
  cameraStarting=false;cameraActive=true;setCameraUI();cameraStatus.textContent='Preparing face tracking…';
  stream.getVideoTracks()[0]?.addEventListener('ended',()=>{if(generation===cameraGeneration){stopCamera();toast('Camera connection ended. Start it again when you’re ready.');}},{once:true});
  try{
   await tracker.load();
   if(generation!==cameraGeneration||destroyed)return;
   trackingActive=true;cameraStatus.textContent='Find your face in the frame';
  }catch(error){
   if(generation!==cameraGeneration||destroyed)return;
   cameraStatus.textContent='Camera on · tracking unavailable';
   toast('Face tracking could not load. Check your connection, then restart the camera.',true);
   console.warn('Akiime face tracking could not initialize',error);
  }
 }catch(error){
  acquired?.getTracks().forEach(t=>t.stop());
  if(generation!==cameraGeneration||destroyed)return;
  cameraStarting=false;cameraActive=false;stream=null;video.srcObject=null;setCameraUI();toast(cameraError(error),true);
 }
}

function syncMotion(){motionButton.classList.toggle('active',motion.active);motionButton.setAttribute('aria-pressed',String(motion.active));$('motion-label').textContent=motion.active?'Motion on':'Phone motion';}
function resetRecordingUI(){recordButton.classList.remove('recording');recordButton.setAttribute('aria-label','Start recording');$('record-label').textContent='Record';}
async function stopRecording(){
 recordButton.disabled=true;
 try{
  const clip=await recorder.stop();if(destroyed)return;
  if(clipUrl)URL.revokeObjectURL(clipUrl);
  clipUrl=URL.createObjectURL(clip.blob);const download=$<HTMLAnchorElement>('download');download.href=clipUrl;
  download.download=`akiime-${new Date().toISOString().replace(/[:.]/g,'-')}.${clip.extension}`;download.classList.remove('hidden');
  toast('Your clip is ready. Download it below.');
 }catch(error){if(!destroyed)toast(error instanceof Error?error.message:'Recording could not be saved.',true);}
 finally{recordButton.disabled=false;resetRecordingUI();}
}

try{
 stage=new AvatarStage($('avatar-stage'));
 stage.setMotionEnabled(idle);idleButton.setAttribute('aria-pressed',String(idle));
 recorder=new CompositeRecorder(stage.canvas,video,{mirrorCamera:()=>!rear,onError:(error)=>{resetRecordingUI();toast(error.message||'Recording stopped unexpectedly.',true);}});
 let last=performance.now(),start=last;
 const animate=(now:number)=>{
  if(destroyed)return;
  const dt=Math.min(.06,(now-last)/1000);last=now;
  if(cameraActive&&trackingActive){
   try{
    const next=tracker.update(video,now);if(next){face=next;cameraStatus.textContent=next.detected?'Face connected':'Find your face in the frame';dot.classList.toggle('live',next.detected);}
   }catch(error){trackingActive=false;face=null;dot.classList.remove('live');cameraStatus.textContent='Tracking paused';toast('Face tracking paused. Restart the camera to reconnect.',true);console.warn('Akiime face inference paused',error);}
  }
  stage.setGyro(motion.rotation);stage.update((now-start)/1000,dt,face);
  if(recorder.recording){const seconds=Math.floor((now-recordingStarted)/1000);$('record-label').textContent=`Stop ${String(Math.floor(seconds/60)).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;if(seconds>=60)void stopRecording();}
  raf=requestAnimationFrame(animate);
 };
 raf=requestAnimationFrame(animate);
}catch(error){
 console.error('Akiime could not create the 3D studio',error);
 $('avatar-stage').innerHTML='<div class="fatal"><h1>Your studio needs WebGL.</h1><p>Enable hardware acceleration in your browser or try a recent Safari, Chrome, Edge, or Firefox browser.</p></div>';
 recordButton.disabled=true;startButton.disabled=true;motionButton.disabled=true;
}

startButton.addEventListener('click',()=>void startCamera());
$('camera-stop').addEventListener('click',stopCamera);
$('camera-flip').addEventListener('click',()=>{rear=!rear;void startCamera();});
$('calibrate').addEventListener('click',()=>{tracker.calibrate();motion.calibrate();stage?.resetView();toast('Centered. Hold your phone comfortably and look ahead.');});
idleButton.addEventListener('click',()=>{idle=!idle;stage?.setMotionEnabled(idle);idleButton.setAttribute('aria-pressed',String(idle));});
motionButton.addEventListener('click',async()=>{
 if(motion.active){motion.stop();syncMotion();toast('Phone motion off.');return;}
 motionButton.disabled=true;
 try{const result=await motion.start();if(result==='active')toast('Phone motion on. Tilt gently; tap Recenter to reset.');else toast(result==='denied'?'Motion permission was declined. The selfie idle will keep playing.':'Phone motion is unavailable here. Try this page on your phone.',result==='denied');}
 catch{toast('Could not read phone motion. The selfie idle will keep playing.',true);}
 finally{motionButton.disabled=false;syncMotion();}
});
recordButton.addEventListener('click',()=>{
 if(!recorder)return;
 if(recorder.recording){void stopRecording();return;}
 try{recorder.start();recordingStarted=performance.now();recordButton.classList.add('recording');recordButton.setAttribute('aria-label','Stop recording');$('record-label').textContent='Stop 00:00';toast('Recording both panels. Tap Stop when you’re done. Maximum 60 seconds.');}
 catch(error){toast(error instanceof Error?error.message:'Recording is unavailable in this browser.',true);}
});
const lookNames:Record<OutfitColor,string>={ink:'01 / Ink',chalk:'02 / Chalk',oxide:'03 / Oxide'};
document.querySelectorAll<HTMLButtonElement>('[data-outfit]').forEach(button=>button.addEventListener('click',()=>{
 const outfit=button.dataset.outfit as OutfitColor;stage?.setOutfit(outfit);$('look-name').textContent=lookNames[outfit];document.querySelectorAll('[data-outfit]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
}));
(['full','portrait'] as const).forEach(view=>$(`view-${view}`).addEventListener('click',()=>{stage?.setView(view);$('view-full').setAttribute('aria-pressed',String(view==='full'));$('view-portrait').setAttribute('aria-pressed',String(view==='portrait'));}));

const dialog=$<HTMLDialogElement>('embed-dialog');
const embedUrl=new URL(location.href);embedUrl.search='?embed=1';embedUrl.hash='';
const embedCode=`<iframe\n  src="${embedUrl.href}"\n  title="Akiime Selfie Studio"\n  allow="camera; accelerometer; gyroscope"\n  style="width:100%;height:780px;border:0;border-radius:16px"\n></iframe>`;
$('embed-code').textContent=embedCode;
$('embed-open').addEventListener('click',()=>dialog.showModal());
$('embed-close').addEventListener('click',()=>dialog.close());
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
$('embed-copy').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(embedCode);toast('Embed code copied.');}catch{const selection=window.getSelection();selection?.removeAllRanges();const range=document.createRange();range.selectNodeContents($('embed-code'));selection?.addRange(range);toast('Select and copy the embed code above.');}});
document.addEventListener('visibilitychange',()=>{if(document.hidden){stopCamera();motion.stop();syncMotion();if(recorder?.recording)void stopRecording();}});
window.addEventListener('pagehide',()=>{destroyed=true;cancelAnimationFrame(raf);clearTimeout(toastTimeout);stopCamera();tracker.dispose();motion.stop();recorder?.dispose();stage?.dispose();if(clipUrl)URL.revokeObjectURL(clipUrl);});
window.addEventListener('pageshow',event=>{if(event.persisted)location.reload();});
