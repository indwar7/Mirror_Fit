import './styles.css';

import { applyPreviewMirror, MIRROR } from './camera';
import { enterFeature, goLanding, previewFile, setPreview, switchTab } from './ui';
import { filterAvatars, loadAvatars, selectAvatar, setLiveSourceFile } from './avatars';
import { toggleStreamingVoice } from './voice';
import { releaseInFlightIfRunning, startLiveCamera, toggleLiveSwap } from './liveswap';
import { lpRecalibrate, setLpFollowHead, setLpSourceFile, startLpCamera, toggleLp } from './liveportrait';
import {
  clearFabric, clearGarmentBack, initTryOn, pauseForHiddenTab, renderOutfitScore, resumeIfRunning,
  selectGarmentType, setFabric, setGarment, setGarmentBack, startTryOnCamera, toggleTryOn,
} from './tryon';
import { avCreate, avSetCategory, avSetGender, avSetPhoto, avTryon } from './my-avatar';
import { twinGenerate, twinPreviewAudio, twinPreviewImg, twinToggleRecord } from './ai-twin';
import { populateArGrid, selectArAvatar } from './ar-presets';

// three.js + MediaPipe are only needed by the hidden AR tab: fetched on first use.
const loadAr = () => import('./ar-avatar');
import { installCredits, mfBuy, mfResetCredits } from './credits';
import { faExplainLive, initFitAnalyser } from './fit-analyser';
import { installBuildStamp } from './build-stamp';
import { frExplain, installFitReadout } from './fit-readout';

// index.html keeps the demo's inline onclick/onchange handlers unchanged, so
// every name they use is published here. This is the one place the markup and
// the modules meet; adding a handler to the HTML means adding it here.
const handlers = {
  switchTab, enterFeature, goLanding, setPreview, previewFile,
  filterAvatars, selectAvatar, setLiveSourceFile, toggleStreamingVoice,
  startLiveCamera, toggleLiveSwap,
  startLpCamera, setLpSourceFile, setLpFollowHead, lpRecalibrate, toggleLp,
  startTryOnCamera, setGarment, setFabric, clearFabric, setGarmentBack, clearGarmentBack,
  selectGarmentType, renderOutfitScore, toggleTryOn,
  frExplain,
  startArCamera: () => loadAr().then((ar) => ar.startArCamera()),
  toggleArAvatar: () => loadAr().then((ar) => ar.toggleArAvatar()),
  _selectArAvatar: selectArAvatar,
  avSetPhoto, avSetGender, avSetCategory, avTryon, avCreate,
  twinPreviewImg, twinPreviewAudio, twinToggleRecord, twinGenerate,
  faExplainLive,
  mfBuy, mfResetCredits,
};
Object.assign(window, handlers);

// ── Boot, in the order the single-file demo ran its script blocks ─────────
applyPreviewMirror('liveVideo', MIRROR.ls);
applyPreviewMirror('lpVideo', MIRROR.lp);
applyPreviewMirror('tryonVideo', MIRROR.to);

void loadAvatars();
initTryOn();

// Pause when the tab is hidden. Live swap paces itself from
// requestAnimationFrame, which the browser already stops for a hidden tab;
// coming back it only needs the in-flight slot released.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    pauseForHiddenTab();
  } else {
    releaseInFlightIfRunning();
    resumeIfRunning();
  }
});

installCredits();
initFitAnalyser();
installBuildStamp();
installFitReadout();
populateArGrid();

// Opens on Try-On: the Face Swap tab it used to open on is hidden.
switchTab('tryon');
