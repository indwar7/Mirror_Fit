// One mirror policy per feature, read by BOTH the on-screen preview and the
// frame that goes to the server. They used to be written out separately at
// each call site and had drifted apart: Live Portrait drew its preview
// mirrored but sent the frame un-mirrored, so the server's result was
// composited flipped relative to the live video underneath it.
//
// Face features mirror (selfie / magic-mirror convention — you move left, the
// image moves left). Try-On does NOT: mirroring the person flips the garment
// with them, so logos and plackets would come back reversed.
export const MIRROR = { ls: true, lp: true, to: false } as const;

// The raw <video> preview panes must agree with the result canvas beside them.
// Driving the CSS from the same constant is what stops that drifting again.
export function applyPreviewMirror(videoId: string, mirror: boolean): void {
  const el = document.getElementById(videoId);
  if (el) el.style.transform = mirror ? 'scaleX(-1)' : 'none';
}

// Draw a centre-cropped square of `video` into `ctx` at `sz`, honouring the
// feature's mirror policy. Every camera draw goes through here so preview and
// capture cannot diverge.
export function drawCam(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  sz: number,
  mirror: boolean,
): void {
  const vw = video.videoWidth || sz, vh = video.videoHeight || sz;
  const side = Math.min(vw, vh);
  const sx = (vw - side) / 2, sy = (vh - side) / 2;
  if (mirror) {
    ctx.save();
    ctx.scale(-1, 1);
    ctx.drawImage(video, sx, sy, side, side, -sz, 0, sz, sz);
    ctx.restore();
  } else {
    ctx.drawImage(video, sx, sy, side, side, 0, 0, sz, sz);
  }
}
