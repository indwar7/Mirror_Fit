/* Browser side of the live session.
 *
 * The 3D mirror (mirror.html) already draws the garment onto a canvas.
 * That canvas — not the raw camera — is what we send: the composite is
 * the guide the model restyles. While no GPU is free, the same canvas
 * is what the user keeps watching, so the screen is never blank.
 */

const SIGNAL = {
  offer: "/offer",
  status: "/status",
  hangup: "/hangup",
};

export class LiveSession {
  /** @param {HTMLCanvasElement} canvas  the 3D composite
   *  @param {HTMLVideoElement} output   where the model's frames play */
  constructor(canvas, output, { onState = () => {}, onStats = () => {} } = {}) {
    this.canvas = canvas;
    this.output = output;
    this.onState = onState;
    this.onStats = onStats;
    this.pc = null;
    this.sid = null;
    this.poll = null;
  }

  async start(fps = 12) {
    this.onState({ state: "queued" });

    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // captureStream pulls frames straight off the 3D canvas — no second
    // camera permission, and the garment is already composited in.
    const stream = this.canvas.captureStream(fps);
    for (const track of stream.getVideoTracks()) {
      this.pc.addTrack(track, stream);
    }

    this.pc.addTransceiver("video", { direction: "recvonly" });

    this.pc.ontrack = (ev) => {
      this.output.srcObject = ev.streams[0];
      this.onState({ state: "live" });
    };

    this.pc.onconnectionstatechange = () => {
      const s = this.pc.connectionState;
      if (s === "failed" || s === "disconnected" || s === "closed") {
        this.onState({ state: "ended", reason: s });
        this.stop();
      }
    };

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    await this._iceComplete();

    // This request blocks server-side until a GPU slot frees up.
    const res = await fetch(SIGNAL.offer, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription.sdp,
        type: this.pc.localDescription.type,
      }),
    });
    if (!res.ok) throw new Error(`offer rejected: ${res.status}`);

    const answer = await res.json();
    this.sid = answer.sid;
    await this.pc.setRemoteDescription(answer);

    this._startPolling();
    return answer;
  }

  /** Trickle ICE would be faster, but the server answers in one shot,
   *  so gather everything before offering. */
  _iceComplete() {
    if (this.pc.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve) => {
      const check = () => {
        if (this.pc.iceGatheringState === "complete") {
          this.pc.removeEventListener("icegatheringstatechange", check);
          resolve();
        }
      };
      this.pc.addEventListener("icegatheringstatechange", check);
    });
  }

  _startPolling() {
    clearInterval(this.poll);
    this.poll = setInterval(async () => {
      if (!this.sid) return;
      try {
        const r = await fetch(`${SIGNAL.status}?sid=${this.sid}`);
        const s = await r.json();
        this.onStats(s);
        if (s.seconds_left !== undefined && s.seconds_left <= 0) this.stop();
      } catch {
        /* a dropped poll is not fatal; the next one will report */
      }
    }, 1000);
  }

  async stop() {
    clearInterval(this.poll);
    this.poll = null;
    if (this.sid) {
      // keepalive so the slot is released even if the tab is closing
      navigator.sendBeacon?.(
        SIGNAL.hangup,
        new Blob([JSON.stringify({ sid: this.sid })], { type: "application/json" })
      );
    }
    if (this.pc) {
      this.pc.close();
      this.pc = null;
    }
    this.sid = null;
    this.onState({ state: "ended" });
  }
}
