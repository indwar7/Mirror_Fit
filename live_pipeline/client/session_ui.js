/* Queue and session UI.
 *
 * The rule from the plan: never a blank screen. While the user waits,
 * the local 3D view keeps running and they see their place in line;
 * when a GPU frees up, the model's output fades in over it.
 */

import { LiveSession } from "./webrtc.js";

export function wireLiveButton({ button, canvas, output, statusEl }) {
  let session = null;

  const show = (text) => {
    if (statusEl) statusEl.textContent = text;
  };

  button.addEventListener("click", async () => {
    if (session) {
      await session.stop();
      session = null;
      button.textContent = "Go live";
      return;
    }

    session = new LiveSession(canvas, output, {
      onState: ({ state, reason }) => {
        if (state === "queued") show("Waiting for a GPU…");
        if (state === "live") {
          output.classList.add("visible");
          button.textContent = "Stop";
        }
        if (state === "ended") {
          output.classList.remove("visible");
          button.textContent = "Go live";
          show(reason ? `Session ended (${reason})` : "Session ended");
          session = null;
        }
      },
      onStats: renderStatus,
    });

    try {
      await session.start();
    } catch (err) {
      show(`Could not start: ${err.message}`);
      session = null;
      button.textContent = "Go live";
    }
  });

  /* What the user reads while they wait, and while they are live.
   *
   * Waiting: lead with the wait time, not the queue position - "about a
   * minute" answers the question people actually have. The position is
   * kept as a second clause so a long queue still feels accounted for.
   *
   * Live: show the countdown. Sessions are 30 seconds and the user is
   * posing for them; hiding the clock makes the cut feel arbitrary.
   */
  function renderStatus(s) {
    if (s.position > 0) {
      const wait = s.estimated_wait;
      const when =
        wait <= 0 ? "You're next" :
        wait < 60 ? `About ${Math.ceil(wait)} seconds` :
        `About ${Math.ceil(wait / 60)} min`;
      const ahead = s.position === 1 ? "1 person ahead" : `${s.position} people ahead`;
      show(wait <= 0 ? "You're next - get into frame" : `${when} - ${ahead}`);
      return;
    }

    if (s.position === 0) {
      const left = Math.max(0, Math.ceil(s.seconds_left ?? 0));
      // The model failing is not the user's problem to solve, but a
      // silent downgrade reads as the product being broken - so name it
      // plainly, without asking them to do anything.
      const degraded = s.stats?.backend === "passthrough";
      show(degraded ? `${left}s left - preview quality` : `${left}s left`);
      return;
    }

    show("Ready");
  }

  return () => session?.stop();
}
