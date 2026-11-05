/* ── Build stamp + scroll audit ─────────────────────────────────────
   The stamp exists because "is the fix actually deployed?" cost several
   rounds: the box had more than one clone of the repo and the server was
   reading a copy that pulls never touched. Now the answer is one line in
   the console, and it comes from git at build time.

   The audit walks the shell after load and reports any element whose
   content is taller than its box while nothing in its ancestor chain can
   scroll — i.e. content a user cannot reach. It only warns.
   ────────────────────────────────────────────────────────────────── */
export function installBuildStamp(): void {
  const BUILD = __BUILD__;
  document.documentElement.setAttribute('data-build', BUILD);
  console.log('%c Mirror Fit ' + BUILD, 'background:#14110F;color:#FBF9F6;padding:2px 6px');

  function scrollable(el: Element): boolean {
    const s = getComputedStyle(el);
    return /(auto|scroll)/.test(s.overflowY) || el === document.scrollingElement;
  }

  function audit(): void {
    const trapped: string[] = [];
    document.querySelectorAll<HTMLElement>('.sidebar, .panel.active, .main, .main > div, .landing')
      .forEach((el) => {
        if (!el.offsetParent && el !== document.body) return;      // hidden
        if (el.scrollHeight - el.clientHeight < 8) return;          // fits
        let n: HTMLElement | null = el, ok = false;
        while (n && n !== document.body) {
          if (scrollable(n) && n.scrollHeight > n.clientHeight + 4) { ok = true; break; }
          n = n.parentElement;
        }
        if (!ok) trapped.push((el.id || el.className || el.tagName) +
          ' (' + el.scrollHeight + 'px in ' + el.clientHeight + 'px)');
      });
    if (trapped.length) {
      console.warn('[scroll-audit] content unreachable in:', trapped);
    } else {
      console.log('[scroll-audit] every pane reachable');
    }
  }

  window.mfScrollAudit = audit;
  window.addEventListener('load', () => { setTimeout(audit, 400); });
  window.addEventListener('resize', () => {
    clearTimeout(window.__sa);
    window.__sa = window.setTimeout(audit, 250);
  });
}
