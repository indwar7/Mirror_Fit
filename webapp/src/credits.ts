/* ── Demo credit meter ──────────────────────────────────────────────
   Additive by design: it hooks enterFeature rather than editing it. The
   balance is held in localStorage for this browser only — no payment is
   taken.
   ────────────────────────────────────────────────────────────────── */
import { maybe } from './dom';
import { onEnterFeature } from './ui';

// Bumping the key is how a new starting balance actually reaches people:
// read() only falls back to START when the key is ABSENT. v3 = 1000 credits,
// enough to demo everything without topping up.
const KEY = 'mf_credits_v3', COST = 10, START = 1000;

function read(): number {
  let raw: string | null = null;
  try { raw = localStorage.getItem(KEY); } catch { /* storage unavailable */ }
  const v = parseInt(raw ?? '', 10);
  return isNaN(v) ? START : Math.max(0, v);
}
function write(v: number): void {
  try { localStorage.setItem(KEY, String(v)); } catch { /* ignore */ }
  paint();
}
function paint(): void {
  const v = read();
  document.querySelectorAll('[data-credit-balance]').forEach((n) => { n.textContent = String(v); });
}

let toastTimer: number | undefined;
function toast(msg: string): void {
  const el = maybe('mfToast');
  if (!el) return;
  el.textContent = msg;
  el.classList.add('is-up');
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { el.classList.remove('is-up'); }, 3000);
}

export function mfBuy(name: string, credits: number): void {
  write(read() + credits);
  toast(name + ' pack — demo only, no payment taken. Added ' + credits + ' credits.');
}

export function mfResetCredits(): void {
  write(START);
  toast('Balance reset to ' + START + ' credits.');
}

export function installCredits(): void {
  /* Charge on launch. */
  onEnterFeature((name) => {
    // Fit Analyser scores in the browser and consumes no GPU time, so it is
    // free. Credits pay for generation, not risk scoring.
    // Live Face Swap is disabled: the :7860 backend is not running.
    if (name === 'liveswap') return false;
    if (name === 'fit') return true;
    const v = read();
    if (v < COST) {
      toast('Not enough credits — each launch costs ' + COST + '.');
      const sec = maybe('lp-credits');
      if (sec) sec.scrollIntoView({ behavior: 'smooth' });
      return false;
    }
    write(v - COST);
    return true;
  });

  /* Reveal on scroll. The landing is the scroll container, not the page. */
  const els = document.querySelectorAll('[data-reveal]');
  if (!('IntersectionObserver' in window)) {
    els.forEach((el) => el.classList.add('is-in'));
  } else {
    const io = new IntersectionObserver((entries) => {
      entries.forEach((en) => {
        if (en.isIntersecting) {
          en.target.classList.add('is-in');
          io.unobserve(en.target);
        }
      });
    }, {
      root: maybe('landing'),
      rootMargin: '0px 0px -8% 0px',
      threshold: 0,
    });
    els.forEach((el) => io.observe(el));
  }

  paint();
}
