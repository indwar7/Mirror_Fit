/* Sign-in, credits, and the session history panel.
 *
 * A live session costs a credit, so the Go live button needs an account
 * behind it. Everything else on the page — the 3D mirror, fabrics, fit —
 * keeps working signed out; only the model costs money.
 */

const KEY = "live_token";

export const auth = {
  get token() { return localStorage.getItem(KEY) || ""; },
  set token(t) { t ? localStorage.setItem(KEY, t) : localStorage.removeItem(KEY); },

  async call(path, opts = {}) {
    const r = await fetch(path, {
      ...opts,
      headers: {
        "Content-Type": "application/json",
        ...(this.token ? { Authorization: `Bearer ${this.token}` } : {}),
        ...(opts.headers || {}),
      },
    });
    if (r.status === 401) { this.token = ""; throw new Error("please sign in"); }
    if (!r.ok) throw new Error((await r.text()) || `error ${r.status}`);
    return r.json();
  },

  async me() { return this.call("/api/me"); },

  async signin(email, password) {
    const d = await this.call("/api/signin", {
      method: "POST", body: JSON.stringify({ email, password }),
    });
    this.token = d.token;
    return d;
  },

  async signup(email, password) {
    const d = await this.call("/api/signup", {
      method: "POST", body: JSON.stringify({ email, password }),
    });
    this.token = d.token;
    return d;
  },

  signout() { this.token = ""; },
};

/** Wires the account strip: sign in / up, credit count, sign out. */
export function wireAccount({ box, onChange = () => {} }) {
  const render = (user) => {
    if (user) {
      box.innerHTML = `
        <span class="who">${user.email}</span>
        <span class="credits">${user.credits} credit${user.credits === 1 ? "" : "s"}</span>
        <button class="quiet" id="signout">Sign out</button>`;
      box.querySelector("#signout").onclick = () => { auth.signout(); render(null); onChange(null); };
    } else {
      box.innerHTML = `
        <form id="af">
          <input id="ae" type="email" placeholder="email" required>
          <input id="ap" type="password" placeholder="password" required>
          <button type="submit">Sign in</button>
          <button type="button" class="quiet" id="asignup">Create account</button>
        </form>
        <span id="amsg"></span>`;
      const go = async (fn) => {
        const msg = box.querySelector("#amsg");
        msg.textContent = "…";
        try {
          await fn(box.querySelector("#ae").value, box.querySelector("#ap").value);
          const u = await auth.me();
          render(u); onChange(u);
        } catch (err) { msg.textContent = err.message; }
      };
      box.querySelector("#af").onsubmit = (e) => { e.preventDefault(); go(auth.signin.bind(auth)); };
      box.querySelector("#asignup").onclick = () => go(auth.signup.bind(auth));
    }
  };

  // A stored token can be expired; ask the server rather than assume.
  if (auth.token) {
    auth.me().then((u) => { render(u); onChange(u); })
             .catch(() => { render(null); onChange(null); });
  } else {
    render(null);
    onChange(null);
  }

  return { refresh: async () => { try { const u = await auth.me(); render(u); onChange(u); } catch { render(null); onChange(null); } } };
}
