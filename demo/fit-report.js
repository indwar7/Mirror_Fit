/* ─────────────────────────────────────────────────────────────────────────
   Fit report — the "why" behind a fit score.

   Both screens that show a fit number now answer the same question:
   *why that number?* — and they answer it in the places a shopper actually
   feels a garment: chest, shoulder, neck. Length and sleeve follow as
   secondary rows.

   There are two ways a reading gets made, and the UI always says which:

     measured  — the shopper's chest and shoulder against the garment's
                 declared size, through the same geometry as outfit-score.js
                 (fit_score/spec.py). Real numbers, real deviations.

     derived   — no measurements on file. The session still has a score, so
                 the score is decomposed back into the per-area deviations
                 that would produce it, using the SAME weights fitScore()
                 uses. The rows are therefore consistent with the number
                 rather than decorative, but they are a reconstruction and
                 every surface labels them as one.

   Nothing here is random. A reading is seeded from the garment id / session
   key, so the same garment always reports the same story — refreshing the
   page cannot change what a case file says.

   Neck is measured but NOT scored. fit_score's weights are chest, shoulder,
   length, sleeve; adding a fifth would mean claiming a model that was never
   trained. The collar reading is real geometry shown alongside the score,
   and the screen says so in one line rather than quietly folding it in.
   ───────────────────────────────────────────────────────────────────────── */

const FitReport = (() => {

  /* ── Collar / neckline, cm ────────────────────────────────────────────
     GARMENT measurements, matching the SIZE_CHART in outfit-score.js.
     Shirts and jackets carry a real collar circumference. A t-shirt has no
     collar size, so the figure is the neckline opening laid flat and
     doubled — the thing that actually decides whether a crew neck strangles
     or sags. Common Indian menswear ready-to-wear. */
  const NECK_CHART = {
    tshirt: { S: 44, M: 46, L: 48, XL: 50, XXL: 52 },
    shirt:  { S: 38, M: 40, L: 41, XL: 43, XXL: 44 },
    jacket: { S: 41, M: 43, L: 44, XL: 46, XXL: 47 },
  };

  /* How much wider than the neck the garment is meant to be. A dress shirt
     collar is worn with about a finger of room; a crew neck has to clear
     the head, so it is far larger by design and is not "loose" for it. */
  const NECK_EASE = { tshirt: 1.30, shirt: 1.06, jacket: 1.14 };

  /* Neck from chest, the usual menswear approximation: a 102 cm chest runs
     with a ~40 cm collar. Used only when the neck was not measured, which
     is almost always — nobody types their collar size into a mirror. */
  const NECK_FROM_CHEST = 0.39;

  /* ── Tolerance ───────────────────────────────────────────────────────
     |deviation| from the cut's intended ease, as a fraction. Below `good`
     nobody would notice; past `watch` it is the reason a garment goes back.
     WEARABLE_TOLERANCE in outfit-score.js is 0.115, and `watch` sits just
     under it deliberately: the screen should start hedging before the score
     does. */
  const BAND = { good: 0.040, watch: 0.095 };

  /* fitScore()'s weights, repeated here because the decomposition has to
     invert exactly that function. If outfit-score.js changes them, change
     them here too or the rows stop reconstructing the number. */
  const SCORE_WEIGHTS = { chest: 0.38, shoulder: 0.32, length: 0.18, sleeve: 0.12 };
  const WEARABLE_TOLERANCE = 0.115;

  const AREAS = [
    { key: 'chest',    label: 'Chest',    scored: true  },
    { key: 'shoulder', label: 'Shoulder', scored: true  },
    { key: 'neck',     label: 'Neck',     scored: false },
    { key: 'length',   label: 'Length',   scored: true  },
    { key: 'sleeve',   label: 'Sleeve',   scored: true  },
  ];
  const CORE = ['chest', 'shoulder', 'neck'];

  /* ── Deterministic seeding ───────────────────────────────────────────
     Math.random() would mean a case file said something different every
     time an engineer opened it, which is the one thing a case file may not
     do. Everything below is a pure function of a string. */
  function seedFrom(str) {
    let h = 2166136261 >>> 0;
    const s = String(str || 'session');
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  }

  function rng(seed) {
    let x = (seed || 1) >>> 0;
    return function () {
      x ^= x << 13; x >>>= 0;
      x ^= x >>> 17;
      x ^= x << 5;  x >>>= 0;
      return x / 4294967296;
    };
  }

  /* ── Copy ────────────────────────────────────────────────────────────
     One sentence per area per direction, written to be read by someone who
     has never seen a size chart. `cm` is the signed difference between the
     garment and what this cut wants at that point. */
  const LINES = {
    chest: {
      good:  (cm) => `Sat clean across the chest, within ${cm} cm of where this cut wants to sit.`,
      tight: (cm) => `Pulled across the chest — about ${cm} cm narrower than the cut needs. This is the first place it reads small.`,
      loose: (cm) => `About ${cm} cm more room across the chest than the cut intends, so it hangs rather than sits.`,
    },
    shoulder: {
      good:  (cm) => `Shoulder seam landed on the shoulder point, inside ${cm} cm.`,
      tight: (cm) => `Seam sat about ${cm} cm inside the shoulder point — it catches when reaching forward.`,
      loose: (cm) => `Seam dropped about ${cm} cm past the shoulder point, so the sleeve head falls onto the upper arm.`,
    },
    neck: {
      good:  (cm, g) => g === 'tshirt'
        ? `Neckline cleared the collarbone with about ${cm} cm to spare — sits where a crew neck should.`
        : `Collar closed with roughly a finger of room, within ${cm} cm of ideal.`,
      tight: (cm, g) => g === 'tshirt'
        ? `Neckline ran about ${cm} cm tight — it grips at the base of the neck.`
        : `Collar was about ${cm} cm tighter than the neck wants; it presses when buttoned.`,
      loose: (cm, g) => g === 'tshirt'
        ? `Neckline sat about ${cm} cm wide, so it slides off-centre and shows more collarbone than intended.`
        : `Collar stood about ${cm} cm away from the neck and gapes at the front.`,
    },
    length: {
      good:  (cm) => `Hem finished where it should, within ${cm} cm.`,
      tight: (cm) => `Hem finished about ${cm} cm short — it rides up when the arms lift.`,
      loose: (cm) => `Hem ran about ${cm} cm long and covers more of the hip than the cut intends.`,
    },
    sleeve: {
      good:  (cm) => `Sleeve ended where it should, within ${cm} cm.`,
      tight: (cm) => `Sleeve ended about ${cm} cm short of the wrist bone.`,
      loose: (cm) => `Sleeve ran about ${cm} cm past the wrist bone and breaks over the hand.`,
    },
  };

  /* A short hem is not "tight" and a long sleeve is not "roomy" — length and
     sleeve run along the body, the other three run around it, and the verdict
     word has to match the axis or the row reads as nonsense. */
  const WORDS = {
    round: {
      good:  { tight: 'Accurate', loose: 'Accurate' },
      watch: { tight: 'Slightly tight', loose: 'Slightly roomy' },
      off:   { tight: 'Too tight', loose: 'Too roomy' },
    },
    long: {
      good:  { tight: 'Accurate', loose: 'Accurate' },
      watch: { tight: 'Slightly short', loose: 'Slightly long' },
      off:   { tight: 'Too short', loose: 'Too long' },
    },
  };
  const AXIS = { chest: 'round', shoulder: 'round', neck: 'round', length: 'long', sleeve: 'long' };

  /* "chest and shoulder and neck" is what join(' and ') gives you. */
  function list(items) {
    if (items.length <= 1) return items.join('');
    return items.slice(0, -1).join(', ') + ' and ' + items[items.length - 1];
  }

  function toneFor(dev) {
    const a = Math.abs(dev);
    if (a <= BAND.good) return 'good';
    if (a <= BAND.watch) return 'watch';
    return 'off';
  }

  /* Signed deviation -> a row the UI can render without doing any maths. */
  function areaRow(key, dev, referenceCm, garmentType) {
    const tone = toneFor(dev);
    const dir = dev < 0 ? 'tight' : 'loose';
    const cm = Math.abs(dev * referenceCm);
    const shown = cm < 1 ? cm.toFixed(1) : String(Math.round(cm));
    const copy = LINES[key][tone === 'good' ? 'good' : dir];
    const area = AREAS.find(a => a.key === key) || { label: key, scored: true };
    return {
      key, label: area.label, scored: area.scored,
      dev, cm, tone,
      word: tone === 'good' ? 'Accurate' : WORDS[AXIS[key] || 'round'][tone][dir],
      line: copy(shown, garmentType),
    };
  }

  /* ── Neck geometry ───────────────────────────────────────────────────
     Real when a chest measurement exists, because the neck follows the
     chest closely enough for tailoring to have used the ratio for a century.
     Returns a signed deviation on the same scale as the other four. */
  function neckDeviation(garmentType, size, bodyChestCm, bodyNeckCm) {
    const chart = NECK_CHART[garmentType];
    if (!chart || !chart[size]) return null;
    const neck = Number(bodyNeckCm) || (Number(bodyChestCm) ? bodyChestCm * NECK_FROM_CHEST : 0);
    if (!neck) return null;
    const ease = NECK_EASE[garmentType] || NECK_EASE.shirt;
    return (chart[size] / neck) / ease - 1.0;
  }

  /* ── Path 1: measured ────────────────────────────────────────────────
     Uses outfit-score.js so this screen and the live overlay cannot
     disagree about the same garment on the same body. */
  function fromMeasurements({ garmentType, size, body }) {
    if (typeof OutfitScore === 'undefined' || !OutfitScore.model) return null;
    const chart = OutfitScore.SIZE_CHART[garmentType];
    if (!chart || !chart[size]) return null;
    if (!(Number(body.chest_cm) > 0 && Number(body.shoulder_cm) > 0)) return null;

    const b = OutfitScore.estimateBody(body);
    const f = OutfitScore.buildFeatures(b, chart[size], garmentType);
    const score100 = OutfitScore.fitScore(f);
    const g = chart[size];

    const rows = [
      areaRow('chest', f.dev_chest, g[0], garmentType),
      areaRow('shoulder', f.dev_shoulder, g[1], garmentType),
    ];
    const nd = neckDeviation(garmentType, size, b.chest_cm, body.neck_cm);
    if (nd !== null) rows.push(areaRow('neck', nd, NECK_CHART[garmentType][size], garmentType));
    rows.push(areaRow('length', f.dev_length, g[2], garmentType));
    rows.push(areaRow('sleeve', f.dev_sleeve, g[3], garmentType));

    return finish({
      source: 'measured', garmentType, size,
      score10: Math.round(score100) / 10,
      mismatch: OutfitScore.mismatchFromScore(score100),
      rows, estimatedLengths: b.estimated,
    });
  }

  /* ── Path 2: derived from the score ──────────────────────────────────
     Inverts fitScore(). Given a score, there is a total penalty budget:

         penalty = (1 - score/100) / 0.45 * WEARABLE_TOLERANCE
         penalty² = Σ w_k · d_k²

     so any split of penalty² across the four scored areas reproduces the
     score exactly. The split is drawn from the seed, which is why a garment
     id always tells the same story. Rows are consistent with the number by
     construction — but they are a reconstruction, not a measurement, and
     `source` says so for every surface that renders them. */
  function fromScore({ garmentType, size, score10, seed, bias }) {
    const type = NECK_CHART[garmentType] ? garmentType : 'tshirt';
    const sz = (OutfitScore.SIZE_CHART[type] || {})[size] ? size : 'M';
    const g = (OutfitScore.SIZE_CHART[type] || OutfitScore.SIZE_CHART.tshirt)[sz];
    const rand = rng(seedFrom(seed));

    const score100 = Math.max(0, Math.min(100, score10 * 10));
    const penalty = ((1 - score100 / 100) / 0.45) * WEARABLE_TOLERANCE;
    const p2 = penalty * penalty;

    const keys = ['chest', 'shoulder', 'length', 'sleeve'];
    // Shares of the squared penalty. The floor stops one area swallowing
    // everything and leaving three rows saying "perfect" on an 6/10.
    const shares = keys.map(() => 0.12 + rand());
    const total = shares.reduce((a, b) => a + b, 0);

    const devs = {};
    keys.forEach((k, i) => {
      const share = shares[i] / total;
      const mag = Math.sqrt((share * p2) / SCORE_WEIGHTS[k]);
      // `bias` lets a case file say which way it went wrong — a shopper who
      // returned something as "too small" should not read a report full of
      // roomy findings.
      const sign = bias === 'tight' ? -1 : bias === 'loose' ? 1 : (rand() < 0.5 ? -1 : 1);
      devs[k] = mag * sign;
    });

    // Neck rides at the scale of the two it sits between, so a garment that
    // was wrong at the chest and shoulder does not report a perfect collar.
    const neckMag = (Math.abs(devs.chest) + Math.abs(devs.shoulder)) / 2 * (0.55 + rand() * 0.6);
    const neckSign = bias === 'tight' ? -1 : bias === 'loose' ? 1 : (devs.chest < 0 ? -1 : 1);
    const neckDev = neckMag * neckSign;

    const rows = [
      areaRow('chest', devs.chest, g[0], type),
      areaRow('shoulder', devs.shoulder, g[1], type),
      areaRow('neck', neckDev, (NECK_CHART[type] || NECK_CHART.tshirt)[sz], type),
      areaRow('length', devs.length, g[2], type),
      areaRow('sleeve', devs.sleeve, g[3], type),
    ];

    return finish({
      source: 'derived', garmentType: type, size: sz,
      score10: Math.round(score100) / 10,
      mismatch: Math.max(0, Math.min(1, (100 - score100) / 100)),
      rows, estimatedLengths: true,
    });
  }

  /* ── Verdict + headline ──────────────────────────────────────────────
     The one line the shopper reads first, and the reason it says that. */
  function finish(r) {
    r.core = r.rows.filter(x => CORE.includes(x.key));
    r.secondary = r.rows.filter(x => !CORE.includes(x.key));
    r.worst = r.rows.slice().sort((a, b) => Math.abs(b.dev) - Math.abs(a.dev))[0];
    r.offCount = r.rows.filter(x => x.tone === 'off').length;
    r.watchCount = r.rows.filter(x => x.tone === 'watch').length;

    r.verdict = r.score10 >= 8.6 ? 'Excellent fit'
              : r.score10 >= 7.5 ? 'Good fit'
              : r.score10 >= 6.0 ? 'Wearable, with one problem'
              : 'Wrong size';

    const clean = r.core.filter(x => x.tone === 'good').map(x => x.label.toLowerCase());
    if (r.offCount === 0 && r.watchCount === 0) {
      r.headline = `Nothing was off. Chest, shoulder and neck all landed inside the tolerance this cut is drawn to, so the score has nothing to deduct for.`;
    } else if (r.worst.tone === 'good') {
      r.headline = `Everything landed inside tolerance. The score stops short of full marks only because ${r.worst.label.toLowerCase()} is the closest to the edge.`;
    } else {
      const rest = clean.length
        ? ` The ${list(clean)} ${clean.length === 1 ? 'was' : 'were'} accurate.`
        : '';
      r.headline = `The score came down to the ${r.worst.label.toLowerCase()}: ${r.worst.word.toLowerCase()}` +
        ` by about ${r.worst.cm < 1 ? r.worst.cm.toFixed(1) : Math.round(r.worst.cm)} cm.` + rest;
    }
    return r;
  }

  /* ── Case files ──────────────────────────────────────────────────────
     What a returns engineer looks up. These are demo records — the screen
     labels them so — but every field is one the live system already has at
     the moment a return is opened, and the decision is taken by the real
     returns model, not written here. */
  const CASES = [
    {
      /* The case the whole screen exists for: a high score and a size
         complaint. The mirror measured the chest as accurate — if anything
         roomy — so the claim and the measurement point opposite ways. */
      id: 'MF-SHR-4471', garment: 'Oxford shirt, mid blue', garmentType: 'shirt', size: 'L',
      customer: 'Renter #8842', reason: 'doesnt_fit', reasonSaid: 'Too small across the chest',
      claimArea: 'chest', claimDir: 'tight',
      condition: 'Good — no marks, tags intact, worn once',
      value: 2450, days: 9, orders: 5, returns: 1, accountAge: 540, discount: 12,
      score10: 8.4, bias: 'loose',
    },
    {
      /* The interesting one. The headline score is fine, and the shopper is
         still right — because the collar is the one point the score does not
         weigh. This is the case that justifies showing neck at all. */
      id: 'MF-TEE-1180', garment: 'Heavyweight tee, black', garmentType: 'tshirt', size: 'M',
      customer: 'Renter #2317', reason: 'doesnt_fit', reasonSaid: 'Collar too tight',
      claimArea: 'neck', claimDir: 'tight',
      condition: 'Good — as sent',
      value: 1180, days: 6, orders: 3, returns: 0, accountAge: 880, discount: 0,
      score10: 6.9, bias: 'tight',
    },
    {
      id: 'MF-JKT-9026', garment: 'Unstructured jacket, stone', garmentType: 'jacket', size: 'XL',
      customer: 'Renter #4409', reason: 'doesnt_fit', reasonSaid: 'Shoulders too big',
      claimArea: 'shoulder', claimDir: 'loose',
      condition: 'Good — no damage',
      value: 6800, days: 24, orders: 12, returns: 9, accountAge: 372, discount: 4.8,
      score10: 8.8, bias: 'loose',
    },
    {
      /* The failure that belongs to the sale, not to the returns desk. The
         mirror already knew, and the garment shipped anyway. */
      id: 'MF-SHR-7734', garment: 'Linen shirt, sand', garmentType: 'shirt', size: 'M',
      customer: 'Renter #1902', reason: 'doesnt_fit', reasonSaid: 'Runs small everywhere',
      claimArea: null, claimDir: 'tight',
      condition: 'Good — unworn',
      value: 1990, days: 4, orders: 8, returns: 2, accountAge: 1240, discount: 20,
      score10: 4.0, bias: 'tight',
    },
    {
      id: 'MF-TEE-5563', garment: 'Pocket tee, ecru', garmentType: 'tshirt', size: 'L',
      customer: 'Renter #7781', reason: 'not_as_described', reasonSaid: 'Fabric felt thinner than the listing',
      claimArea: null, claimDir: null,
      condition: 'Fair — light pilling at the hem',
      value: 1091, days: 10, orders: 11, returns: 9, accountAge: 146, discount: 41.9,
      score10: 8.1, bias: null,
    },
    {
      id: 'MF-JKT-3315', garment: 'Quilted jacket, navy', garmentType: 'jacket', size: 'M',
      customer: 'Renter #5528', reason: 'damaged', reasonSaid: 'Zip pull came away',
      claimArea: null, claimDir: null,
      condition: 'Damaged — zip pull detached, confirmed at intake',
      value: 5400, days: 3, orders: 2, returns: 0, accountAge: 95, discount: 0,
      score10: 8.9, bias: null,
    },
  ];

  /* Any id an engineer types resolves to something. An unknown id gets a
     record built deterministically from the id itself, clearly marked as
     generated, so the screen can be demonstrated without a seeded database
     — and so a typo never produces a blank page with no explanation. */
  function synthesise(id) {
    const rand = rng(seedFrom('case:' + id));
    const types = ['tshirt', 'shirt', 'jacket'];
    const names = { tshirt: 'Cotton tee', shirt: 'Casual shirt', jacket: 'Light jacket' };
    const t = types[Math.floor(rand() * 3)];
    const sizes = ['S', 'M', 'L', 'XL'];
    const reasons = ['doesnt_fit', 'doesnt_fit', 'not_as_described', 'changed_mind'];
    const reason = reasons[Math.floor(rand() * reasons.length)];
    const orders = 2 + Math.floor(rand() * 12);
    return {
      id, generated: true,
      garment: names[t], garmentType: t, size: sizes[Math.floor(rand() * sizes.length)],
      customer: 'Renter #' + (1000 + Math.floor(rand() * 8999)),
      reason,
      reasonSaid: reason === 'doesnt_fit' ? 'Size was wrong'
                : reason === 'not_as_described' ? 'Not what the listing showed'
                : 'Changed their mind',
      condition: 'Good — no damage',
      value: 900 + Math.floor(rand() * 6000),
      days: 2 + Math.floor(rand() * 26),
      orders, returns: Math.floor(rand() * (orders + 1)),
      accountAge: 40 + Math.floor(rand() * 1500),
      discount: Math.round(rand() * 400) / 10,
      score10: Math.round((5.5 + rand() * 3.4) * 10) / 10,
      bias: rand() < 0.5 ? 'tight' : 'loose',
      claimArea: null, claimDir: null,
    };
  }

  function lookup(query) {
    const q = String(query || '').trim().toUpperCase();
    if (!q) return null;
    const hit = CASES.find(c => c.id.toUpperCase() === q)
             || CASES.find(c => c.id.toUpperCase().replace(/-/g, '') === q.replace(/-/g, ''));
    return hit || synthesise(q);
  }

  /* The reading attached to a case. Measured readings are not available at
     the returns desk — the mirror ran weeks earlier and only the score was
     stored — so a case always reconstructs from its stored score. */
  function readingForCase(c) {
    return fromScore({
      garmentType: c.garmentType, size: c.size,
      score10: c.score10, seed: c.id, bias: c.bias,
    });
  }

  return {
    seedFrom, rng, fromMeasurements, fromScore, readingForCase,
    lookup, areaRow, neckDeviation, toneFor,
    get cases() { return CASES; },
    NECK_CHART, NECK_EASE, BAND, AREAS, CORE,
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = FitReport;
