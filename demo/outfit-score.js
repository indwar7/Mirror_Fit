/* ─────────────────────────────────────────────────────────────────────────
   Outfit rating for live try-on.

   Gives the shopper a 0-10 rating of what they are wearing, from two parts:

     fit     — geometry, via the fit_score model (fit_model.json)
     colour  — the garment's dominant colour against their skin tone

   What this deliberately does NOT do
   ----------------------------------
   It does not read fit off the try-on render. CatVTON warps the garment onto
   the torso mask, so the output *always* looks like it fits — measuring it
   would produce a confident number that means nothing. Fit here comes from the
   shopper's own measurements against the garment's declared size, which is
   what fit_score actually models.

   Which model, and why not the other one
   --------------------------------------
   fit_model.json (geometric, 5-class) is used. fit_model_real.json is trained
   on real outcomes and is the better model, but its features are RentTheRunway
   specific — bust_cup, age, and collaborative rates like item_small_rate and
   user_small_rate. None of those exist for a shopper pointing a webcam at
   themselves in an uploaded garment, and the README is explicit that without
   the collaborative signal it falls back to the global prior, which answers
   "fit" every time. A model that cannot disagree is not worth asking.

   The guard
   ---------
   Advice is suppressed below a confidence bar, mirroring real_model.gate().
   That gate is the difference between the fit work preventing returns and
   causing them: ungated it caused 14,334 returns against 6,592 prevented — a
   net of -7,742. Staying quiet is a feature, not a failure.
   ───────────────────────────────────────────────────────────────────────── */

const OutfitScore = (() => {
  let MODEL = null;

  /* fit_model.json ships no gate of its own — the measured one belongs to the
     real model. 0.60 is chosen, not learned: with five classes chance is 0.20,
     so this is a meaningful bar while still letting the model speak on the
     clear-cut cases the geometry actually settles. Raise it if shoppers report
     bad sizing advice; the cost of raising it is only silence. */
  const MIN_CONFIDENCE = 0.60;

  /* ── Garment size charts ──────────────────────────────────────────────
     GARMENT measurements in cm — the finished garment, not the body it fits.
     Chest is the full circumference. Common Indian menswear ready-to-wear.

     A declared size chart is used rather than measuring the uploaded garment
     photo. Measuring a photographed garment needs a length reference that a
     product shot does not carry, so any number derived from it would be scaled
     by a guess. A size label is data the shopper actually has. */
  const SIZE_CHART = {
    //        [chest, shoulder, length, sleeve]
    tshirt: {
      S: [96, 42, 68, 19], M: [102, 44, 70, 20], L: [108, 46, 72, 21],
      XL: [114, 48, 74, 22], XXL: [120, 50, 76, 23],
    },
    shirt: {
      S: [100, 43, 72, 59], M: [106, 45, 74, 60], L: [112, 47, 76, 61],
      XL: [118, 49, 78, 62], XXL: [124, 51, 80, 63],
    },
    jacket: {
      S: [106, 45, 68, 61], M: [112, 47, 70, 62], L: [118, 49, 72, 63],
      XL: [124, 51, 74, 64], XXL: [130, 53, 76, 65],
    },
  };

  /* T-shirts in the chart are short-sleeved, but the model's sleeve_ratio is
     garment sleeve against the wearer's whole ARM (IDEAL_EASE puts tshirt
     sleeve at 1.00, i.e. a full-length sleeve). Feeding a 20 cm sleeve against
     a 58 cm arm would read as catastrophically undersized on a garment that is
     simply short-sleeved. So sleeve is neutralised for these — scored as
     exactly ideal, which is the same as not scoring it. */
  const SHORT_SLEEVED = new Set(['tshirt']);

  const SIZES = ['S', 'M', 'L', 'XL', 'XXL'];

  /* ── Body measurements ───────────────────────────────────────────────── */

  /* Torso and arm are rarely known off-hand, so they are estimated from height
     when not supplied. Ratios are the usual tailoring approximations (nape-to-
     hem ≈ 0.41 x height, shoulder-to-wrist ≈ 0.33 x height). They are estimates
     and the UI says so — chest and shoulder are the two the model leans on
     anyway (spec.py derives core_dev_mean from exactly those two). */
  function estimateBody(body) {
    const h = Number(body.height_cm) || 0;
    return {
      chest_cm: Number(body.chest_cm),
      shoulder_cm: Number(body.shoulder_cm),
      torso_cm: Number(body.torso_cm) || (h ? h * 0.41 : 0),
      arm_cm: Number(body.arm_cm) || (h ? h * 0.33 : 0),
      estimated: !body.torso_cm || !body.arm_cm,
    };
  }

  /* ── Features ─────────────────────────────────────────────────────────
     Mirrors fit_score/spec.py derive() and fit_score/data.py, minus the
     training-time noise terms. If either changes, this must change with it. */
  function buildFeatures(body, garment, garmentType) {
    const ease = MODEL.ease[garmentType] || MODEL.ease.tshirt;
    const [easeChest, easeShoulder, easeLength, easeSleeve] = ease;

    const chest_ratio = garment[0] / body.chest_cm;
    const shoulder_ratio = garment[1] / body.shoulder_cm;
    const length_ratio = body.torso_cm ? garment[2] / body.torso_cm : easeLength;
    const sleeve_ratio = SHORT_SLEEVED.has(garmentType)
      ? easeSleeve                                   // neutralised — see above
      : (body.arm_cm ? garment[3] / body.arm_cm : easeSleeve);

    // data.py: (chest_ratio / ease - 1) * 2.4, clipped.
    const drape_slack = Math.max(-0.35, Math.min(1.0,
      (chest_ratio / easeChest - 1.0) * 2.4));
    // data.py: how far the shoulder seam falls past the shoulder point, in cm.
    const shoulder_drop_cm =
      (shoulder_ratio / easeShoulder - 1.0) * body.shoulder_cm;

    // spec.py deviations(): signed distance from ideal, as a fraction.
    const dev_chest = chest_ratio / easeChest - 1.0;
    const dev_shoulder = shoulder_ratio / easeShoulder - 1.0;
    const dev_length = length_ratio / easeLength - 1.0;
    const dev_sleeve = sleeve_ratio / easeSleeve - 1.0;
    const devs = [dev_chest, dev_shoulder, dev_length, dev_sleeve];

    return {
      chest_ratio, shoulder_ratio, length_ratio, sleeve_ratio,
      drape_slack, shoulder_drop_cm,
      dev_chest, dev_shoulder, dev_length, dev_sleeve,
      // These three are what make the classes linearly separable at all —
      // see the long comment in spec.py derive().
      worst_abs_dev: Math.max(...devs.map(Math.abs)),
      dev_spread: Math.max(...devs) - Math.min(...devs),
      core_dev_mean: (dev_chest + dev_shoulder) / 2,
      garment_type: garmentType,
    };
  }

  /* Standardise, one-hot, multinomial logistic, softmax. Same layout as the
     fit-analyser block in index.html, minus the imputation the real model
     needs and this one does not ship. */
  function classify(v) {
    const P = MODEL.preprocess, F = MODEL.features;
    const row = F.numeric.map((name, i) =>
      ((Number(v[name]) || 0) - P.scaler_mean[i]) / P.scaler_scale[i]);

    F.categorical.forEach((col) => {
      P.categories[col].forEach(level => row.push(v[col] === level ? 1 : 0));
    });

    const logits = MODEL.model.coef.map((c, k) =>
      c.reduce((s, w, i) => s + w * (row[i] || 0), MODEL.model.intercept[k]));
    const top = Math.max(...logits);
    const exp = logits.map(z => Math.exp(z - top));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probs = exp.map(z => z / sum);
    const best = probs.indexOf(Math.max(...probs));
    return { cls: MODEL.classes[best], confidence: probs[best], probs };
  }

  /* ── Colour ───────────────────────────────────────────────────────────── */

  /* Dominant colour of the garment, ignoring transparent and near-white
     pixels — product shots are mostly background, and averaging that in turns
     every garment beige. Sampled on a stride so a large PNG is still cheap. */
  function dominantColour(source) {
    const c = document.createElement('canvas');
    const w = c.width = Math.min(source.naturalWidth || source.width || 128, 128);
    const h = c.height = Math.min(source.naturalHeight || source.height || 128, 128);
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(source, 0, 0, w, h);

    let data;
    try {
      data = ctx.getImageData(0, 0, w, h).data;
    } catch (_) {
      return null; // tainted canvas — a cross-origin garment image
    }

    let r = 0, g = 0, b = 0, n = 0;
    for (let i = 0; i < data.length; i += 4) {
      const alpha = data[i + 3];
      if (alpha < 128) continue;
      const [pr, pg, pb] = [data[i], data[i + 1], data[i + 2]];
      // Skip the near-white / near-black extremes that are usually backdrop
      // or shadow rather than the garment itself.
      const max = Math.max(pr, pg, pb), min = Math.min(pr, pg, pb);
      if (max > 245 && min > 235) continue;
      if (max < 18) continue;
      r += pr; g += pg; b += pb; n++;
    }
    if (!n) return null;
    return { r: Math.round(r / n), g: Math.round(g / n), b: Math.round(b / n) };
  }

  /* Warm / cool / neutral from the pixel itself.

     StyleEngine.checkColorHarmony in the Flutter app does this from colour
     NAMES off a catalogue item. Live try-on has a photograph and no name, so
     the same three buckets are derived from hue and saturation instead. The
     verdicts and scores are kept identical to the Dart version so the two
     surfaces cannot disagree about the same garment. */
  function colourFamily(rgb) {
    const r = rgb.r / 255, g = rgb.g / 255, b = rgb.b / 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const sat = max === 0 ? 0 : (max - min) / max;

    // Low saturation is neutral regardless of hue — white, black, grey, charcoal.
    if (sat < 0.18) return 'neutral';

    let hue;
    const d = max - min;
    if (d === 0) hue = 0;
    else if (max === r) hue = 60 * (((g - b) / d) % 6);
    else if (max === g) hue = 60 * ((b - r) / d + 2);
    else hue = 60 * ((r - g) / d + 4);
    if (hue < 0) hue += 360;

    // Reds through yellows and the browns/olives read warm; greens through
    // violets read cool.
    return (hue < 75 || hue >= 330) ? 'warm' : 'cool';
  }

  const WARM_SKIN = new Set(['olive', 'medium', 'brown']);

  function colourHarmony(rgb, skinTone) {
    const family = colourFamily(rgb);
    const warmSkin = WARM_SKIN.has(skinTone);

    if (family === 'neutral') {
      return { score: 9.0, verdict: 'Perfect neutral — works with every skin tone', isGreat: true, family };
    }
    if ((warmSkin && family === 'warm') || (!warmSkin && family === 'cool')) {
      return { score: 8.5, verdict: 'Excellent match for your skin tone', isGreat: true, family };
    }
    return {
      score: 6.5,
      verdict: warmSkin
        ? 'A warmer shade would sit better against your skin tone'
        : 'A cooler shade would sit better against your skin tone',
      isGreat: false,
      family,
    };
  }

  /* ── Combined rating ──────────────────────────────────────────────────── */

  // Same bands as StyleScore in the Flutter app, so a 8.4 means the same
  // thing in both places.
  function grade(overall) {
    if (overall >= 9) return { grade: 'S', emoji: '🔥' };
    if (overall >= 8) return { grade: 'A', emoji: '⭐' };
    if (overall >= 7) return { grade: 'B', emoji: '✅' };
    if (overall >= 6) return { grade: 'C', emoji: '🤔' };
    return { grade: 'D', emoji: '😐' };
  }

  // Fit class → a 0-10 contribution. Ordered as the classes are: best first.
  const FIT_POINTS = {
    MADE_FOR_YOU: 9.6, GOOD_FIT: 8.4, OVERSIZED: 6.4, UNDERSIZED: 6.0, POOR_FIT: 4.8,
  };

  /**
   * Rate an outfit.
   *
   * `fit` is omitted entirely when measurements or size are missing, or when
   * the model is not confident enough — the rating then reflects colour alone
   * and says so, rather than inventing a fit number to fill the space.
   */
  function rate({ garmentImage, garmentType, size, body, skinTone }) {
    const out = { ready: !!MODEL, fit: null, colour: null, overall: null, notes: [] };
    if (!MODEL) {
      out.notes.push('Fit model still loading.');
      return out;
    }

    if (garmentImage) {
      const rgb = dominantColour(garmentImage);
      if (rgb) out.colour = { ...colourHarmony(rgb, skinTone || 'medium'), rgb };
      else out.notes.push('Could not read the garment colour.');
    }

    const chart = SIZE_CHART[garmentType];
    const haveBody = body && Number(body.chest_cm) > 0 && Number(body.shoulder_cm) > 0;
    const haveSize = chart && size && chart[size];

    if (haveBody && haveSize) {
      const b = estimateBody(body);
      const features = buildFeatures(b, chart[size], garmentType);
      const { cls, confidence, probs } = classify(features);

      // The guard. Below the bar we report the reading but withhold the
      // verdict — see the header comment for why silence is the safe answer.
      const confident = confidence >= MIN_CONFIDENCE;
      const copy = MODEL.class_copy[cls] || [cls, ''];

      const score100 = fitScore(features);
      out.fit = {
        cls, confidence, probs, confident,
        // Carried on every rating, gated or not: the returns model wants the
        // measured geometry, and withholding *advice* from the shopper is a
        // different decision from withholding a *feature* from another model.
        score100,
        mismatch: mismatchFromScore(score100),
        label: confident ? copy[0] : 'No fit call',
        detail: confident
          ? copy[1]
          : `The model leans "${copy[0]}" but only at ${(confidence * 100).toFixed(0)}% ` +
            `confidence, under the ${(MIN_CONFIDENCE * 100).toFixed(0)}% bar. Staying quiet: ` +
            `bad sizing advice moves a shopper to a worse size and causes the return it ` +
            `meant to prevent.`,
        estimatedLengths: b.estimated,
        features,
      };
      if (b.estimated) {
        out.notes.push('Length and sleeve are estimated from height.');
      }
    } else if (!haveBody) {
      out.notes.push('Add your chest and shoulder measurements for a fit rating.');
    } else if (!haveSize) {
      out.notes.push('Pick the garment size for a fit rating.');
    }

    // Weighting: fit is the half of this that has a trained model behind it, so
    // it carries more when present. With no fit call, colour stands alone
    // rather than being diluted by a neutral stand-in.
    const fitPoints = out.fit && out.fit.confident ? FIT_POINTS[out.fit.cls] : null;
    const colourPoints = out.colour ? out.colour.score : null;

    if (fitPoints !== null && colourPoints !== null) {
      out.overall = Math.round((fitPoints * 0.6 + colourPoints * 0.4) * 10) / 10;
    } else if (colourPoints !== null) {
      out.overall = Math.round(colourPoints * 10) / 10;
      out.notes.push('Rating is colour only — no fit call.');
    } else if (fitPoints !== null) {
      out.overall = Math.round(fitPoints * 10) / 10;
    }

    if (out.overall !== null) Object.assign(out, grade(out.overall));
    return out;
  }

  /* ── The join to Returns Guard ────────────────────────────────────────
     Ports fit_score/spec.py fit_score() and mismatch_from_score() exactly. Do
     not re-derive these from the class probabilities: the returns model was
     trained against THIS quantity, and the fit README is explicit that this is
     the join between the two systems — "what the mirror measured at purchase
     is what the returns desk sees weeks later". A different formula here would
     silently feed returns_risk a feature it was not trained on. */
  const WEARABLE_TOLERANCE = 0.115;
  const SCORE_WEIGHTS = { chest: 0.38, shoulder: 0.32, length: 0.18, sleeve: 0.12 };

  /** 0-100. How close this garment is to being cut for this body. */
  function fitScore(features) {
    const d = {
      chest: features.dev_chest, shoulder: features.dev_shoulder,
      length: features.dev_length, sleeve: features.dev_sleeve,
    };
    let sum = 0;
    for (const k of Object.keys(SCORE_WEIGHTS)) sum += SCORE_WEIGHTS[k] * d[k] * d[k];
    const penalty = Math.sqrt(sum);
    const score = 100 * (1 - (penalty / WEARABLE_TOLERANCE) * 0.45);
    return Math.max(0, Math.min(100, score));
  }

  /** The 0-1 `fit_mismatch_score` the returns model consumes. */
  function mismatchFromScore(score) {
    return Math.max(0, Math.min(1, (100 - score) / 100));
  }

  async function load(url = 'fit_model.json') {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} not found (${res.status})`);
    MODEL = await res.json();
    return MODEL;
  }

  return {
    load, rate, grade,
    dominantColour, colourFamily, colourHarmony, buildFeatures, classify, estimateBody,
    fitScore, mismatchFromScore,
    SIZES, SIZE_CHART, MIN_CONFIDENCE, WEARABLE_TOLERANCE,
    get model() { return MODEL; },
  };
})();

if (typeof module !== 'undefined' && module.exports) module.exports = OutfitScore;
