/* ─────────────────────────────────────────────────────────────────────────
   Outfit rating for live try-on.

   Gives the shopper a 0-10 rating of what they are wearing, from two parts:

     fit     — geometry, via the fit_score model (fit_model.json)
     colour  — the garment's dominant colour against their skin tone

   It does NOT read fit off the try-on render. CatVTON warps the garment onto
   the torso mask, so the output *always* looks like it fits. Fit here comes
   from the shopper's own measurements against the garment's declared size,
   which is what fit_score actually models.

   fit_model.json (geometric, 5-class) is used. fit_model_real.json is trained
   on real outcomes but its features are RentTheRunway specific and do not
   exist for a shopper pointing a webcam at themselves.

   Advice is suppressed below a confidence bar, mirroring real_model.gate().
   Ungated, the fit work caused 14,334 returns against 6,592 prevented.
   Staying quiet is a feature, not a failure.
   ───────────────────────────────────────────────────────────────────────── */

export type GarmentType = 'tshirt' | 'shirt' | 'jacket';
export type Size = 'S' | 'M' | 'L' | 'XL' | 'XXL';
/** [chest, shoulder, length, sleeve] of the finished garment, cm. */
export type GarmentDims = [number, number, number, number];

export interface FitModel {
  ease: Record<string, GarmentDims>;
  preprocess: {
    scaler_mean: number[];
    scaler_scale: number[];
    categories: Record<string, string[]>;
  };
  features: { numeric: string[]; categorical: string[] };
  model: { coef: number[][]; intercept: number[] };
  classes: string[];
  class_copy: Record<string, [string, string]>;
}

export interface BodyInput {
  chest_cm?: number | string;
  shoulder_cm?: number | string;
  height_cm?: number | string;
  torso_cm?: number | string;
  arm_cm?: number | string;
  neck_cm?: number | string;
}

export interface Body {
  chest_cm: number;
  shoulder_cm: number;
  torso_cm: number;
  arm_cm: number;
  estimated: boolean;
}

export interface Features {
  chest_ratio: number; shoulder_ratio: number; length_ratio: number; sleeve_ratio: number;
  drape_slack: number; shoulder_drop_cm: number;
  dev_chest: number; dev_shoulder: number; dev_length: number; dev_sleeve: number;
  worst_abs_dev: number; dev_spread: number; core_dev_mean: number;
  garment_type: string;
  [k: string]: number | string;
}

export interface FitCall {
  cls: string; confidence: number; probs: number[]; confident: boolean;
  score100: number; mismatch: number;
  label: string; detail: string;
  estimatedLengths: boolean;
  features: Features;
}

export interface ColourCall {
  score: number; verdict: string; isGreat: boolean; family: 'warm' | 'cool' | 'neutral';
  rgb: { r: number; g: number; b: number };
}

export interface Rating {
  ready: boolean;
  fit: FitCall | null;
  colour: ColourCall | null;
  overall: number | null;
  notes: string[];
  grade?: string;
  emoji?: string;
}

let MODEL: FitModel | null = null;

/* fit_model.json ships no gate of its own. 0.60 is chosen, not learned: with
   five classes chance is 0.20, so this is a meaningful bar while still
   letting the model speak on the clear-cut cases. */
const MIN_CONFIDENCE = 0.60;

/* GARMENT measurements in cm — the finished garment, not the body it fits.
   Common Indian menswear ready-to-wear. A declared size chart is used rather
   than measuring the uploaded garment photo: a product shot carries no
   length reference. */
const SIZE_CHART: Record<GarmentType, Record<Size, GarmentDims>> = {
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

/* T-shirts are short-sleeved, but the model's sleeve_ratio is garment sleeve
   against the wearer's whole ARM. So sleeve is neutralised for these —
   scored as exactly ideal. */
const SHORT_SLEEVED = new Set<string>(['tshirt']);

const SIZES: Size[] = ['S', 'M', 'L', 'XL', 'XXL'];

/* Torso and arm are estimated from height when not supplied (nape-to-hem ≈
   0.41 x height, shoulder-to-wrist ≈ 0.33 x height). */
function estimateBody(body: BodyInput): Body {
  const h = Number(body.height_cm) || 0;
  return {
    chest_cm: Number(body.chest_cm),
    shoulder_cm: Number(body.shoulder_cm),
    torso_cm: Number(body.torso_cm) || (h ? h * 0.41 : 0),
    arm_cm: Number(body.arm_cm) || (h ? h * 0.33 : 0),
    estimated: !body.torso_cm || !body.arm_cm,
  };
}

/* Mirrors fit_score/spec.py derive() and fit_score/data.py, minus the
   training-time noise terms. If either changes, this must change with it. */
function buildFeatures(body: Body, garment: GarmentDims, garmentType: string): Features {
  const model = MODEL!;
  const ease = model.ease[garmentType] || model.ease.tshirt;
  const [easeChest, easeShoulder, easeLength, easeSleeve] = ease;

  const chest_ratio = garment[0] / body.chest_cm;
  const shoulder_ratio = garment[1] / body.shoulder_cm;
  const length_ratio = body.torso_cm ? garment[2] / body.torso_cm : easeLength;
  const sleeve_ratio = SHORT_SLEEVED.has(garmentType)
    ? easeSleeve                                   // neutralised — see above
    : (body.arm_cm ? garment[3] / body.arm_cm : easeSleeve);

  // data.py: (chest_ratio / ease - 1) * 2.4, clipped.
  const drape_slack = Math.max(-0.35, Math.min(1.0, (chest_ratio / easeChest - 1.0) * 2.4));
  // data.py: how far the shoulder seam falls past the shoulder point, in cm.
  const shoulder_drop_cm = (shoulder_ratio / easeShoulder - 1.0) * body.shoulder_cm;

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
    worst_abs_dev: Math.max(...devs.map(Math.abs)),
    dev_spread: Math.max(...devs) - Math.min(...devs),
    core_dev_mean: (dev_chest + dev_shoulder) / 2,
    garment_type: garmentType,
  };
}

/* Standardise, one-hot, multinomial logistic, softmax. */
function classify(v: Features): { cls: string; confidence: number; probs: number[] } {
  const model = MODEL!;
  const P = model.preprocess, F = model.features;
  const row = F.numeric.map((name, i) => ((Number(v[name]) || 0) - P.scaler_mean[i]) / P.scaler_scale[i]);

  F.categorical.forEach((col) => {
    P.categories[col].forEach((level) => row.push(v[col] === level ? 1 : 0));
  });

  const logits = model.model.coef.map((c, k) =>
    c.reduce((s, w, i) => s + w * (row[i] || 0), model.model.intercept[k]));
  const top = Math.max(...logits);
  const exp = logits.map((z) => Math.exp(z - top));
  const sum = exp.reduce((a, b) => a + b, 0);
  const probs = exp.map((z) => z / sum);
  const best = probs.indexOf(Math.max(...probs));
  return { cls: model.classes[best], confidence: probs[best], probs };
}

/* Dominant colour of the garment, ignoring transparent and near-white pixels.
   Sampled on a stride so a large PNG is still cheap. */
function dominantColour(source: HTMLImageElement | HTMLCanvasElement): { r: number; g: number; b: number } | null {
  const c = document.createElement('canvas');
  const natural = source as HTMLImageElement;
  const w = c.width = Math.min(natural.naturalWidth || source.width || 128, 128);
  const h = c.height = Math.min(natural.naturalHeight || source.height || 128, 128);
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(source, 0, 0, w, h);

  let data: Uint8ClampedArray;
  try {
    data = ctx.getImageData(0, 0, w, h).data;
  } catch {
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

/* Warm / cool / neutral from the pixel itself. Same three buckets and the
   same verdicts as StyleEngine.checkColorHarmony in the Flutter app. */
function colourFamily(rgb: { r: number; g: number; b: number }): 'warm' | 'cool' | 'neutral' {
  const r = rgb.r / 255, g = rgb.g / 255, b = rgb.b / 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const sat = max === 0 ? 0 : (max - min) / max;

  // Low saturation is neutral regardless of hue.
  if (sat < 0.18) return 'neutral';

  let hue: number;
  const d = max - min;
  if (d === 0) hue = 0;
  else if (max === r) hue = 60 * (((g - b) / d) % 6);
  else if (max === g) hue = 60 * ((b - r) / d + 2);
  else hue = 60 * ((r - g) / d + 4);
  if (hue < 0) hue += 360;

  return (hue < 75 || hue >= 330) ? 'warm' : 'cool';
}

const WARM_SKIN = new Set(['olive', 'medium', 'brown']);

function colourHarmony(rgb: { r: number; g: number; b: number }, skinTone: string): Omit<ColourCall, 'rgb'> {
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

// Same bands as StyleScore in the Flutter app.
function grade(overall: number): { grade: string; emoji: string } {
  if (overall >= 9) return { grade: 'S', emoji: '🔥' };
  if (overall >= 8) return { grade: 'A', emoji: '⭐' };
  if (overall >= 7) return { grade: 'B', emoji: '✅' };
  if (overall >= 6) return { grade: 'C', emoji: '🤔' };
  return { grade: 'D', emoji: '😐' };
}

// Fit class → a 0-10 contribution.
const FIT_POINTS: Record<string, number> = {
  MADE_FOR_YOU: 9.6, GOOD_FIT: 8.4, OVERSIZED: 6.4, UNDERSIZED: 6.0, POOR_FIT: 4.8,
};

export interface RateInput {
  garmentImage: HTMLImageElement | null;
  garmentType: string;
  size: string;
  body: BodyInput;
  skinTone: string;
}

/** Rate an outfit. `fit` is omitted when measurements or size are missing,
 *  or when the model is not confident enough. */
function rate({ garmentImage, garmentType, size, body, skinTone }: RateInput): Rating {
  const out: Rating = { ready: !!MODEL, fit: null, colour: null, overall: null, notes: [] };
  if (!MODEL) {
    out.notes.push('Fit model still loading.');
    return out;
  }

  if (garmentImage) {
    const rgb = dominantColour(garmentImage);
    if (rgb) out.colour = { ...colourHarmony(rgb, skinTone || 'medium'), rgb };
    else out.notes.push('Could not read the garment colour.');
  }

  const chart = SIZE_CHART[garmentType as GarmentType];
  const haveBody = body && Number(body.chest_cm) > 0 && Number(body.shoulder_cm) > 0;
  const haveSize = chart && size && chart[size as Size];

  if (haveBody && haveSize) {
    const b = estimateBody(body);
    const features = buildFeatures(b, chart[size as Size], garmentType);
    const { cls, confidence, probs } = classify(features);

    // The guard. Below the bar we report the reading but withhold the verdict.
    const confident = confidence >= MIN_CONFIDENCE;
    const copy = MODEL.class_copy[cls] || [cls, ''];

    const score100 = fitScore(features);
    out.fit = {
      cls, confidence, probs, confident,
      // Carried on every rating, gated or not: the returns model wants the
      // measured geometry.
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

  // Fit carries more when present; with no fit call, colour stands alone.
  const fitPoints = out.fit && out.fit.confident ? FIT_POINTS[out.fit.cls] : null;
  const colourPoints = out.colour ? out.colour.score : null;

  if (fitPoints !== null && fitPoints !== undefined && colourPoints !== null) {
    out.overall = Math.round((fitPoints * 0.6 + colourPoints * 0.4) * 10) / 10;
  } else if (colourPoints !== null) {
    out.overall = Math.round(colourPoints * 10) / 10;
    out.notes.push('Rating is colour only — no fit call.');
  } else if (fitPoints !== null && fitPoints !== undefined) {
    out.overall = Math.round(fitPoints * 10) / 10;
  }

  if (out.overall !== null) Object.assign(out, grade(out.overall));
  return out;
}

/* Ports fit_score/spec.py fit_score() and mismatch_from_score() exactly. The
   returns model was trained against THIS quantity; a different formula here
   would silently feed returns_risk a feature it was not trained on. */
const WEARABLE_TOLERANCE = 0.115;
const SCORE_WEIGHTS = { chest: 0.38, shoulder: 0.32, length: 0.18, sleeve: 0.12 } as const;

/** 0-100. How close this garment is to being cut for this body. */
function fitScore(features: Features): number {
  const d = {
    chest: features.dev_chest, shoulder: features.dev_shoulder,
    length: features.dev_length, sleeve: features.dev_sleeve,
  };
  let sum = 0;
  for (const k of Object.keys(SCORE_WEIGHTS) as (keyof typeof SCORE_WEIGHTS)[]) sum += SCORE_WEIGHTS[k] * d[k] * d[k];
  const penalty = Math.sqrt(sum);
  const score = 100 * (1 - (penalty / WEARABLE_TOLERANCE) * 0.45);
  return Math.max(0, Math.min(100, score));
}

/** The 0-1 `fit_mismatch_score` the returns model consumes. */
function mismatchFromScore(score: number): number {
  return Math.max(0, Math.min(1, (100 - score) / 100));
}

async function load(url = 'fit_model.json'): Promise<FitModel> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} not found (${res.status})`);
  MODEL = (await res.json()) as FitModel;
  return MODEL;
}

export const OutfitScore = {
  load, rate, grade,
  dominantColour, colourFamily, colourHarmony, buildFeatures, classify, estimateBody,
  fitScore, mismatchFromScore,
  SIZES, SIZE_CHART, MIN_CONFIDENCE, WEARABLE_TOLERANCE,
  get model(): FitModel | null { return MODEL; },
};

/* Everything that sizes a garment — the Fit Analyser, the live readout —
   needs the size charts that arrive with
   fit_model.json. They poll for it rather than racing it, but the poll gives
   up after ~6s and calls onFail, so a missing model surfaces as a named
   failure instead of a screen that never finishes loading. */
export function whenOutfitScoreReady(onReady: () => void, onFail?: () => void, waited = 0): void {
  if (OutfitScore.model) { onReady(); return; }
  if (waited >= 6000) { if (onFail) onFail(); return; }
  setTimeout(() => whenOutfitScoreReady(onReady, onFail, waited + 120), 120);
}
