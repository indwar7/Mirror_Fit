// Renders public/icon.svg to the PNG sizes the manifest and iOS need. Run once
// when the mark changes:  node scripts/icons.mjs
import { readFileSync, mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from 'playwright';

const svg = readFileSync(new URL('../public/icon.svg', import.meta.url), 'utf8');
mkdirSync(new URL('../public/icons/', import.meta.url), { recursive: true });

// Maskable icons must keep the mark inside the central 80% safe zone, so the
// mark is drawn smaller on a full-bleed background for that variant.
const variants = [
  { file: 'icon-192.png',          size: 192, pad: 0 },
  { file: 'icon-512.png',          size: 512, pad: 0 },
  { file: 'apple-touch-icon.png',  size: 180, pad: 0 },
  { file: 'icon-512-maskable.png', size: 512, pad: 0.12 },
];

const browser = await chromium.launch();
const page = await browser.newPage();
for (const v of variants) {
  const inner = Math.round(v.size * (1 - 2 * v.pad));
  const html = `<html><body style="margin:0;background:#14110F">` +
    `<div style="width:${v.size}px;height:${v.size}px;display:flex;align-items:center;justify-content:center">` +
    svg.replace('width="64" height="64"', `width="${inner}" height="${inner}"`) +
    `</div></body></html>`;
  await page.setViewportSize({ width: v.size, height: v.size });
  await page.setContent(html);
  const png = await page.screenshot({ clip: { x: 0, y: 0, width: v.size, height: v.size }, omitBackground: false });
  writeFileSync(new URL('../public/icons/' + v.file, import.meta.url), png);
  console.log('wrote', v.file, png.length, 'bytes');
}
await browser.close();
