/* Does the card's media reach the border at every width?
 *
 * Screenshots the card at 2x and walks inward from each edge: past whatever
 * page background the element clip caught, past the rule, and then asks what
 * the next pixel is. Video is near black; card background there is a gap. */
import { chromium } from 'playwright';
import { PNG } from 'pngjs';

const URL = process.argv[2] || 'http://127.0.0.1:8907/';
const RULE = [225, 230, 235], BG = [252, 253, 254];
const near = (c, t) => Math.abs(c[0]-t[0]) + Math.abs(c[1]-t[1]) + Math.abs(c[2]-t[2]) < 12;

const b = await chromium.launch();
const bad = [];
for (let w = 380; w <= 1000; w += 13) {
  const ctx = await b.newContext({ viewport: { width: w, height: 1200 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto(URL, { waitUntil: 'networkidle' });
  await p.waitForTimeout(1000);
  const png = PNG.sync.read(await (await p.$('.acard:has(video)')).screenshot());
  const px = (x, y) => { const i = (png.width * y + x) << 2; return [png.data[i], png.data[i+1], png.data[i+2]]; };
  // Walk inward along a line, return how many background pixels sit between
  // the rule and the first pixel of the clip.
  const walk = (at, n) => {
    let i = 0;
    while (i < n && near(at(i), BG)) i++;          // outside the card
    while (i < n && near(at(i), RULE)) i++;        // the rule itself
    let gap = 0;
    while (i + gap < n && near(at(i + gap), BG)) gap++;
    return gap;
  };
  const y = Math.round(png.height * 0.2), xm = png.width >> 1;
  const g = {
    left: walk((i) => px(i, y), 20),
    right: walk((i) => px(png.width - 1 - i, y), 20),
    top: walk((i) => px(xm, i), 20),
  };
  const hit = Object.entries(g).filter(([, v]) => v > 0);
  if (hit.length) bad.push(`${w}: ${hit.map(([k, v]) => `${k} ${v}px`).join('  ')}`);
  await ctx.close();
}
await b.close();
console.log(bad.length ? 'GAPS (2x device pixels):\n' + bad.join('\n') : 'flush at every width tested');
