import { chromium } from 'playwright';
import fs from 'node:fs';
const OUT = '/tmp/irun';
fs.rmSync(OUT, { recursive: true, force: true }); fs.mkdirSync(OUT, { recursive: true });
const b = await chromium.launch({ headless: false, args: ['--hide-scrollbars', '--autoplay-policy=no-user-gesture-required'] });
const ctx = await b.newContext({ viewport: { width: 1400, height: 560 }, deviceScaleFactor: 1 });
const p = await ctx.newPage();
p.on('console', (m) => { const t = m.text(); if (t.includes('INSTRUMENTAL')) console.log('page:', t.slice(0, 120)); });
await p.goto('http://127.0.0.1:8801/', { waitUntil: 'load' });
await p.waitForTimeout(800);
await p.setInputFiles('#fileInput', '/Users/phil/instrumental/app/demos/lie_original.wav');
await p.waitForTimeout(500);
await p.click('#matchBtn');
console.log('match clicked');
const t0 = Date.now();
let i = 0, startClicked = false;
const marks = [];
while (Date.now() - t0 < 300000) {
  await p.screenshot({ path: `${OUT}/f${String(i).padStart(5, '0')}.jpg`, type: 'jpeg', quality: 85 });
  const st = await p.evaluate(() => ({
    stem: !document.getElementById('stemSection').classList.contains('hidden'),
    prog: !document.getElementById('progressSection').classList.contains('hidden'),
    res: !document.getElementById('resultSection').classList.contains('hidden'),
    phase: (document.getElementById('progressPhase') || {}).textContent,
    evals: (document.getElementById('evalCount') || {}).textContent,
    gate: (() => {
      const b = [...document.querySelectorAll('#stemContainer .match-btn')].find((x) => x.offsetParent);
      return b ? b.textContent : null;
    })(),
  }));
  if (i % 20 === 0) console.log(i, ((Date.now() - t0) / 1000).toFixed(0) + 's', JSON.stringify(st));
  marks.push({ i, t: Date.now() - t0, ...st });
  if (st.gate) {
    await p.evaluate(() => [...document.querySelectorAll('#stemContainer .match-btn')].find((x) => x.offsetParent).click());
    console.log('gate clicked:', st.gate, 'frame', i);
    await p.waitForTimeout(400);
  }
  if (st.res) { console.log('result at frame', i); break; }
  i++;
  await p.waitForTimeout(120);
}
fs.writeFileSync('/tmp/irun-marks.json', JSON.stringify(marks));
await p.waitForTimeout(4000);
for (let k = 0; k < 40; k++) { await p.screenshot({ path: `${OUT}/f${String(i + k).padStart(5, '0')}.jpg`, type: 'jpeg', quality: 85 }); await p.waitForTimeout(120); }
console.log('frames', i + 40);
await b.close();
