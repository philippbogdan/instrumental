import { chromium } from 'playwright';
const b = await chromium.launch();
const out = [];
for (let w = 360; w <= 780; w += 7) {
  const ctx = await b.newContext({ viewport: { width: w, height: 900 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto('https://philippbogdan.com/', { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(900);
  const r = await p.evaluate(() => {
    const card = [...document.querySelectorAll('.acard')].find(c => c.querySelector('video'));
    const v = card.querySelector('video');
    const cb = card.getBoundingClientRect(), vb = v.getBoundingClientRect();
    return { cw: cb.width, vw: vb.width, l: vb.left - cb.left, r: cb.right - vb.right };
  });
  out.push(`${w}\tcard ${r.cw.toFixed(2)}\tvid ${r.vw.toFixed(2)}\tgapL ${r.l.toFixed(2)}\tgapR ${r.r.toFixed(2)}`);
  await ctx.close();
}
console.log(out.join('\n'));
await b.close();
