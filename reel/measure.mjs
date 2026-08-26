import { chromium } from 'playwright';
const b = await chromium.launch();
for (const [w, dpr] of [[1440,2],[1280,2],[1512,2],[900,1]]) {
  const ctx = await b.newContext({ viewport: { width: w, height: 900 }, deviceScaleFactor: dpr });
  const p = await ctx.newPage();
  await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
  const r = await p.evaluate(() => {
    const card = [...document.querySelectorAll('.acard')].find(c => c.querySelector('video'));
    const v = card.querySelector('video');
    const cb = card.getBoundingClientRect(), vb = v.getBoundingClientRect();
    return { card: [cb.left, cb.right, cb.width], vid: [vb.left, vb.right, vb.width],
             vw: v.videoWidth, vh: v.videoHeight, cs: getComputedStyle(card).borderWidth };
  });
  console.log(w, dpr, JSON.stringify(r));
  await ctx.close();
}
await b.close();
