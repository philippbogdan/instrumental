import { chromium } from 'playwright';
const b = await chromium.launch();
for (const w of [390, 700, 1100]) {
  const ctx = await b.newContext({ viewport: { width: w, height: 1200 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(1200);
  console.log(w, await p.evaluate(() => JSON.stringify(
    [...document.querySelectorAll('.acard')].map((c) => {
      const m = c.querySelector('.acard-img');
      const body = c.querySelector('.acard-body');
      return {
        t: c.querySelector('.acard-title').textContent.slice(0, 12),
        media: m ? [Math.round(m.getBoundingClientRect().width), Math.round(m.getBoundingClientRect().height)] : null,
        body: Math.round(body.getBoundingClientRect().height),
      };
    }))));
  await ctx.close();
}
await b.close();
