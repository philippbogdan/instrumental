import { chromium } from 'playwright';
const b = await chromium.launch();
for (const dpr of [2, 3]) {
  const ctx = await b.newContext({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: dpr });
  const p = await ctx.newPage();
  await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(600);
  console.log(dpr + 'x', await p.evaluate(() => {
    const img = document.querySelector('.portrait');
    const r = img.getBoundingClientRect();
    return JSON.stringify({ src: img.currentSrc.split('/').pop(), rendered: [r.width, r.height] });
  }));
  if (dpr === 2) await (await p.$('.portrait')).screenshot({ path: '/tmp/pfp-new.png' });
  await ctx.close();
}
await b.close();
