import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1100, height: 800 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
await p.waitForTimeout(800);
console.log(await p.evaluate(() => {
  const img = document.querySelector('.portrait');
  return JSON.stringify({
    currentSrc: img.currentSrc.split('/').pop(),
    natural: [img.naturalWidth, img.naturalHeight],
    rendered: [img.getBoundingClientRect().width, img.getBoundingClientRect().height],
    dpr: devicePixelRatio,
    attrs: { w: img.getAttribute('width'), h: img.getAttribute('height'), sizes: img.getAttribute('sizes') },
    srcset: (img.closest('picture').querySelector('source')?.srcset || img.srcset).slice(0, 300),
  }, null, 1);
}));
const shot = await (await p.$('.portrait')).screenshot({ path: '/tmp/pfp-live.png' });
await b.close();
