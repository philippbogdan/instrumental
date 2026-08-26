import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1100, height: 1200 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto(process.argv[2] || 'http://127.0.0.1:8907/', { waitUntil: 'networkidle' });
await p.waitForTimeout(1500);
const card = await p.$('.acard:has(.acard-sound)');
await card.scrollIntoViewIfNeeded();
const state = () => p.evaluate(() => {
  const btn = document.querySelector('.acard-sound');
  const v = btn.parentElement.querySelector('video');
  return { muted: v.muted, on: btn.classList.contains('on'),
           label: btn.getAttribute('aria-label'), url: location.pathname,
           opacity: getComputedStyle(btn).opacity };
});
console.log('before:', JSON.stringify(await state()));
await p.screenshot({ path: '/tmp/snd-muted.png', clip: await card.boundingBox() });
await p.click('.acard-sound');
await p.waitForTimeout(300);
console.log('after: ', JSON.stringify(await state()));
await p.screenshot({ path: '/tmp/snd-on.png', clip: await card.boundingBox() });
await p.click('.acard-sound');
await p.waitForTimeout(200);
console.log('again: ', JSON.stringify(await state()));
await b.close();
