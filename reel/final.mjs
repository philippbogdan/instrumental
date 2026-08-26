import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1100, height: 1400 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
await p.waitForTimeout(2500);
const cards = await p.$$('.acard');
console.log('cards:', cards.length);
const vids = await p.evaluate(() => [...document.querySelectorAll('.acard video')].map(v => ({
  src: v.getAttribute('src'), playing: !v.paused, w: v.videoWidth, h: v.videoHeight })));
console.log(JSON.stringify(vids));
const pfp = await p.evaluate(() => { const r = document.querySelector('.portrait').getBoundingClientRect(); return [r.width, r.height]; });
console.log('portrait', pfp);
await p.screenshot({ path: '/tmp/final-home.png', fullPage: true });
await b.close();
