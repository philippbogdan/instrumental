import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
const p = await ctx.newPage();
await p.goto('https://philippbogdan.com/instrumental/', { waitUntil: 'load' });
await p.waitForTimeout(1500);
await p.screenshot({ path: '/tmp/inst-mob.png', fullPage: true });
console.log(await p.evaluate(() => {
  const vp = document.querySelector('meta[name=viewport]');
  const logo = document.querySelector('.logo').getBoundingClientRect();
  const paper = document.querySelector('.paper-link').getBoundingClientRect();
  return JSON.stringify({ viewport: vp && vp.content, docW: document.documentElement.scrollWidth, innerW: innerWidth,
    logo: [Math.round(logo.left), Math.round(logo.right), Math.round(logo.top), Math.round(logo.bottom)],
    paper: [Math.round(paper.left), Math.round(paper.right), Math.round(paper.top), Math.round(paper.bottom)] });
}));
await b.close();
