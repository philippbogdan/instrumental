import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1100, height: 1200 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto('http://127.0.0.1:8907/', { waitUntil: 'networkidle' });
await p.waitForTimeout(1200);
console.log(await p.evaluate(() => {
  const cards = [...document.querySelectorAll('.acard')].map((c) => ({
    tag: c.tagName, self: c.getAttribute('href'),
    media: c.querySelector('.acard-medialink')?.getAttribute('href'),
    text: c.querySelector('.acard-textlink')?.getAttribute('href'),
    title: c.querySelector('.acard-title').textContent.slice(0, 22),
  }));
  return JSON.stringify({ cards, companions: document.querySelectorAll('.entry-companion').length }, null, 1);
}));
await p.goto('http://127.0.0.1:8907/instrumental/', { waitUntil: 'load' });
await p.waitForTimeout(800);
const pl = await p.$('.paper-link');
console.log('paper-link:', await p.evaluate((el) => JSON.stringify({
  href: el.href, text: el.textContent,
  rect: [Math.round(el.getBoundingClientRect().right), Math.round(el.getBoundingClientRect().top)],
  color: getComputedStyle(el).color, op: getComputedStyle(el).opacity,
}), pl));
await p.hover('.paper-link');
await p.waitForTimeout(250);
console.log('hover opacity:', await p.evaluate(() => getComputedStyle(document.querySelector('.paper-link')).opacity));
await p.screenshot({ path: '/tmp/instr-paper.png', clip: { x: 700, y: 0, width: 400, height: 130 } });
await b.close();
