import { webkit, devices } from 'playwright';
const b = await webkit.launch();
const ctx = await b.newContext({ ...devices['iPhone 13'] });
const p = await ctx.newPage();
await p.goto('http://127.0.0.1:8907/instrumental/', { waitUntil: 'load' });
await p.waitForTimeout(1000);
console.log(await p.evaluate(() => {
  const l = document.querySelector('.logo').getBoundingClientRect();
  const pa = document.querySelector('.paper-link').getBoundingClientRect();
  const h = document.querySelector('.home-link').getBoundingClientRect();
  return JSON.stringify({ logoTop: Math.round(l.top), paperBottom: Math.round(pa.bottom),
    overlap: l.top < pa.bottom && l.right > pa.left, home: [Math.round(h.left), Math.round(h.top)], homeHref: document.querySelector('.home-link').href });
}));
await p.tap('.demo-card[data-demo="gunnr"]');
await p.waitForTimeout(3500);
await p.screenshot({ path: '/tmp/wk-demo2.png', fullPage: true });
await p.goto('http://127.0.0.1:8907/atoms/', { waitUntil: 'load' });
await p.waitForTimeout(2500);
console.log('atoms back:', await p.evaluate(() => {
  const a = document.querySelector('.headbtn.back');
  const r = a.getBoundingClientRect();
  return JSON.stringify({ href: a.href, rect: [Math.round(r.left), Math.round(r.top)], text: a.textContent });
}));
await p.screenshot({ path: '/tmp/atoms-head.png', clip: { x: 0, y: 0, width: 390, height: 60 } });
await b.close();
