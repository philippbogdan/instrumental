import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 700, height: 1200 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto('http://127.0.0.1:8907/', { waitUntil: 'networkidle' });
await p.waitForTimeout(1200);
const cards = await p.$$('.acard');
const opAt = (el) => p.evaluate((e) => getComputedStyle(e.closest('.acard')).opacity, el);
// press on essay body, essay hero, split-card hero, split-card body
const spots = [
  ['essay body', '.acard[href*="essays"] .acard-body'],
  ['essay hero', '.acard[href*="essays"] .acard-img'],
  ['split hero', '.acard-medialink[href="/instrumental"]'],
  ['split body', '.acard-textlink[href*="2603"]'],
];
for (const [name, sel] of spots) {
  const el = await p.$(sel);
  await el.scrollIntoViewIfNeeded();
  const box = await el.boundingBox();
  await p.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await p.mouse.down();
  await p.waitForTimeout(180);
  console.log(name, 'pressed opacity:', await opAt(el));
  await p.mouse.move(5, 5);   // release off the link so no navigation fires
  await p.mouse.up();
  await p.waitForTimeout(400);
  if (!p.url().endsWith(':8907/')) { await p.goto('http://127.0.0.1:8907/', { waitUntil: 'domcontentloaded' }); }
  await p.waitForTimeout(700);
}
// media sizes incl. new reel
console.log(await p.evaluate(() => JSON.stringify([...document.querySelectorAll('.acard-img')].map((m) =>
  [m.tagName, Math.round(m.getBoundingClientRect().width), Math.round(m.getBoundingClientRect().height), m.videoWidth || m.naturalWidth]))));
console.log('bodies:', await p.evaluate(() => JSON.stringify([...document.querySelectorAll('.acard-body')].map((b) => Math.round(b.getBoundingClientRect().height)))));
await b.close();
