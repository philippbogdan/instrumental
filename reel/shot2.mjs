import { chromium } from 'playwright';
const b = await chromium.launch();
for (const w of [500, 760, 1000]) {
  const ctx = await b.newContext({ viewport: { width: w, height: 1000 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
  await p.waitForTimeout(2500);
  const card = await p.$('.acard:has(video)');
  await card.screenshot({ path: `/tmp/card-${w}.png` });
  await ctx.close();
}
await b.close();
