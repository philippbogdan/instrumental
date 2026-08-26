import { chromium } from 'playwright';
const b = await chromium.launch();
const ctx = await b.newContext({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 2 });
const p = await ctx.newPage();
await p.goto('https://philippbogdan.com/', { waitUntil: 'networkidle' });
await p.waitForTimeout(2500);
const card = await p.$('.acard:has(video)');
await card.screenshot({ path: '/tmp/card2x.png' });
await b.close();
