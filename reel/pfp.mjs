import { chromium } from 'playwright';
const b = await chromium.launch();
for (const w of [1000, 760, 640, 560, 500, 420, 360]) {
  const ctx = await b.newContext({ viewport: { width: w, height: 800 }, deviceScaleFactor: 2 });
  const p = await ctx.newPage();
  await p.goto(process.argv[2] || 'https://philippbogdan.com/', { waitUntil: 'domcontentloaded' });
  await p.waitForTimeout(600);
  const r = await p.evaluate(() => {
    const img = document.querySelector('.portrait');
    const pic = img.closest('picture');
    const ib = img.getBoundingClientRect(), pb = pic ? pic.getBoundingClientRect() : null;
    return { img: [ib.width.toFixed(1), ib.height.toFixed(1)], pic: pb && [pb.width.toFixed(1), pb.height.toFixed(1)] };
  });
  console.log(w, JSON.stringify(r));
  await ctx.close();
}
await b.close();
