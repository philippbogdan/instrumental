import { chromium } from 'playwright';
const b = await chromium.launch({ headless: false });
const ctx = await b.newContext({
  viewport: { width: 390, height: 780 }, deviceScaleFactor: 2,
  isMobile: true, hasTouch: true,
});
const p = await ctx.newPage();
await p.goto(process.argv[2] || 'http://127.0.0.1:8907/atoms/', { waitUntil: 'load' });
await p.waitForTimeout(3500);
// jump deep into the run so plenty of atoms exist, then select one via a
// synthetic keyboard shortcut? No shortcut selects deterministically, so
// click at points until the panel opens.
await p.evaluate(() => { const s = document.getElementById('scrub'); s.value = 700; s.dispatchEvent(new Event('input', { bubbles: true })); });
await p.waitForTimeout(2500);
const cloud = await p.$('#cloud');
const box = await cloud.boundingBox();
let open = false;
outer:
for (let dy = -60; dy <= 60; dy += 20) {
  for (let dx = -80; dx <= 80; dx += 16) {
    await p.mouse.click(box.x + box.width / 2 + dx, box.y + box.height / 2 + dy);
    await p.waitForTimeout(250);
    if (await p.evaluate(() => !document.getElementById('apanel').hidden)) { open = true; break outer; }
  }
}
console.log('panel open:', open);
if (open) {
  const st = () => p.evaluate(() => {
    const ap = document.getElementById('apanel');
    return {
      collapsed: ap.classList.contains('collapsed'),
      peekShown: !!document.querySelector('.apanel.collapsed .appeek') && getComputedStyle(document.getElementById('ap-peek')).display,
      peekText: document.getElementById('ap-peek-text').textContent.slice(0, 40),
      tickOpacity: getComputedStyle(document.querySelector('.tick')).opacity,
      toggleShown: getComputedStyle(document.getElementById('ap-toggle')).display,
    };
  });
  console.log('collapsed state:', JSON.stringify(await st()));
  await p.screenshot({ path: '/tmp/atom-collapsed.png' });
  await p.tap('#ap-peek');
  await p.waitForTimeout(400);
  console.log('after tap:     ', JSON.stringify(await st()));
  await p.screenshot({ path: '/tmp/atom-open.png' });
  await p.tap('#ap-toggle');
  await p.waitForTimeout(300);
  console.log('after fold:    ', JSON.stringify(await st()));
  await p.tap('#ap-close');
  await p.waitForTimeout(400);
  console.log('after close: hidden=', await p.evaluate(() => document.getElementById('apanel').hidden),
    'tick', await p.evaluate(() => getComputedStyle(document.querySelector('.tick')).opacity));
}
await b.close();
