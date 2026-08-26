import { chromium } from 'playwright';
const b = await chromium.launch({ headless: false, args: ['--hide-scrollbars'] });
const ctx = await b.newContext({ viewport: { width: 1100, height: 440 }, deviceScaleFactor: 1 });
const p = await ctx.newPage();
await p.addInitScript(() => {
  const put = () => {
    const s = document.createElement('style');
    s.textContent = 'header{padding:16px 0 4px!important} .logo{font-size:1.3rem!important}';
    (document.head || document.documentElement).appendChild(s);
  };
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', put, { once: true }) : put();
});
await p.goto('http://127.0.0.1:8801/', { waitUntil: 'load' });
await p.waitForTimeout(700);
await p.screenshot({ path: '/tmp/fr-landing.png' });
await p.setInputFiles('#fileInput', '/Users/phil/instrumental/app/demos/lie_original.wav');
await p.waitForTimeout(400);
await p.click('#matchBtn');
await p.waitForTimeout(3000);
await p.screenshot({ path: '/tmp/fr-sep.png' });
await b.close();
