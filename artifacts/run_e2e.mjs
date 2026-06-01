// E2E script: Berlin -> Munich route plan on NexDash
// Runs headless Chromium from the ms-playwright cache.
// Usage: node artifacts/run_e2e.mjs

import { chromium } from 'playwright';
import { writeFileSync } from 'fs';
import { fileURLToPath } from 'url';
import path from 'path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_PATH = path.join(__dirname, 'ours-berlin-munich.png');
const RESULTS_PATH    = path.join(__dirname, 'e2e-results.json');

const CHROMIUM_EXEC =
  process.env.CHROMIUM_EXECUTABLE ||
  '/Users/alwinpaul/Library/Caches/ms-playwright/chromium-1181/chrome-mac/Chromium.app/Contents/MacOS/Chromium';

// ── helpers ──────────────────────────────────────────────────────────────────

/** Set a <input type="range"> to a target value by dispatching React-friendly events. */
async function setSlider(page, ariaLabel, targetValue) {
  const slider = page.locator(`input[type="range"][aria-label="${ariaLabel}"]`);
  await slider.waitFor({ state: 'visible', timeout: 10000 });

  // Set via evaluate first (fastest), then fire events so React state updates.
  await slider.evaluate((el, val) => {
    const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
      window.HTMLInputElement.prototype, 'value'
    ).set;
    nativeInputValueSetter.call(el, val);
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, String(targetValue));

  // Read back actual value
  const actual = await slider.evaluate(el => el.value);
  return Number(actual);
}

/** Type into a LocationSearch input and click the first dropdown suggestion. */
async function fillLocation(page, placeholder, query) {
  const input = page.locator(`input[placeholder="${placeholder}"]`);
  await input.waitFor({ state: 'visible', timeout: 10000 });
  await input.click();
  await input.fill('');
  await input.type(query, { delay: 80 });

  // Wait for autocomplete list
  const listItem = page.locator('ul li button').first();
  await listItem.waitFor({ state: 'visible', timeout: 15000 });
  const labelText = await listItem.textContent();
  await listItem.click();
  // Wait for dropdown to close
  await page.waitForFunction(() => document.querySelector('ul li button') === null, { timeout: 5000 }).catch(() => {});
  return labelText?.trim();
}

// ── main ─────────────────────────────────────────────────────────────────────

const networkLog = [];

const browser = await chromium.launch({
  executablePath: CHROMIUM_EXEC,
  headless: true,
  args: ['--no-sandbox', '--disable-dev-shm-usage'],
});

const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  recordVideo: undefined,
});

// Capture network requests
context.on('request', req => {
  const url = req.url();
  if (
    url.includes('/api/optimize') ||
    url.includes('/api/route-plan') ||
    url.includes('api.tomtom.com') ||
    url.includes('open-meteo') ||
    url.includes('nominatim') ||
    url.includes('geocod')
  ) {
    networkLog.push({ method: req.method(), url });
  }
});

const page = await context.newPage();

console.log('Navigating to http://localhost:4173/ ...');
await page.goto('http://localhost:4173/', { waitUntil: 'networkidle', timeout: 30000 });
await page.waitForTimeout(1500);

// ── 1. Starting Battery = 90% ─────────────────────────────────────────────
console.log('Setting Starting Battery to 90%...');
const actualStartSoc = await setSlider(page, 'Starting battery', 90);
console.log(`  → Starting battery actual: ${actualStartSoc}%`);

// ── 2. Origin = Berlin ────────────────────────────────────────────────────
console.log('Setting Origin to Berlin...');
const originLabel = await fillLocation(page, 'Where does the trip start?', 'Berlin');
console.log(`  → Origin selected: ${originLabel}`);
await page.waitForTimeout(800);

// ── 3. Add Stop → Munich ─────────────────────────────────────────────────
console.log('Adding destination: Munich...');
const addStopBtn = page.locator('button', { hasText: 'Add Stop' });
await addStopBtn.waitFor({ state: 'visible', timeout: 5000 });
await addStopBtn.click();
await page.waitForTimeout(500);

const destLabel = await fillLocation(page, 'Add a destination…', 'Munich');
console.log(`  → Destination selected: ${destLabel}`);
await page.waitForTimeout(800);

// ── 4. Open "More Options" ────────────────────────────────────────────────
console.log('Opening More Options...');
const moreBtn = page.locator('button[aria-expanded]', { hasText: 'More Options' });
await moreBtn.waitFor({ state: 'visible', timeout: 5000 });
const isExpanded = await moreBtn.getAttribute('aria-expanded');
if (isExpanded === 'false') {
  await moreBtn.click();
  await page.waitForTimeout(500);
}

// ── 5. Arrive with at least = 15% (min=0, max=50) ────────────────────────
console.log('Setting Arrive with at least to 15%...');
const actualMinSoc = await setSlider(page, 'Minimum SOC floor', 15);
console.log(`  → Arrive with at least actual: ${actualMinSoc}%`);

// ── 6. Safety Reserve = 20% (min=0, max=25) ──────────────────────────────
console.log('Setting Safety Reserve to 20%...');
const actualReserve = await setSlider(page, 'Safety reserve', 20);
console.log(`  → Safety reserve actual: ${actualReserve}%`);

// ── 7. Max Charging Detour = 50 km (min=5, max=100, step=5) ──────────────
console.log('Setting Max Charging Detour to 50 km...');
const actualDetour = await setSlider(page, 'Maximum charging detour', 50);
console.log(`  → Max Charging Detour actual: ${actualDetour} km`);

// ── 8. Max Charging Speed = 400 kW (min=100, max=400, step=10) ───────────
console.log('Setting Max Charging Speed to 400 kW...');
const actualChargeKw = await setSlider(page, 'Max Charging Speed', 400);
// Also try without label match (no aria-label set for this slider per source)
console.log(`  → Max Charging Speed actual: ${actualChargeKw} kW`);

// ── 9. Payload = 18000 kg ────────────────────────────────────────────────
console.log('Setting Payload to 18000 kg...');
// Payload slider: aria-label not set in source (Slider component without ariaLabel prop)
// Try by position: it's the last slider inside the More Options section
const payloadActual = await page.evaluate(() => {
  // Find all range inputs
  const sliders = Array.from(document.querySelectorAll('input[type="range"]'));
  // The payload slider is the last one rendered (after maxChargeKw slider)
  const s = sliders[sliders.length - 1];
  if (!s) return null;
  const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
    window.HTMLInputElement.prototype, 'value'
  ).set;
  nativeInputValueSetter.call(s, '18000');
  s.dispatchEvent(new Event('input', { bubbles: true }));
  s.dispatchEvent(new Event('change', { bubbles: true }));
  return s.value;
});
console.log(`  → Payload actual: ${payloadActual} kg`);

await page.waitForTimeout(500);

// ── 10. Click "Optimize Route" ────────────────────────────────────────────
console.log('Clicking Optimize Route...');
const optimizeBtn = page.locator('button', { hasText: 'Optimize Route' });
await optimizeBtn.waitFor({ state: 'visible', timeout: 5000 });
await optimizeBtn.click();
console.log('Waiting for results (up to 90s)...');

// Wait for the results panel to appear — look for "Route Info" heading or SOC gauge
await page.waitForFunction(() => {
  const headings = Array.from(document.querySelectorAll('h3'));
  return headings.some(h =>
    h.textContent.includes('Route Info') ||
    h.textContent.includes('Energy Overview') ||
    h.textContent.includes('Charging Stops') ||
    h.textContent.includes('Driver Hours')
  );
}, { timeout: 90000 });

console.log('Results appeared. Waiting extra 3s for full render...');
await page.waitForTimeout(3000);

// ── 11. Take full-page screenshot ─────────────────────────────────────────
console.log('Taking screenshot...');
await page.screenshot({ path: SCREENSHOT_PATH, fullPage: true });
console.log(`Screenshot saved: ${SCREENSHOT_PATH}`);

// ── 12. Extract result values ─────────────────────────────────────────────
console.log('Extracting result values...');

const results = await page.evaluate(() => {
  function txt(sel) {
    const el = document.querySelector(sel);
    return el ? el.textContent.trim() : null;
  }
  function allTxt(sel) {
    return Array.from(document.querySelectorAll(sel)).map(el => el.textContent.trim());
  }

  // ── SOC Gauge: look for the SVG text elements or aria labels
  const gaugeTexts = allTxt('svg text');

  // ── Route Info cards: find all InfoCard-style divs
  // Each card has: icon span, value p (text-xl), label p (text-[11px])
  const cards = Array.from(document.querySelectorAll('.rounded-xl.bg-surface-low.border')).map(card => {
    const valuePara = card.querySelector('p.text-xl');
    const labelPara = card.querySelector('p.text-\\[11px\\]');
    const icon = card.querySelector('.material-symbols-outlined');
    return {
      icon: icon?.textContent?.trim(),
      value: valuePara?.textContent?.trim(),
      label: labelPara?.textContent?.trim(),
    };
  }).filter(c => c.value && c.label);

  // ── Section headings with their content
  const sections = Array.from(document.querySelectorAll('h3')).map(h => ({
    heading: h.textContent.trim(),
    parentText: h.parentElement?.textContent?.trim()?.slice(0, 500),
  }));

  // ── Charging stops list
  const chargingStopsSection = document.querySelector('h3');
  const chargingItems = Array.from(document.querySelectorAll('ul li')).map(li => {
    const name = li.querySelector('p.text-sm.font-semibold');
    const details = li.querySelectorAll('div span');
    const badges = li.querySelectorAll('span.rounded-full');
    return {
      name: name?.textContent?.trim(),
      details: Array.from(details).map(s => s.textContent.trim()).filter(Boolean),
      badges: Array.from(badges).map(s => s.textContent.trim()).filter(Boolean),
    };
  }).filter(li => li.name);

  // ── Driver hours section
  const driverSection = Array.from(document.querySelectorAll('div')).find(d =>
    d.querySelector('h3')?.textContent?.includes('Driver Hours')
  );
  const driverText = driverSection?.textContent?.trim()?.slice(0, 400);

  // ── Energy overview section
  const energySection = Array.from(document.querySelectorAll('div')).find(d =>
    d.querySelector('h3')?.textContent?.includes('Energy Overview')
  );
  const energyText = energySection?.textContent?.trim()?.slice(0, 400);

  // ── Conditions panel
  const condSection = Array.from(document.querySelectorAll('div')).find(d =>
    d.querySelector('h3')?.textContent?.includes('Route Conditions')
  );
  const condText = condSection?.textContent?.trim()?.slice(0, 400);

  // ── Full page text for fallback parsing
  const bodyText = document.body.innerText.slice(0, 8000);

  return {
    gaugeTexts,
    cards,
    sections: sections.map(s => ({ heading: s.heading, parentText: s.parentText?.slice(0, 300) })),
    chargingItems,
    driverText,
    energyText,
    condText,
    bodyText,
  };
});

// Write raw JSON for inspection
writeFileSync(RESULTS_PATH, JSON.stringify({ results, networkLog }, null, 2));
console.log(`Results JSON saved: ${RESULTS_PATH}`);

// ── Print structured report ───────────────────────────────────────────────
console.log('\n========== EXTRACTED RESULTS ==========\n');
console.log('GAUGE TEXTS:', JSON.stringify(results.gaugeTexts));
console.log('\nCARDS:');
results.cards.forEach((c, i) => console.log(`  [${i}] icon=${c.icon} value=${c.value} label=${c.label}`));
console.log('\nSECTIONS:');
results.sections.forEach(s => {
  console.log(`\n  --- ${s.heading} ---`);
  console.log(`  ${s.parentText}`);
});
console.log('\nCHARGING ITEMS:');
results.chargingItems.forEach((ci, i) => {
  console.log(`  [${i+1}] ${ci.name}`);
  console.log(`       details: ${JSON.stringify(ci.details)}`);
  console.log(`       badges:  ${JSON.stringify(ci.badges)}`);
});
console.log('\nDRIVER TEXT:', results.driverText);
console.log('\nENERGY TEXT:', results.energyText);
console.log('\nCONDITIONS TEXT:', results.condText);
console.log('\nNETWORK LOG:');
networkLog.forEach(n => console.log(`  ${n.method} ${n.url}`));
console.log('\nBODY TEXT (first 3000 chars):\n', results.bodyText.slice(0, 3000));

await browser.close();
console.log('\nDone.');
