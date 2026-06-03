# NexDash UI/UX Polish — Design Spec (2026-06-04)

**Goal:** Whole-app modern visual polish — refine spacing, hierarchy, typography, motion,
and the dark theme — **without changing structure, flow, props, data, or any logic**.
Approach **B** (token system + elevate high-impact components). Theme: **both** a refined
light and a premium dark, with a working toggle (default = system preference).

## Non-negotiable constraint
Styling only. No change to component props/signatures, state logic, data flow, event
handlers, API calls, or JSX structure that alters behavior. `npm run build` must stay
clean and every interaction must keep working. Status colors (SOC red/amber/green, charge
amber, incident severity) stay theme-independent.

## 1. Token foundation (Phase 1)
- Convert `tailwind.config.js` colors to **CSS-variable-backed semantic tokens**:
  `'surface-lowest': 'rgb(var(--c-surface-lowest) / <alpha-value>)'`, etc. Existing class
  names (`bg-surface-lowest`, `text-on-surface-variant`, `border-outline-variant`, …) stay
  identical, so components become theme-aware with near-zero markup change.
- Define variable values twice in `index.css`: `:root` = refined light, `[data-theme="dark"]`
  = premium dark (charcoal `nex-bg` surfaces, emerald `nex-accent`, soft rgba borders).
- Add scale tokens: type, spacing, radii, shadow (elevation), motion (durations/easings).
- Add a small set of reusable component utilities in `@layer components` (e.g. `.nx-card`,
  hover-lift, focus ring, standard transition) so Phase 2 stays consistent.
- `src/lib/theme.js`: init from `localStorage` ?? `prefers-color-scheme`; set
  `document.documentElement.dataset.theme`; toggle + persist. Init runs in `main.jsx`
  before render (no flash).
- `src/components/ThemeToggle.jsx`: sun/moon switch; wired into `TopBar.jsx`.

## 2. Component elevation (Phase 2, parallel — strict file ownership)
- **SHELL**: `App.jsx`, `Sidebar.jsx`, `TopBar.jsx` (keep the toggle), `RoutesView.jsx`,
  `ChatWidget.jsx`, `ChatPanel.jsx`, `ErrorBoundary.jsx`.
- **PLANNER**: `PlannerForm.jsx`, `LocationSearch.jsx`, `TruckCard.jsx` — modern inputs,
  sliders, toggles, focus/hover states.
- **RESULTS**: `RouteResults.jsx`, `SocGauge.jsx`, `TripTimeline.jsx`, `ChargingStopsList.jsx`,
  `ConditionsPanel.jsx`, `ElevationProfile.jsx`, `SpeedProfile.jsx` — refined gauge, cards,
  timeline rail, hierarchy.
- **MAP**: `RouteMap.jsx`, `MapControls.jsx`, `MapLayersPanel.jsx` — pins, controls, layers
  panel, route stroke (divIcon inline styles only; do NOT edit index.css).
- Phase-2 agents must NOT edit `tailwind.config.js` or `index.css` (Phase-1 owned).

## 3. Verification (Phase 3 + main session)
- `npm run build` clean.
- `git diff` audit: confirm only styling/className/token changes — no logic/props/structure
  behavior changes.
- Screenshots in BOTH themes; manual interaction pass (optimize, sliders, map layers/search,
  theme + tile toggles).
