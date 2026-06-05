import { Fragment, useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  Polyline,
  Marker,
  Tooltip,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { geocode } from "../../lib/routePlanner.js";
import MapLayersPanel from "./MapLayersPanel.jsx";
import MapControls from "./MapControls.jsx";

// --------------------------------------------------------------------------- //
// Tile styles — TomTom raster tiles, CartoDB fallbacks when no key.
// --------------------------------------------------------------------------- //
const TOMTOM_KEY = import.meta.env.VITE_TOMTOM_API_KEY;
const MAPTILER_KEY = import.meta.env.VITE_MAPTILER_API_KEY;

function tomtomTile(layer, style, ext) {
  return `https://{s}.api.tomtom.com/map/1/tile/${layer}/${style}/{z}/{x}/{y}.${ext}?key=${TOMTOM_KEY}&tileSize=256`;
}

function maptilerTile(style, ext) {
  return `https://api.maptiler.com/maps/${style}/{z}/{x}/{y}.${ext}?key=${MAPTILER_KEY}`;
}

// Preferred provider: MapTiler (matches the reference look) when a key is set.
const MAPTILER_ATTR =
  '&copy; <a href="https://www.maptiler.com/">MapTiler</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>';
const MAPTILER_STYLES = {
  streets: { label: "Streets", icon: "map", url: maptilerTile("streets-v2", "png"), attribution: MAPTILER_ATTR },
  // MapTiler "hybrid" already includes place-name + road labels over imagery.
  satellite: { label: "Satellite", icon: "satellite", url: maptilerTile("hybrid", "jpg"), attribution: MAPTILER_ATTR },
  dark: { label: "Dark", icon: "dark_mode", url: maptilerTile("streets-v2-dark", "png"), attribution: MAPTILER_ATTR },
};

const TILE_STYLES = MAPTILER_KEY
  ? MAPTILER_STYLES
  : TOMTOM_KEY
  ? {
      streets: {
        label: "Streets",
        icon: "map",
        url: tomtomTile("basic", "main", "png"),
        attribution: '&copy; <a href="https://www.tomtom.com">TomTom</a>',
        subdomains: ["a", "b", "c", "d"],
      },
      satellite: {
        label: "Satellite",
        icon: "satellite",
        url: tomtomTile("sat", "main", "jpg"),
        // Transparent roads + place-name overlay so satellite shows names
        // (Google/Apple-style hybrid).
        overlay: tomtomTile("hybrid", "main", "png"),
        attribution: '&copy; <a href="https://www.tomtom.com">TomTom</a>',
        subdomains: ["a", "b", "c", "d"],
      },
      dark: {
        label: "Dark",
        icon: "dark_mode",
        url: tomtomTile("basic", "night", "png"),
        attribution: '&copy; <a href="https://www.tomtom.com">TomTom</a>',
        subdomains: ["a", "b", "c", "d"],
      },
    }
  : {
      streets: {
        label: "Streets",
        icon: "map",
        url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: ["a", "b", "c", "d"],
      },
      satellite: {
        label: "Satellite",
        icon: "satellite",
        url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        overlay: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png",
        attribution: "&copy; Esri",
        subdomains: ["a", "b", "c", "d"],
      },
      dark: {
        label: "Dark",
        icon: "dark_mode",
        url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: ["a", "b", "c", "d"],
      },
    };

const DEFAULT_LAYERS = { route: true, charging: true, stops: true, drain: true, incidents: true };

// Friendly charge-duration label, e.g. 71 -> "71 min", 132 -> "2 h 12 min".
function fmtChargeTime(min) {
  if (!Number.isFinite(min) || min <= 0) return "";
  if (min < 90) return `${Math.round(min)} min`;
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

// Battery-level → colour, in 5 distinct bands so the route line reads as a
// high→low SOC gradient (deep-green full, through yellow/amber, to red near empty).
function socColor(soc) {
  if (soc >= 80) return "#15803d"; // 80-100% — deep green (comfortable)
  if (soc >= 60) return "#22c55e"; // 60-80%  — green
  if (soc >= 40) return "#eab308"; // 40-60%  — yellow
  if (soc >= 20) return "#f59e0b"; // 20-40%  — amber (getting low)
  return "#ef4444";                // <20%    — red (near reserve)
}

function pinIcon(kind, num) {
  const color = kind === "origin" ? "#1ca64c" : "#0059bb";
  // Teardrop location pin: colored drop, white ring. Origin shows a center dot,
  // destinations a small flag glyph inside the ring.
  // Destinations show their sequence number inside the ring (so Destination 1 vs
  // 2 are distinguishable on the map, matching the numbered list); origin shows a
  // center dot. A dest with no number falls back to the flag glyph.
  const center =
    kind === "origin"
      ? `<circle cx="12" cy="12" r="3" fill="${color}"/>`
      : Number.isFinite(num)
      ? `<text x="12" y="12" text-anchor="middle" dominant-baseline="central" font-family="Inter,sans-serif" font-size="9" font-weight="700" fill="${color}">${num}</text>`
      : "";
  const flag =
    kind === "origin" || Number.isFinite(num)
      ? ""
      : `<span class="material-symbols-outlined" style="position:absolute;left:19px;top:18px;transform:translate(-50%,-50%);font-size:14px;color:${color};">flag</span>`;
  const gradId = `pin-${kind}-${Number.isFinite(num) ? num : "x"}`;
  return L.divIcon({
    className: "",
    html: `<div style="position:relative;width:38px;height:48px;filter:drop-shadow(0 4px 6px rgba(0,0,0,0.28)) drop-shadow(0 1px 1px rgba(0,0,0,0.18));">
      <svg width="38" height="48" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="${gradId}" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${color}"/>
            <stop offset="100%" stop-color="${color}" stop-opacity="0.82"/>
          </linearGradient>
        </defs>
        <path d="M12 0C5.37 0 0 5.37 0 12c0 8.5 10.5 18.7 11.1 19.3a1.3 1.3 0 0 0 1.8 0C13.5 30.7 24 20.5 24 12 24 5.37 18.63 0 12 0z" fill="url(#${gradId})" stroke="#fff" stroke-width="1.75"/>
        <circle cx="12" cy="12" r="6.5" fill="#fff"/>
        ${center}
      </svg>
      ${flag}
    </div>`,
    iconSize: [38, 48],
    iconAnchor: [19, 48],
  });
}

function gpsIcon() {
  return L.divIcon({
    className: "",
    html: `<div style="position:relative;width:18px;height:18px;transform:translate(-50%,-50%);">
      <div class="gps-pulse"></div>
      <div class="gps-dot"></div>
    </div>`,
    iconSize: [18, 18],
    iconAnchor: [0, 0],
  });
}

function chargeIcon(num) {
  return L.divIcon({
    className: "",
    html: `<div style="transform:translate(-50%,-50%);background:linear-gradient(160deg,#fbbf24,#f59e0b);color:#fff;border-radius:9999px;width:25px;height:25px;display:flex;align-items:center;justify-content:center;font:700 12px Inter,sans-serif;border:2.5px solid #fff;box-shadow:0 3px 7px rgba(245,158,11,0.45),0 1px 2px rgba(0,0,0,0.25);">${num}</div>`,
    iconSize: [25, 25],
    iconAnchor: [0, 0],
  });
}

// Traffic-incident glyph + color (TomTom iconCategory / magnitudeOfDelay).
const INCIDENT_GLYPH = {
  1: "warning", // accident
  6: "traffic", // jam
  7: "block", // lane closed
  8: "block", // road closed
  9: "construction", // road works
  14: "car_repair", // broken-down vehicle
};
function incidentColor(mag) {
  return mag >= 3 ? "#ba1a1a" : mag === 2 ? "#f97316" : "#f59e0b";
}
function incidentGlyph(cat) {
  return INCIDENT_GLYPH[cat] || "report";
}
function incidentIcon(inc) {
  const color = incidentColor(inc.magnitude);
  return L.divIcon({
    className: "",
    html: `<div style="transform:translate(-50%,-50%);background:${color};color:#fff;border-radius:9999px;width:27px;height:27px;display:flex;align-items:center;justify-content:center;border:2.5px solid #fff;box-shadow:0 3px 7px ${color}55,0 1px 2px rgba(0,0,0,0.28);">
      <span class="material-symbols-outlined" style="font-size:16px;">${incidentGlyph(inc.category)}</span>
    </div>`,
    iconSize: [26, 26],
    iconAnchor: [0, 0],
  });
}

// Map the SOC profile onto the geometry and emit colored sub-segments.
function buildSocSegments(geometry, socProfile, totalKm, chargingStops) {
  if (!geometry || geometry.length < 2) return [];
  if (!socProfile || socProfile.length < 2 || !totalKm) {
    return [{ positions: geometry, color: "#00d166" }];
  }
  const cum = [0];
  const toRad = (d) => (d * Math.PI) / 180;
  for (let i = 1; i < geometry.length; i++) {
    const a = geometry[i - 1];
    const b = geometry[i];
    const R = 6371;
    const dLat = toRad(b[0] - a[0]);
    const dLng = toRad(b[1] - a[1]);
    const h =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
    cum.push(cum[i - 1] + 2 * R * Math.asin(Math.sqrt(h)));
  }
  // Charge KNOTS drive the colouring. The displayed route detours into each
  // charging station (extra geometry the direct-route socProfile doesn't have),
  // so mapping by distance smears the post-charge colour and paints a long red
  // stretch AFTER the stop. Instead we pin each station to its nearest geometry
  // vertex and treat SOC as a drain BETWEEN knots: it falls from the previous
  // knot's depart-SOC to the next knot's arrive-SOC, then jumps UP at the
  // station vertex (arrive → depart). The jump is a hard break exactly at the
  // marker, so everything before reads as the low arrival colour (red) and
  // everything after reads as the topped-up colour (green) — no leak.
  const hav = (a, b) => {
    const R = 6371;
    const dLat = toRad(b[0] - a[0]);
    const dLng = toRad(b[1] - a[1]);
    const h =
      Math.sin(dLat / 2) ** 2 +
      Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
    return 2 * R * Math.asin(Math.sqrt(h));
  };
  const lastIdx = geometry.length - 1;
  const startSoc = socProfile[0].soc;
  const endSoc = socProfile[socProfile.length - 1].soc;

  // Each charge stop → its nearest geometry vertex, carrying arrive/depart SOC.
  const charges = [];
  if (Array.isArray(chargingStops)) {
    for (const st of chargingStops) {
      if (!Number.isFinite(st?.lat) || !Number.isFinite(st?.lng)) continue;
      if (!Number.isFinite(st?.arriveSoc) || !Number.isFinite(st?.departSoc)) continue;
      let bi = 0;
      let bd = Infinity;
      for (let i = 0; i < geometry.length; i++) {
        const d = hav(geometry[i], [st.lat, st.lng]);
        if (d < bd) { bd = d; bi = i; }
      }
      charges.push({ vi: bi, arrive: st.arriveSoc, depart: st.departSoc });
    }
  }
  charges.sort((a, b) => a.vi - b.vi);

  // Knots: vertex 0 (leave = startSoc) → each charge (enter = arrive, leave =
  // depart) → final vertex (enter = endSoc). Drop charges that don't advance.
  const knots = [{ vi: 0, leave: startSoc }];
  const chargeAt = new Map(); // vertexIndex -> { arrive, depart }
  let prevVi = 0;
  for (const c of charges) {
    if (c.vi <= prevVi || c.vi >= lastIdx) continue;
    knots.push({ vi: c.vi, enter: c.arrive, leave: c.depart });
    chargeAt.set(c.vi, c);
    prevVi = c.vi;
  }
  knots.push({ vi: lastIdx, enter: endSoc });

  // SOC entering vertex i: linear in geometry distance from the previous knot's
  // depart-SOC down to the next knot's arrive-SOC.
  const socAtVertex = (i) => {
    for (let k = 0; k < knots.length - 1; k++) {
      const a = knots[k];
      const b = knots[k + 1];
      if (i >= a.vi && i <= b.vi) {
        const ga = cum[a.vi];
        const gb = cum[b.vi];
        const f = gb > ga ? (cum[i] - ga) / (gb - ga) : 0;
        return a.leave + f * (b.enter - a.leave);
      }
    }
    return endSoc;
  };

  // FINE segmentation: break on colour change OR each 1% SOC tick OR a charge
  // vertex, so every piece carries a tight "X% -> Y%" transition for the hover
  // tooltip. These fine pieces drive the (invisible) hit-lines; the VISIBLE
  // line uses mergeBands() to fuse same-colour runs into smooth per-colour
  // bands, so the route never reads as a string of beads.
  const soc0 = socAtVertex(0);
  const segs = [];
  let cur = { positions: [geometry[0]], color: socColor(soc0), startSoc: soc0, endSoc: soc0 };
  let curBand = Math.round(soc0);
  for (let i = 1; i < geometry.length; i++) {
    const charge = chargeAt.get(i);
    const soc = charge ? charge.arrive : socAtVertex(i);
    cur.positions.push(geometry[i]);
    cur.endSoc = soc;
    if (charge) {
      // Hard jump at the station: close the pre-charge (low/red) segment here,
      // then open a fresh post-charge (topped-up/green) segment at the SAME
      // vertex so nothing downstream inherits the pre-charge colour.
      if (cur.positions.length > 1) segs.push(cur);
      const leave = charge.depart;
      cur = { positions: [geometry[i]], color: socColor(leave), startSoc: leave, endSoc: leave };
      curBand = Math.round(leave);
      continue;
    }
    const color = socColor(soc);
    const band = Math.round(soc);
    if ((color !== cur.color || band !== curBand) && i < geometry.length - 1) {
      segs.push(cur);
      cur = { positions: [geometry[i]], color, startSoc: soc, endSoc: soc };
      curBand = band;
    }
  }
  if (cur.positions.length > 1) segs.push(cur);
  return segs;
}

// Fuse consecutive same-colour FINE segments into per-colour BANDS for the
// visible line. Each fine segment starts at the vertex the previous one ended
// on, so concatenating (minus the shared first point) yields one continuous
// polyline per colour — smooth, with no per-1% round-cap beading.
function mergeBands(segs) {
  const out = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    if (last && last.color === s.color) {
      last.positions = last.positions.concat(s.positions.slice(1));
      last.endSoc = s.endSoc;
    } else {
      out.push({ positions: [...s.positions], color: s.color, startSoc: s.startSoc, endSoc: s.endSoc });
    }
  }
  return out;
}

// STATIC direction chevrons spaced evenly along the route, each rotated to the
// local travel direction (origin -> destination). Small white "^" arrowheads,
// like a map's direction-of-travel hints — no animation (replaced the old truck).
function RouteDirectionArrows({ geometry }) {
  const map = useMap();
  useEffect(() => {
    if (!Array.isArray(geometry) || geometry.length < 2) return undefined;

    const toRad = (d) => (d * Math.PI) / 180;
    const toDeg = (r) => (r * 180) / Math.PI;

    // Cumulative great-circle distance along the polyline.
    const cum = [0];
    for (let i = 1; i < geometry.length; i++) {
      const a = geometry[i - 1];
      const b = geometry[i];
      const R = 6371;
      const dLat = toRad(b[0] - a[0]);
      const dLng = toRad(b[1] - a[1]);
      const h =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLng / 2) ** 2;
      cum.push(cum[i - 1] + 2 * R * Math.asin(Math.sqrt(h)));
    }
    const total = cum[cum.length - 1] || 1;

    // Position + travel bearing at a given distance `d` along the route.
    const at = (d) => {
      let lo = 1;
      while (lo < cum.length - 1 && cum[lo] < d) lo++;
      const a = geometry[lo - 1];
      const b = geometry[lo];
      const segLen = cum[lo] - cum[lo - 1] || 1;
      const f = (d - cum[lo - 1]) / segLen;
      const pos = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
      const y = Math.sin(toRad(b[1] - a[1])) * Math.cos(toRad(b[0]));
      const x =
        Math.cos(toRad(a[0])) * Math.sin(toRad(b[0])) -
        Math.sin(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.cos(toRad(b[1] - a[1]));
      const bearing = (toDeg(Math.atan2(y, x)) + 360) % 360;
      return { pos, bearing };
    };

    // Space chevrons evenly along the route (~1 per 70 km, capped) so every leg of
    // a multi-stop trip shows its travel direction — not just the overall midpoint.
    const count = Math.max(1, Math.min(10, Math.round(total / 70)));
    const markers = [];
    for (let k = 1; k <= count; k++) {
      const { pos, bearing } = at((total * k) / (count + 1));
      const icon = L.divIcon({
        className: "",
        html:
          `<div class="route-chevron" style="transform:rotate(${bearing}deg)">` +
          `<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">` +
          `<path d="M5 15 L12 8 L19 15" fill="none" stroke="#ffffff" stroke-width="3.5" ` +
          `stroke-linecap="round" stroke-linejoin="round"/></svg></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8],
      });
      markers.push(
        L.marker(pos, {
          icon,
          interactive: false,
          keyboard: false,
          zIndexOffset: 600,
        }).addTo(map)
      );
    }

    return () => {
      markers.forEach((m) => map.removeLayer(m));
    };
  }, [map, geometry]);
  return null;
}

function FitBounds({ geometry }) {
  const map = useMap();
  useEffect(() => {
    if (!geometry || geometry.length === 0) return;
    if (geometry.length === 1) {
      map.setView(geometry[0], 11);
      return;
    }
    const bounds = L.latLngBounds(geometry);
    map.fitBounds(bounds, { padding: [40, 40] });
  }, [map, geometry]);
  return null;
}

// Recenters the map when an external coordinate is requested (from search box).
function Recenter({ target }) {
  const map = useMap();
  useEffect(() => {
    if (target && Number.isFinite(target.lat) && Number.isFinite(target.lng)) {
      map.setView([target.lat, target.lng], 12);
    }
  }, [map, target]);
  return null;
}

// Keep Leaflet's size in sync with its container. Fires several times after
// mount/trigger (the map can mount before its flex container has its final
// height, which otherwise leaves white space under the tiles) and on window
// resize so tiles always fill the card.
function InvalidateOnResize({ trigger }) {
  const map = useMap();
  useEffect(() => {
    const fire = () => map.invalidateSize();
    const timers = [60, 250, 500, 900].map((ms) => setTimeout(fire, ms));
    window.addEventListener("resize", fire);
    return () => {
      timers.forEach(clearTimeout);
      window.removeEventListener("resize", fire);
    };
  }, [map, trigger]);
  return null;
}

// On-map search box (light theme). Debounced; selecting recenters the map.
function SearchBox({ onPick }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const q = query.trim();
    if (q.length < 3) {
      setResults([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    const t = setTimeout(async () => {
      const res = await geocode(q);
      setResults(Array.isArray(res) ? res.slice(0, 6) : []);
      setOpen(true);
      setLoading(false);
    }, 350);
    return () => clearTimeout(t);
  }, [query]);

  function pick(r) {
    onPick({ ...r });
    setQuery(r.label);
    setOpen(false);
    setResults([]);
  }

  return (
    <div className="absolute top-5 left-5 z-[1000] w-80">
      <div className="group relative flex items-center">
        <span
          className="material-symbols-outlined absolute left-3.5 text-on-surface-variant pointer-events-none transition-colors duration-snappy group-focus-within:text-primary"
          style={{ fontSize: "22px" }}
        >
          search
        </span>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
          placeholder="Search locations…"
          className="w-full pl-11 pr-10 py-3.5 rounded-card bg-surface-lowest/95 backdrop-blur border border-outline-variant/50 text-base text-on-surface placeholder:text-on-surface-variant/70 outline-none focus:ring-2 focus:ring-primary/50 focus:border-primary/40 shadow-nx-md transition-[box-shadow,border-color] duration-smooth ease-nx-out"
        />
        {loading ? (
          <span
            className="material-symbols-outlined absolute right-2.5 text-primary animate-spin"
            style={{ fontSize: "18px" }}
          >
            progress_activity
          </span>
        ) : query ? (
          <button
            type="button"
            onClick={() => {
              setQuery("");
              setResults([]);
              setOpen(false);
            }}
            aria-label="Clear"
            className="absolute right-2.5 flex items-center justify-center rounded-pill p-0.5 text-on-surface-variant hover:text-on-surface hover:bg-surface transition-colors duration-snappy"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
              close
            </span>
          </button>
        ) : null}
      </div>

      {open && results.length > 0 && (
        <ul className="mt-2 w-full rounded-control bg-surface-lowest/95 backdrop-blur border border-outline-variant/50 shadow-nx-lg overflow-hidden max-h-60 overflow-y-auto p-1">
          {results.map((r, i) => (
            <li key={`${r.lat},${r.lng},${i}`}>
              <button
                type="button"
                onClick={() => pick(r)}
                className="flex w-full items-start gap-2.5 rounded-control px-2.5 py-2 text-left text-sm text-on-surface-variant hover:bg-surface-low hover:text-on-surface transition-colors duration-snappy"
              >
                <span
                  className="material-symbols-outlined mt-0.5 text-primary/80 shrink-0"
                  style={{ fontSize: "16px" }}
                >
                  location_on
                </span>
                <span className="leading-snug">{r.label}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

// Compact pill dock to switch base tiles (Streets / Satellite / Dark). The
// selected option is a raised neutral chip with an accent-coloured icon, so it
// reads clearly without a loud fill.
function TileSwitcher({ value, onChange }) {
  return (
    <div className="absolute bottom-5 left-5 z-[1000] flex items-center gap-0.5 rounded-pill bg-surface-lowest/90 backdrop-blur-md border border-outline-variant/50 shadow-nx-md p-1">
      {Object.entries(TILE_STYLES).map(([key, s]) => {
        const on = value === key;
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            aria-pressed={on}
            className={`flex items-center gap-1.5 rounded-pill px-3 py-1.5 text-[13px] font-medium transition-colors duration-snappy ease-nx-out nx-focus ${
              on
                ? "bg-surface text-on-surface shadow-nx-sm ring-1 ring-outline-variant/60"
                : "text-on-surface-variant hover:text-on-surface"
            }`}
          >
            <span
              className="material-symbols-outlined"
              style={{ fontSize: "18px", color: on ? "rgb(var(--c-primary))" : undefined }}
            >
              {s.icon}
            </span>
            {s.label}
          </button>
        );
      })}
    </div>
  );
}

export default function RouteMap({ plan, waypoints = [] }) {
  const [tileStyle, setTileStyle] = useState("streets");
  const [layers, setLayers] = useState(DEFAULT_LAYERS);
  const [searchTarget, setSearchTarget] = useState(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [userLoc, setUserLoc] = useState(null);

  const toggleLayer = (key) =>
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));

  const geometry = plan?.geometry || [];
  const socSegs = useMemo(
    () => buildSocSegments(geometry, plan?.socProfile, plan?.summary?.distanceKm, plan?.chargingStops),
    [geometry, plan?.socProfile, plan?.summary?.distanceKm, plan?.chargingStops]
  );
  // Smooth visible line = same-colour fine segments fused into bands; the fine
  // socSegs stay as the precise hover-tooltip hit-lines.
  const socBands = useMemo(() => mergeBands(socSegs), [socSegs]);

  // Identify origin/destinations by ROLE, not array position: when the user
  // adds a destination before setting an origin, the destination must NOT be
  // drawn with the origin pin. Falls back to position-based for any legacy
  // caller that doesn't tag waypoints with `kind`.
  const hasKinds = waypoints.some((w) => w?.kind);
  const origin = hasKinds ? waypoints.find((w) => w.kind === "origin") : waypoints[0];
  const dests = hasKinds ? waypoints.filter((w) => w.kind === "dest") : waypoints.slice(1);
  // Number destination pins by the order the route ACTUALLY visits them. With
  // stop-order optimisation off (default) the plan keeps the typed order, so the
  // pin number is just the list position; when it's on, `optimizedOrder[k]` is the
  // typed index visited at position k, so a dest's pin number is its position in
  // that array. This keeps the pin numbers consistent with the drawn route line.
  const optOrder = plan?.optimization?.optimizedOrder;
  // `typedIdx` is the destination's 0-based position in the dispatcher's typed
  // list. With optimisation off the pin number is just typedIdx+1; with it on,
  // `optOrder` maps typed index -> visit order, so the pin follows the route.
  const visitNum = (typedIdx) =>
    Array.isArray(optOrder) && optOrder.indexOf(typedIdx) >= 0
      ? optOrder.indexOf(typedIdx) + 1
      : typedIdx + 1;
  const center = geometry[0] || [51.0, 10.2];
  const stops = (plan?.chargingStops || []).filter(
    (s) => Number.isFinite(s.lat) && Number.isFinite(s.lng)
  );
  const incidents = (plan?.traffic?.incidents || []).filter(
    (i) => Number.isFinite(i.lat) && Number.isFinite(i.lng)
  );

  const tile = TILE_STYLES[tileStyle];

  return (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 z-[2000] h-screen w-screen bg-background"
          : "relative h-full w-full"
      }
    >
      <MapContainer
        center={center}
        zoom={6}
        minZoom={4}
        maxZoom={18}
        scrollWheelZoom={true}
        zoomControl={false}
        attributionControl={true}
        style={{ height: "100%", width: "100%", background: "#eff4ff" }}
      >
        <TileLayer
          key={tileStyle}
          url={tile.url}
          attribution={tile.attribution}
          subdomains={tile.subdomains || "abc"}
        />
        {/* Transparent labels/roads overlay (satellite hybrid when provider has no native labels). */}
        {tile.overlay && (
          <TileLayer
            key={`${tileStyle}-overlay`}
            url={tile.overlay}
            subdomains={tile.subdomains || "abc"}
          />
        )}

        {/* Current location — Google/Apple-style blue pulsing dot. */}
        {userLoc && (
          <Marker
            position={[userLoc.lat, userLoc.lng]}
            icon={gpsIcon()}
            zIndexOffset={1000}
          >
            <Tooltip direction="top" offset={[0, -8]}>You are here</Tooltip>
          </Marker>
        )}

        {/* Route — SOC-gradient when Battery Drain on, plain primary line otherwise. */}
        {layers.route &&
          (layers.drain ? (
            <>
              {/* One continuous dark casing under the WHOLE route so the line
                  reads crisply over light/satellite tiles with no per-segment
                  caps to bead the line. */}
              <Polyline
                positions={geometry}
                pathOptions={{
                  color: "#0b1c30",
                  weight: 12,
                  opacity: 0.18,
                  lineCap: "round",
                  lineJoin: "round",
                }}
              />
              {/* Visible line: one smooth polyline per colour band. */}
              {socBands.map((seg, i) => (
                <Polyline
                  key={`band-${i}`}
                  positions={seg.positions}
                  pathOptions={{
                    color: seg.color,
                    weight: 8,
                    opacity: 0.95,
                    lineCap: "round",
                    lineJoin: "round",
                  }}
                />
              ))}
              {/* Fine invisible hit-lines: a wide hover target per 1%-SOC piece,
                  so the sticky tooltip shows a tight "X% → Y%" local transition. */}
              {socSegs.map((seg, i) => (
                <Polyline
                  key={`hit-${i}`}
                  positions={seg.positions}
                  pathOptions={{ color: seg.color, weight: 16, opacity: 0 }}
                >
                  <Tooltip sticky direction="top" offset={[0, -6]} className="battery-tip">
                    <span className="bt-label">Battery Level</span>
                    <span className="bt-value">
                      <span className="bt-dot" style={{ background: seg.color }} />
                      {Math.round(seg.startSoc) === Math.round(seg.endSoc)
                        ? `${Math.round(seg.startSoc)}%`
                        : `${Math.round(seg.startSoc)}% → ${Math.round(seg.endSoc)}%`}
                    </span>
                  </Tooltip>
                </Polyline>
              ))}
            </>
          ) : geometry.length >= 2 ? (
            <Fragment>
              <Polyline
                positions={geometry}
                pathOptions={{
                  color: "#0b1c30",
                  weight: 12,
                  opacity: 0.18,
                  lineCap: "round",
                  lineJoin: "round",
                }}
              />
              <Polyline
                positions={geometry}
                pathOptions={{
                  color: "#006d32",
                  weight: 8,
                  opacity: 0.95,
                  lineCap: "round",
                  lineJoin: "round",
                }}
              />
            </Fragment>
          ) : null)}

        {/* A single live "runner" travelling source -> destination on a loop. */}
        {layers.route && geometry.length >= 2 && <RouteDirectionArrows geometry={geometry} />}

        {/* Planned stops — origin + destinations. */}
        {layers.stops && origin && Number.isFinite(origin.lat) && (
          <Marker position={[origin.lat, origin.lng]} icon={pinIcon("origin")} title={`Origin: ${origin.label || ""}`}>
            <Tooltip direction="top" offset={[0, -46]}>
              <span style={{ fontWeight: 600 }}>Origin</span>
              <br />
              {origin.label}
            </Tooltip>
          </Marker>
        )}

        {layers.stops &&
          dests.map((d, i) => {
            if (!Number.isFinite(d.lat)) return null;
            // Number by the destination's TYPED list position (carried as `num`),
            // not its index among the geocoded subset — so an un-geocoded earlier
            // stop can't renumber the later ones. Falls back to array position.
            const typedIdx = Number.isInteger(d.num) ? d.num - 1 : i;
            const n = visitNum(typedIdx);
            return (
              <Marker key={`dest-${i}`} position={[d.lat, d.lng]} icon={pinIcon("dest", n)} title={`Destination ${n}: ${d.label || ""}`}>
                <Tooltip direction="top" offset={[0, -46]}>
                  <span style={{ fontWeight: 600 }}>Destination {n}</span>
                  <br />
                  {d.label}
                </Tooltip>
              </Marker>
            );
          })}

        {/* Final-approach connector: a short dashed line from the route to each
            charger that sits off the truck-accessible road, so it's clear where
            to leave the route to reach the station. */}
        {layers.charging &&
          stops.map((s, i) =>
            Number.isFinite(s.routeLat) && Number.isFinite(s.routeLng) && (s.routeGapM ?? 0) > 25 ? (
              <Polyline
                key={`charge-link-${i}`}
                positions={[
                  [s.routeLat, s.routeLng],
                  [s.lat, s.lng],
                ]}
                pathOptions={{ color: "#f5a623", weight: 3, opacity: 0.9, dashArray: "3 7", lineCap: "round" }}
              />
            ) : null
          )}

        {/* Charging stations. */}
        {layers.charging &&
          stops.map((s, i) => (
            <Marker key={`charge-${i}`} position={[s.lat, s.lng]} icon={chargeIcon(i + 1)} title={`Charging stop ${i + 1}: ${s.name || ""}`}>
              <Tooltip direction="top" offset={[0, -14]} className="map-tip charge-tip">
                <span className="ct-head">
                  <span className="material-symbols-outlined mt-icon" style={{ color: "#f5a623" }}>
                    ev_station
                  </span>
                  <span className="mt-name">
                    Stop {i + 1}: {s.name || `Charging Stop ${i + 1}`}
                  </span>
                </span>
                {(s.maxPowerKw || (s.connectors && s.connectors.length)) && (
                  <span className="ct-line">
                    {s.maxPowerKw ? `⚡ ${s.maxPowerKw} kW` : "⚡"}
                    {s.connectors && s.connectors.length
                      ? ` · ${s.connectors.map((c) => c.label).join(", ")}`
                      : ""}
                  </span>
                )}
                <span className="ct-line ct-sub">
                  {s.availability ? (
                    <span
                      className="ct-avail"
                      style={s.availability.available > 0 ? undefined : { color: "#ff8a80" }}
                      title="Live availability at planning time — the truck arrives hours later, so the number of free chargers may differ on arrival."
                    >
                      {s.availability.available} free
                    </span>
                  ) : null}
                  {Number.isFinite(s.kWh) ? (
                    <span>
                      +{Math.round(s.kWh)} kWh
                      {Number.isFinite(s.chargeMinutes) ? ` · ~${fmtChargeTime(s.chargeMinutes)}` : ""}
                    </span>
                  ) : null}
                </span>
              </Tooltip>
            </Marker>
          ))}

        {/* Live traffic incidents (accidents, jams, closures, road works). */}
        {layers.incidents &&
          incidents.map((inc, i) => (
            <Marker
              key={`inc-${i}`}
              position={[inc.lat, inc.lng]}
              icon={incidentIcon(inc)}
              zIndexOffset={500}
              title="Traffic incident near the route (its delay is local — the route may steer around it)"
            >
              <Tooltip direction="top" offset={[0, -14]} className="map-tip">
                <span
                  className="material-symbols-outlined mt-icon"
                  style={{ color: incidentColor(inc.magnitude) }}
                >
                  {incidentGlyph(inc.category)}
                </span>
                <span className="mt-name">
                  {inc.description}
                  {inc.road ? ` · ${inc.road}` : ""}
                </span>
                {inc.delayS > 0 ? (
                  <span className="mt-meta">+{Math.round(inc.delayS / 60)} min</span>
                ) : null}
              </Tooltip>
            </Marker>
          ))}

        <FitBounds geometry={geometry} />
        <Recenter target={searchTarget} />
        <InvalidateOnResize trigger={isFullscreen} />
        <MapControls
          isFullscreen={isFullscreen}
          onToggleFullscreen={() => setIsFullscreen((f) => !f)}
          onLocated={setUserLoc}
        />
      </MapContainer>

      {/* Overlays outside MapContainer (no map ctx needed). */}
      <SearchBox onPick={setSearchTarget} />
      <MapLayersPanel layers={layers} onToggle={toggleLayer} />
      <TileSwitcher value={tileStyle} onChange={setTileStyle} />
    </div>
  );
}
