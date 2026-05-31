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

function socColor(soc) {
  if (soc >= 50) return "#00d166";
  if (soc >= 25) return "#f59e0b";
  return "#ba1a1a";
}

function pinIcon(kind) {
  const color = kind === "origin" ? "#1ca64c" : "#0059bb";
  // Teardrop location pin: colored drop, white ring. Origin shows a center dot,
  // destinations a small flag glyph inside the ring.
  const center =
    kind === "origin"
      ? `<circle cx="12" cy="12" r="3" fill="${color}"/>`
      : "";
  const flag =
    kind === "origin"
      ? ""
      : `<span class="material-symbols-outlined" style="position:absolute;left:19px;top:18px;transform:translate(-50%,-50%);font-size:14px;color:${color};">flag</span>`;
  return L.divIcon({
    className: "",
    html: `<div style="position:relative;width:38px;height:48px;filter:drop-shadow(0 3px 4px rgba(0,0,0,0.35));">
      <svg width="38" height="48" viewBox="0 0 24 32" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 0C5.37 0 0 5.37 0 12c0 8.5 10.5 18.7 11.1 19.3a1.3 1.3 0 0 0 1.8 0C13.5 30.7 24 20.5 24 12 24 5.37 18.63 0 12 0z" fill="${color}" stroke="#fff" stroke-width="1.5"/>
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
    html: `<div style="transform:translate(-50%,-50%);background:#f59e0b;color:#fff;border-radius:9999px;width:24px;height:24px;display:flex;align-items:center;justify-content:center;font:600 12px Inter,sans-serif;border:2px solid #fff;box-shadow:0 2px 5px rgba(0,0,0,0.3);">${num}</div>`,
    iconSize: [24, 24],
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
    html: `<div style="transform:translate(-50%,-50%);background:${color};color:#fff;border-radius:9999px;width:26px;height:26px;display:flex;align-items:center;justify-content:center;border:2px solid #fff;box-shadow:0 2px 5px rgba(0,0,0,0.35);">
      <span class="material-symbols-outlined" style="font-size:16px;">${incidentGlyph(inc.category)}</span>
    </div>`,
    iconSize: [26, 26],
    iconAnchor: [0, 0],
  });
}

// Map the SOC profile onto the geometry and emit colored sub-segments.
function buildSocSegments(geometry, socProfile, totalKm) {
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
  const geomTotal = cum[cum.length - 1] || totalKm;
  const scale = totalKm / geomTotal;

  const socAt = (km) => {
    if (km <= socProfile[0].distKm) return socProfile[0].soc;
    const last = socProfile[socProfile.length - 1];
    if (km >= last.distKm) return last.soc;
    for (let i = 1; i < socProfile.length; i++) {
      if (socProfile[i].distKm >= km) {
        const a = socProfile[i - 1];
        const b = socProfile[i];
        const f = (km - a.distKm) / (b.distKm - a.distKm || 1);
        return a.soc + f * (b.soc - a.soc);
      }
    }
    return last.soc;
  };

  // Break a new sub-segment whenever the bucket color changes OR the rounded
  // battery percentage ticks down by 1, so each hoverable piece carries a tight
  // "X% → Y%" transition (matching the reference battery-level tooltip).
  const soc0 = socAt(0);
  const segs = [];
  let cur = { positions: [geometry[0]], color: socColor(soc0), startSoc: soc0, endSoc: soc0 };
  let curBand = Math.round(soc0);
  for (let i = 1; i < geometry.length; i++) {
    const km = cum[i] * scale;
    const soc = socAt(km);
    const color = socColor(soc);
    const band = Math.round(soc);
    cur.positions.push(geometry[i]);
    cur.endSoc = soc;
    if ((color !== cur.color || band !== curBand) && i < geometry.length - 1) {
      segs.push(cur);
      cur = { positions: [geometry[i]], color, startSoc: soc, endSoc: soc };
      curBand = band;
    }
  }
  if (cur.positions.length > 1) segs.push(cur);
  return segs;
}

// A single marker that travels the route source -> destination on a loop,
// giving a live sense of direction/progress without cluttering the line.
// Animated imperatively via requestAnimationFrame (rAF timestamp only — no
// per-frame React re-render) and torn down cleanly when the route changes.
function RouteRunner({ geometry }) {
  const map = useMap();
  useEffect(() => {
    if (!Array.isArray(geometry) || geometry.length < 2) return undefined;

    const toRad = (d) => (d * Math.PI) / 180;
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

    const icon = L.divIcon({
      className: "",
      html: `<div class="route-runner"><span class="rr-halo"></span><video class="rr-video" src="/truck.mp4" autoplay loop muted playsinline></video></div>`,
      iconSize: [44, 44],
      iconAnchor: [22, 22],
    });
    const marker = L.marker(geometry[0], {
      icon,
      interactive: false,
      keyboard: false,
      zIndexOffset: 650,
    }).addTo(map);

    // Respect the OS "reduce motion" setting (WCAG 2.2.2): pause the looping
    // video and leave a static truck marker rather than animating it along the
    // route. CSS can't stop <video loop>, so this is handled here in JS.
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

    // Some browsers don't autoplay a freshly-injected <video>; nudge it and slow
    // the clip down so the truck reads as a calm cruise rather than a blur.
    const video = marker.getElement()?.querySelector("video");
    if (video) {
      if (reduceMotion) {
        video.removeAttribute("loop");
        video.pause();
      } else {
        video.playbackRate = 0.5;
        video.play().catch(() => {});
      }
    }

    // Slow, smooth lap: scale with route length, clamped to 16-32 s.
    const duration = Math.min(32000, Math.max(16000, total * 45));

    const posAt = (d) => {
      if (d <= 0) return geometry[0];
      if (d >= total) return geometry[geometry.length - 1];
      let lo = 1;
      while (lo < cum.length - 1 && cum[lo] < d) lo++;
      const a = geometry[lo - 1];
      const b = geometry[lo];
      const segLen = cum[lo] - cum[lo - 1] || 1;
      const f = (d - cum[lo - 1]) / segLen;
      return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
    };

    let raf;
    let start = null;
    const tick = (ts) => {
      if (start === null) start = ts;
      const t = ((ts - start) % duration) / duration; // 0..1, loops
      marker.setLatLng(posAt(t * total));
      raf = requestAnimationFrame(tick);
    };
    // Only drive the moving truck when motion is allowed; otherwise it stays put.
    if (!reduceMotion) {
      raf = requestAnimationFrame(tick);
    }

    return () => {
      if (raf) cancelAnimationFrame(raf);
      map.removeLayer(marker);
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
      <div className="relative flex items-center">
        <span
          className="material-symbols-outlined absolute left-3 text-on-surface-variant pointer-events-none"
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
          className="w-full pl-11 pr-10 py-3.5 rounded-2xl bg-surface-lowest/95 backdrop-blur border border-outline-variant/60 text-base text-on-surface placeholder:text-on-surface-variant/70 outline-none focus:ring-2 focus:ring-primary/40 shadow-md transition"
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
            className="absolute right-2 text-on-surface-variant hover:text-on-surface transition-colors"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
              close
            </span>
          </button>
        ) : null}
      </div>

      {open && results.length > 0 && (
        <ul className="mt-1 w-full rounded-xl bg-surface-lowest border border-outline-variant/60 shadow-lg overflow-hidden max-h-60 overflow-y-auto">
          {results.map((r, i) => (
            <li key={`${r.lat},${r.lng},${i}`}>
              <button
                type="button"
                onClick={() => pick(r)}
                className="flex w-full items-start gap-2 px-3 py-2 text-left text-sm text-on-surface-variant hover:bg-surface-low transition-colors"
              >
                <span
                  className="material-symbols-outlined mt-0.5 text-on-surface-variant shrink-0"
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

// Segmented tile-style switcher (Streets / Satellite / Dark).
function TileSwitcher({ value, onChange }) {
  return (
    <div className="absolute bottom-5 left-5 z-[1000] flex rounded-2xl bg-surface-lowest/95 backdrop-blur border border-outline-variant/60 shadow-md overflow-hidden">
      {Object.entries(TILE_STYLES).map(([key, s]) => {
        const on = value === key;
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange(key)}
            className={`flex items-center gap-2 px-4 py-3 text-sm font-medium transition-colors ${
              on
                ? "bg-primary text-white"
                : "text-on-surface-variant hover:bg-primary/10 hover:text-primary"
            }`}
          >
            <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
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
    () => buildSocSegments(geometry, plan?.socProfile, plan?.summary?.distanceKm),
    [geometry, plan?.socProfile, plan?.summary?.distanceKm]
  );

  const origin = waypoints[0];
  const dests = waypoints.slice(1);
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
            socSegs.map((seg, i) => (
              <Fragment key={`soc-${i}`}>
                <Polyline
                  positions={seg.positions}
                  pathOptions={{
                    color: seg.color,
                    weight: 8,
                    opacity: 0.95,
                    lineCap: "round",
                    lineJoin: "round",
                  }}
                />
                {/* Invisible wide hit-line widens the hover target; the sticky
                    tooltip then follows the cursor showing the battery drain. */}
                <Polyline
                  positions={seg.positions}
                  pathOptions={{ color: seg.color, weight: 16, opacity: 0 }}
                >
                  <Tooltip sticky direction="top" offset={[0, -6]} className="battery-tip">
                    <span className="bt-label">Battery Level</span>
                    <span className="bt-value">
                      <span className="bt-dot" style={{ background: seg.color }} />
                      {Math.round(seg.startSoc)}% → {Math.round(seg.endSoc)}%
                    </span>
                  </Tooltip>
                </Polyline>
              </Fragment>
            ))
          ) : geometry.length >= 2 ? (
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
          ) : null)}

        {/* A single live "runner" travelling source -> destination on a loop. */}
        {layers.route && geometry.length >= 2 && <RouteRunner geometry={geometry} />}

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
          dests.map((d, i) =>
            Number.isFinite(d.lat) ? (
              <Marker key={`dest-${i}`} position={[d.lat, d.lng]} icon={pinIcon("dest")} title={`Destination ${i + 1}: ${d.label || ""}`}>
                <Tooltip direction="top" offset={[0, -46]}>
                  <span style={{ fontWeight: 600 }}>Destination {i + 1}</span>
                  <br />
                  {d.label}
                </Tooltip>
              </Marker>
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
                    <span className="ct-avail">
                      {s.availability.available} of {s.availability.total} free
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
              title="Traffic incident on route"
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
