import { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Tooltip, ZoomControl, useMap } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// Keep Leaflet sized to its (flex-grown) container, re-running on any resize —
// e.g. when the card stretches to match the Reachability Watch list height.
function FillSize() {
  const map = useMap();
  useEffect(() => {
    const fire = () => map.invalidateSize({ animate: false });
    // rAF-wrap so we measure after layout; observe the container for any
    // size change (card stretching once the fleet/reachability data loads).
    const ro = new ResizeObserver(() => requestAnimationFrame(fire));
    ro.observe(map.getContainer());
    const timers = [50, 200, 500, 1000, 2000].map((ms) => setTimeout(fire, ms));
    window.addEventListener("resize", fire);
    map.whenReady(fire);
    return () => {
      ro.disconnect();
      timers.forEach(clearTimeout);
      window.removeEventListener("resize", fire);
    };
  }, [map]);
  return null;
}

// EV-green theme status colors (match the legend below).
const COLORS = { enroute: "#00d166", charging: "#0059bb", low: "#ba1a1a" };

// Fallback fleet shown only if /api/fleet is unavailable.
const FALLBACK = [
  { id: "EA-204", name: "EA-204", lat: 48.137, lng: 11.575, soc: 58, status: "in_transit", atRisk: false },
  { id: "EA-118", name: "EA-118", lat: 50.11, lng: 8.682, soc: 41, status: "charging", atRisk: false },
  { id: "EA-377", name: "EA-377", lat: 53.551, lng: 9.993, soc: 72, status: "in_transit", atRisk: false },
  { id: "EA-091", name: "EA-091", lat: 52.52, lng: 13.405, soc: 17, status: "in_transit", atRisk: true },
  { id: "EA-256", name: "EA-256", lat: 49.452, lng: 11.077, soc: 63, status: "available", atRisk: false },
];

// Real map data from TomTom (set VITE_TOMTOM_API_KEY in frontend/.env).
// Falls back to the free CartoDB Positron basemap if no key is configured.
const TOMTOM_KEY = import.meta.env.VITE_TOMTOM_API_KEY;
const TILES = TOMTOM_KEY
  ? {
      url: `https://{s}.api.tomtom.com/map/1/tile/basic/main/{z}/{x}/{y}.png?key=${TOMTOM_KEY}&tileSize=256`,
      attribution: '&copy; <a href="https://www.tomtom.com">TomTom</a>',
      subdomains: ["a", "b", "c", "d"],
    }
  : {
      url: "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
      subdomains: ["a", "b", "c", "d"],
    };

function bucket(t) {
  if (t.status === "charging") return "charging";
  if (t.atRisk || t.status === "maintenance" || (typeof t.soc === "number" && t.soc < 20)) return "low";
  return "enroute";
}
const LABEL = { enroute: "En route", charging: "Charging", low: "Low SOC" };

// Teardrop pin with a truck glyph, colored by status (vs. a flat dot).
function pinIcon(kind) {
  const color = COLORS[kind];
  return L.divIcon({
    className: "",
    html: `<div style="transform:translate(-50%,-100%);display:flex;flex-direction:column;align-items:center;">
      <div style="background:${color};color:#fff;border-radius:9999px;width:28px;height:28px;display:flex;align-items:center;justify-content:center;box-shadow:0 2px 6px rgba(0,0,0,0.35);border:2px solid #fff;">
        <span class="material-symbols-outlined" style="font-size:16px;">local_shipping</span>
      </div>
      <div style="width:2px;height:7px;background:${color};margin-top:-1px;"></div>
    </div>`,
    iconSize: [28, 35],
    iconAnchor: [0, 0],
  });
}

export default function MapOverview() {
  // Real fleet positions from /api/fleet (same source as Reachability Watch).
  const [trucks, setTrucks] = useState(FALLBACK);

  useEffect(() => {
    let active = true;
    fetch("/api/fleet")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!active || !d || !Array.isArray(d.trucks)) return;
        const pts = d.trucks.filter((t) => Number.isFinite(t.lat) && Number.isFinite(t.lng));
        if (pts.length) setTrucks(pts);
      })
      .catch(() => {
        /* keep fallback fleet on failure */
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="xl:col-span-2 h-[640px] bg-surface-lowest rounded-2xl border border-outline-variant/40 overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-5 pt-5">
        <div>
          <h2 className="font-headline font-semibold text-lg text-on-surface">Fleet Map Overview</h2>
          <p className="text-xs text-on-surface-variant">Live positions · {trucks.length} trucks</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-on-surface-variant">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent"></span> En route</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-secondary"></span> Charging</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-error"></span> Low SOC</span>
        </div>
      </div>
      <div className="relative m-5 mt-4 flex-1 min-h-0 rounded-xl overflow-hidden border border-outline-variant/40">
        <MapContainer
          center={[51.0, 10.2]}
          zoom={6}
          minZoom={5}
          maxZoom={18}
          scrollWheelZoom={true}
          doubleClickZoom={true}
          zoomControl={false}
          attributionControl={true}
          style={{ height: "100%", width: "100%", background: "#eff4ff" }}
        >
          <ZoomControl position="bottomright" />
          <TileLayer url={TILES.url} attribution={TILES.attribution} subdomains={TILES.subdomains} />

          {trucks.map((t) => {
            const kind = bucket(t);
            return (
              <Marker key={t.id} position={[t.lat, t.lng]} icon={pinIcon(kind)}>
                <Tooltip direction="top" offset={[0, -30]} opacity={1}>
                  <span style={{ fontWeight: 600 }}>{t.name || t.id}</span>
                  {typeof t.soc === "number" ? ` · ${Math.round(t.soc)}% SOC` : ""}
                  <br />
                  {LABEL[kind]}
                  {t.nextStop && t.nextStop.label ? ` → ${t.nextStop.label}` : ""}
                </Tooltip>
              </Marker>
            );
          })}
        </MapContainer>
      </div>
    </div>
  );
}
