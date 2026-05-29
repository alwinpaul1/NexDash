import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Polyline, Tooltip, ZoomControl } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { getRoute } from "../lib/routes";
import MovingTruck from "./MovingTruck";

// EV-green theme status colors (match the legend below).
const STATUS = {
  enroute: "#00d166", // accent
  charging: "#0059bb", // secondary
  low: "#ba1a1a", // error
};

// Live fleet positions across the German long-haul corridor. [lat, lng].
const TRUCKS = [
  { id: "EA-204", city: "München", pos: [48.137, 11.575], status: "enroute" },
  { id: "EA-118", city: "Frankfurt", pos: [50.11, 8.682], status: "charging" },
  { id: "EA-377", city: "Hamburg", pos: [53.551, 9.993], status: "enroute" },
  { id: "EA-091", city: "Berlin", pos: [52.52, 13.405], status: "low" },
  { id: "EA-256", city: "Nürnberg", pos: [49.452, 11.077], status: "enroute" },
];

// Dashed route segments along the corridor (München → Frankfurt → Hamburg, Frankfurt → Berlin).
const ROUTES = [
  { from: "München", to: "Hamburg", color: "#9ca3af", path: [[48.137, 11.575], [50.11, 8.682], [53.551, 9.993]] },
  { from: "Frankfurt", to: "Berlin", color: "#00d166", path: [[50.11, 8.682], [49.452, 11.077], [52.52, 13.405]] },
];

const STATUS_LABEL = { enroute: "En route", charging: "Charging", low: "Low SOC" };

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

export default function MapOverview() {
  // Real road geometry per corridor route, keyed by `${from}-${to}`.
  // Starts empty; falls back to the straight path until getRoute resolves.
  const [roads, setRoads] = useState({});

  useEffect(() => {
    let active = true;
    ROUTES.forEach((r) => {
      getRoute(r.path).then((points) => {
        if (!active) return;
        setRoads((prev) => ({ ...prev, [`${r.from}-${r.to}`]: points }));
      });
    });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="xl:col-span-2 bg-surface-lowest rounded-2xl border border-outline-variant/40 overflow-hidden">
      <div className="flex items-center justify-between px-5 pt-5">
        <div>
          <h2 className="font-headline font-semibold text-lg text-on-surface">Fleet Map Overview</h2>
          <p className="text-xs text-on-surface-variant">Live positions</p>
        </div>
        <div className="flex items-center gap-3 text-xs text-on-surface-variant">
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-accent"></span> En route</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-secondary"></span> Charging</span>
          <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-error"></span> Low SOC</span>
        </div>
      </div>
      <div className="relative m-5 mt-4 h-72 rounded-xl overflow-hidden border border-outline-variant/40">
        <MapContainer
          center={[51.0, 10.2]}
          zoom={6}
          minZoom={5}
          maxZoom={18}
          scrollWheelZoom={true}
          doubleClickZoom={true}
          zoomControl={false}
          attributionControl={false}
          style={{ height: "100%", width: "100%", background: "#eff4ff" }}
        >
          <ZoomControl position="bottomright" />
          <TileLayer url={TILES.url} attribution={TILES.attribution} subdomains={TILES.subdomains} />

          {ROUTES.map((r) => {
            const key = `${r.from}-${r.to}`;
            const roadPoints = roads[key];
            const loaded = Boolean(roadPoints);
            // Distinct key per state so React mounts a FRESH Polyline when the
            // road geometry arrives — otherwise react-leaflet reuses the same
            // Leaflet layer and setStyle never clears the fallback's dashArray.
            return (
              <Polyline
                key={loaded ? `${key}-road` : `${key}-fallback`}
                positions={roadPoints || r.path}
                pathOptions={{
                  color: r.color,
                  weight: loaded ? 4 : 2.5,
                  opacity: loaded ? 0.9 : 0.85,
                  lineCap: "round",
                  lineJoin: "round",
                  dashArray: loaded ? null : "7 6",
                }}
              />
            );
          })}

          {ROUTES.map((r) => {
            const key = `${r.from}-${r.to}`;
            const roadPoints = roads[key];
            return roadPoints ? (
              <MovingTruck key={`truck-${key}`} path={roadPoints} color={r.color} />
            ) : null;
          })}

          {TRUCKS.map((t) => (
            <CircleMarker
              key={t.id}
              center={t.pos}
              radius={8}
              pathOptions={{
                color: "#ffffff",
                weight: 2,
                fillColor: STATUS[t.status],
                fillOpacity: 1,
              }}
            >
              <Tooltip direction="top" offset={[0, -8]} opacity={1}>
                <span style={{ fontWeight: 600 }}>{t.city}</span> · {t.id}
                <br />
                {STATUS_LABEL[t.status]}
              </Tooltip>
            </CircleMarker>
          ))}
        </MapContainer>
      </div>
    </div>
  );
}
