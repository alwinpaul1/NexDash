// Elevation profile area chart from plan.elevationProfile ([{distKm, elevM}]),
// with charging stops marked along the distance axis. Pure inline SVG (no deps).

import { useRef, useState } from "react";

function niceCeil(v) {
  if (v <= 0) return 1;
  const mag = Math.pow(10, Math.floor(Math.log10(v)));
  const n = v / mag;
  const step = n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10;
  return step * mag;
}

export default function ElevationProfile({ profile = [], chargingStops = [] }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);
  const pts = (profile || []).filter(
    (p) => p && Number.isFinite(p.distKm) && Number.isFinite(p.elevM)
  );

  if (pts.length < 2) {
    return (
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
        <div className="flex items-center gap-2 mb-2">
          <span className="material-symbols-outlined text-primary" style={{ fontSize: "20px" }}>
            terrain
          </span>
          <h3 className="font-headline font-semibold text-lg text-on-surface">Elevation Profile</h3>
        </div>
        <p className="text-sm text-on-surface-variant">No elevation data available for this route.</p>
      </div>
    );
  }

  // Viewbox coordinate system (responsive via preserveAspectRatio).
  const W = 800;
  const H = 220;
  const padL = 44;
  const padR = 16;
  const padT = 14;
  const padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const maxDist = pts[pts.length - 1].distKm || 1;
  const elevs = pts.map((p) => p.elevM);
  let minElev = Math.min(...elevs);
  let maxElev = Math.max(...elevs);
  if (maxElev - minElev < 1) maxElev = minElev + 1; // avoid flat-line div0
  // Pad the top a little for headroom.
  const span = maxElev - minElev;
  const yTop = maxElev + span * 0.1;
  const yBottom = Math.max(0, minElev - span * 0.1);

  const x = (d) => padL + (d / maxDist) * plotW;
  const y = (e) => padT + (1 - (e - yBottom) / (yTop - yBottom)) * plotH;

  const linePath = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p.distKm).toFixed(1)} ${y(p.elevM).toFixed(1)}`).join(" ");
  const areaPath =
    `M ${x(pts[0].distKm).toFixed(1)} ${(padT + plotH).toFixed(1)} ` +
    pts.map((p) => `L ${x(p.distKm).toFixed(1)} ${y(p.elevM).toFixed(1)}`).join(" ") +
    ` L ${x(pts[pts.length - 1].distKm).toFixed(1)} ${(padT + plotH).toFixed(1)} Z`;

  // Y gridlines.
  const yTicks = 3;
  const tickStep = niceCeil((yTop - yBottom) / yTicks);
  const ticks = [];
  for (let t = Math.ceil(yBottom / tickStep) * tickStep; t <= yTop; t += tickStep) {
    ticks.push(t);
  }

  // Interpolate elevation at an arbitrary distance for stop markers.
  const elevAt = (dist) => {
    if (dist <= pts[0].distKm) return pts[0].elevM;
    if (dist >= pts[pts.length - 1].distKm) return pts[pts.length - 1].elevM;
    for (let i = 1; i < pts.length; i++) {
      if (pts[i].distKm >= dist) {
        const a = pts[i - 1];
        const b = pts[i];
        const f = (dist - a.distKm) / (b.distKm - a.distKm || 1);
        return a.elevM + f * (b.elevM - a.elevM);
      }
    }
    return pts[pts.length - 1].elevM;
  };

  const stops = (chargingStops || [])
    .map((s) => (Number.isFinite(s.distKm) ? s.distKm : s.atKm))
    .filter((d) => Number.isFinite(d) && d > 0 && d < maxDist);

  // Map cursor → distance along the route, then read the interpolated elevation.
  // The SVG uses preserveAspectRatio="none", so viewBox X maps linearly to pixels.
  const handleMove = (e) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const relX = e.clientX - rect.left;
    const vbX = (relX / rect.width) * W;
    const dist = Math.min(Math.max(((vbX - padL) / plotW) * maxDist, 0), maxDist);
    const elev = elevAt(dist);
    setHover({
      distKm: dist,
      elevM: elev,
      // Pixel positions (relative to the SVG box) for the HTML overlay.
      px: (x(dist) / W) * rect.width,
      py: (y(elev) / H) * rect.height,
      topPx: (padT / H) * rect.height,
      botPx: ((padT + plotH) / H) * rect.height,
      flip: relX > rect.width * 0.62,
    });
  };

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
      <div className="flex items-center gap-2 mb-1">
        <span className="material-symbols-outlined text-primary" style={{ fontSize: "20px" }}>
          terrain
        </span>
        <h3 className="font-headline font-semibold text-lg text-on-surface">Elevation Profile</h3>
        <span className="text-[11px] text-on-surface-variant ml-auto">
          {Math.round(minElev)}–{Math.round(maxElev)} m
        </span>
      </div>
      <div className="relative">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${W} ${H}`}
        className="w-full h-56"
        preserveAspectRatio="none"
        onMouseMove={handleMove}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id="elevFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00d166" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#00d166" stopOpacity="0.04" />
          </linearGradient>
        </defs>

        {/* Y gridlines + labels */}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={padL} y1={y(t)} x2={W - padR} y2={y(t)} stroke="#e5eeff" strokeWidth="1" />
            <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize="11" fill="#3c4a3d">
              {Math.round(t)}
            </text>
          </g>
        ))}

        {/* X axis labels (start / mid / end) */}
        {[0, maxDist / 2, maxDist].map((d, i) => (
          <text
            key={i}
            x={Math.min(Math.max(x(d), padL + 12), W - padR - 12)}
            y={H - 8}
            textAnchor={i === 0 ? "start" : i === 2 ? "end" : "middle"}
            fontSize="11"
            fill="#3c4a3d"
          >
            {Math.round(d)} km
          </text>
        ))}

        {/* Area + line */}
        <path d={areaPath} fill="url(#elevFill)" />
        <path d={linePath} fill="none" stroke="#006d32" strokeWidth="2" strokeLinejoin="round" />

        {/* Charging stop markers */}
        {stops.map((d, i) => (
          <g key={i}>
            <line x1={x(d)} y1={padT} x2={x(d)} y2={padT + plotH} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3" />
            <circle cx={x(d)} cy={y(elevAt(d))} r="5" fill="#f59e0b" stroke="#ffffff" strokeWidth="2" />
          </g>
        ))}

        {/* Hover crosshair (vertical guide line) */}
        {hover && (
          <line
            x1={x(hover.distKm)}
            y1={padT}
            x2={x(hover.distKm)}
            y2={padT + plotH}
            stroke="#006d32"
            strokeWidth="1"
            strokeDasharray="3 3"
            vectorEffect="non-scaling-stroke"
          />
        )}
      </svg>

      {/* Hover dot + tooltip as crisp HTML overlay (undistorted by the SVG scale). */}
      {hover && (
        <>
          <div
            className="pointer-events-none absolute z-10 w-3 h-3 rounded-full bg-primary border-2 border-white shadow"
            style={{ left: hover.px, top: hover.py, transform: "translate(-50%, -50%)" }}
          />
          <div
            className="pointer-events-none absolute z-10 rounded-xl bg-on-surface text-white shadow-lg px-3 py-2 whitespace-nowrap"
            style={{
              left: hover.px,
              top: hover.topPx,
              transform: `translate(${hover.flip ? "calc(-100% - 10px)" : "10px"}, 0)`,
            }}
          >
            <div className="text-[13px] font-semibold leading-tight">{Math.round(hover.elevM)} m</div>
            <div className="text-[11px] text-white/70 leading-tight">{Math.round(hover.distKm)} km</div>
          </div>
        </>
      )}
      </div>
      {stops.length > 0 && (
        <div className="flex items-center gap-1.5 mt-1 text-[11px] text-on-surface-variant">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 ring-1 ring-white"></span>
          Charging stop
        </div>
      )}
    </div>
  );
}
