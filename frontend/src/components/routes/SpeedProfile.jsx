// Posted speed-limit profile (step chart) from plan.speedLimits ([{fromKm,toKm,kmh}]),
// capped at the 80 km/h truck legal max, with charging stops marked. Makes slow
// zones (30/50 km/h) visible — they're otherwise diluted into a leg's ~73 km/h
// average. The ETA is NOT derived from this; it stays TomTom's measured time, which
// already accounts for the truck averaging just under these signposted limits.
// Pure inline SVG (no deps), mirrors ElevationProfile.

import { useRef, useState } from "react";

const TRUCK_CAP_KPH = 80;

// Build contiguous speed-limit segments over [0, totalKm], capped at the truck max,
// filling any gaps with the cap (open-road default). Returns [{from, to, kmh}].
function buildSegments(speedLimits, totalKm) {
  const spans = (speedLimits || [])
    .filter((s) => s && Number.isFinite(s.fromKm) && Number.isFinite(s.toKm) && s.toKm > s.fromKm && Number.isFinite(s.kmh))
    .map((s) => ({ from: Math.max(0, s.fromKm), to: s.toKm, kmh: Math.min(s.kmh, TRUCK_CAP_KPH) }))
    .sort((a, b) => a.from - b.from);
  const end = Number.isFinite(totalKm) && totalKm > 0 ? totalKm : spans.length ? spans[spans.length - 1].to : 0;
  const segs = [];
  let cursor = 0;
  for (const sp of spans) {
    const from = Math.max(sp.from, cursor);
    if (from > cursor + 0.05) segs.push({ from: cursor, to: from, kmh: TRUCK_CAP_KPH });
    if (sp.to > from + 0.001) segs.push({ from, to: sp.to, kmh: sp.kmh });
    cursor = Math.max(cursor, sp.to);
  }
  if (end > cursor + 0.05) segs.push({ from: cursor, to: end, kmh: TRUCK_CAP_KPH });
  return segs;
}

export default function SpeedProfile({ speedLimits = [], totalKm = 0, chargingStops = [] }) {
  const svgRef = useRef(null);
  const [hover, setHover] = useState(null);
  const segs = buildSegments(speedLimits, totalKm);

  if (segs.length < 1) {
    return (
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
        <div className="flex items-center gap-2 mb-2">
          <span className="material-symbols-outlined" style={{ fontSize: "20px", color: "#0059bb" }}>speed</span>
          <h3 className="font-headline font-semibold text-lg text-on-surface">Speed Limit Profile</h3>
        </div>
        <p className="text-sm text-on-surface-variant">No posted speed-limit data available for this route.</p>
      </div>
    );
  }

  const W = 800, H = 200, padL = 44, padR = 16, padT = 14, padB = 28;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const maxDist = segs[segs.length - 1].to || 1;
  const yTop = 90; // headroom above the 80 km/h cap

  const x = (d) => padL + (d / maxDist) * plotW;
  const y = (v) => padT + (1 - v / yTop) * plotH;

  // Step line + area: each segment is a horizontal run; boundaries jump vertically.
  const stepPts = [];
  for (const s of segs) {
    stepPts.push([s.from, s.kmh]);
    stepPts.push([s.to, s.kmh]);
  }
  const linePath = stepPts.map((p, i) => `${i === 0 ? "M" : "L"} ${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`).join(" ");
  const areaPath =
    `M ${x(stepPts[0][0]).toFixed(1)} ${(padT + plotH).toFixed(1)} ` +
    stepPts.map((p) => `L ${x(p[0]).toFixed(1)} ${y(p[1]).toFixed(1)}`).join(" ") +
    ` L ${x(stepPts[stepPts.length - 1][0]).toFixed(1)} ${(padT + plotH).toFixed(1)} Z`;

  const ticks = [30, 50, 80].filter((t) => t <= yTop);
  const limitAt = (d) => {
    for (const s of segs) if (d >= s.from && d <= s.to) return s.kmh;
    return segs[segs.length - 1].kmh;
  };
  const stops = (chargingStops || [])
    .map((s) => (Number.isFinite(s?.distKm) ? s.distKm : s?.atKm))
    .filter((d) => Number.isFinite(d) && d > 0 && d < maxDist);

  // How much of the route sits below the truck cap (the "slow" share) — a quick badge.
  const slowKm = segs.filter((s) => s.kmh < TRUCK_CAP_KPH).reduce((a, s) => a + (s.to - s.from), 0);
  const minLimit = Math.min(...segs.map((s) => s.kmh));

  const handleMove = (e) => {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    const relX = e.clientX - rect.left;
    const vbX = (relX / rect.width) * W;
    const dist = Math.min(Math.max(((vbX - padL) / plotW) * maxDist, 0), maxDist);
    const v = limitAt(dist);
    setHover({
      distKm: dist,
      kmh: v,
      px: (x(dist) / W) * rect.width,
      py: (y(v) / H) * rect.height,
      topPx: (padT / H) * rect.height,
      flip: relX > rect.width * 0.62,
    });
  };

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
      <div className="flex items-center gap-2 mb-1">
        <span className="material-symbols-outlined" style={{ fontSize: "20px", color: "#0059bb" }}>speed</span>
        <h3 className="font-headline font-semibold text-lg text-on-surface">Speed Limit Profile</h3>
        <span
          className="text-[11px] text-on-surface-variant ml-auto"
          title="Posted speed limits along the route, capped at the 80 km/h truck legal max. The ETA uses TomTom's measured time, which already has the truck averaging just below these signs."
        >
          {minLimit < TRUCK_CAP_KPH ? `dips to ${minLimit} km/h · ${Math.round(slowKm)} km below 80` : "80 km/h throughout"}
        </span>
      </div>
      <div className="relative">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-52"
          preserveAspectRatio="none"
          onMouseMove={handleMove}
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            <linearGradient id="speedFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#0059bb" stopOpacity="0.28" />
              <stop offset="100%" stopColor="#0059bb" stopOpacity="0.04" />
            </linearGradient>
          </defs>

          {ticks.map((t) => (
            <g key={t}>
              <line x1={padL} y1={y(t)} x2={W - padR} y2={y(t)} stroke="#e5eeff" strokeWidth="1" />
              <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize="11" fill="#3c4a3d">{t}</text>
            </g>
          ))}

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

          <path d={areaPath} fill="url(#speedFill)" />
          <path d={linePath} fill="none" stroke="#0059bb" strokeWidth="2" strokeLinejoin="round" />

          {stops.map((d, i) => (
            <g key={i}>
              <line x1={x(d)} y1={padT} x2={x(d)} y2={padT + plotH} stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3" />
              <circle cx={x(d)} cy={y(limitAt(d))} r="5" fill="#f59e0b" stroke="#ffffff" strokeWidth="2" />
            </g>
          ))}

          {hover && (
            <line x1={x(hover.distKm)} y1={padT} x2={x(hover.distKm)} y2={padT + plotH} stroke="#0059bb" strokeWidth="1" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
          )}
        </svg>

        {hover && (
          <>
            <div
              className="pointer-events-none absolute z-10 w-3 h-3 rounded-full border-2 border-white shadow"
              style={{ left: hover.px, top: hover.py, transform: "translate(-50%, -50%)", background: "#0059bb" }}
            />
            <div
              className="pointer-events-none absolute z-10 rounded-xl bg-on-surface text-white shadow-lg px-3 py-2 whitespace-nowrap"
              style={{ left: hover.px, top: hover.topPx, transform: `translate(${hover.flip ? "calc(-100% - 10px)" : "10px"}, 0)` }}
            >
              <div className="text-[13px] font-semibold leading-tight">{Math.round(hover.kmh)} km/h limit</div>
              <div className="text-[11px] text-white/70 leading-tight">{Math.round(hover.distKm)} km</div>
            </div>
          </>
        )}
      </div>
      <p className="mt-1 text-[10px] leading-tight text-on-surface-variant/70">
        Posted limits (capped at the 80 km/h truck max). The ETA uses TomTom's measured time — the truck averages
        just below these signs once ramps, hills and traffic are factored, which is why a leg's mean reads ~73–76 km/h.
      </p>
      {stops.length > 0 && (
        <div className="flex items-center gap-1.5 mt-1 text-[11px] text-on-surface-variant">
          <span className="w-2.5 h-2.5 rounded-full bg-amber-500 ring-1 ring-white"></span>
          Charging stop
        </div>
      )}
    </div>
  );
}
