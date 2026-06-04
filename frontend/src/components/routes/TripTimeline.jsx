// Ordered vertical timeline of trip segments: drive / rest / charge.
// Reads the `segments` array from PlanResult (light theme).
import { to12h } from "../../lib/time.js";

function fmtTime(t) {
  if (!t) return "--:--";
  return to12h(t);
}

function fmtRange(start, end) {
  return `${fmtTime(start)} – ${fmtTime(end)}`;
}

function fmtDur(min) {
  if (min == null) return "";
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  if (h && m) return `${h}h ${m}m`;
  if (h) return `${h}h`;
  return `${m}m`;
}

function socColor(soc) {
  if (soc >= 80) return "#15803d"; // 80-100% deep green
  if (soc >= 60) return "#22c55e"; // 60-80% green
  if (soc >= 40) return "#eab308"; // 40-60% yellow
  if (soc >= 20) return "#f59e0b"; // 20-40% amber
  return "#ef4444"; // <20% red
}

function Marker({ icon, tint }) {
  return (
    <div
      className="flex items-center justify-center w-8 h-8 rounded-pill ring-1 shadow-nx-sm"
      style={{ background: `${tint}1f`, color: tint, borderColor: `${tint}55`, "--tw-ring-color": `${tint}40` }}
    >
      <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
        {icon}
      </span>
    </div>
  );
}

function DriveRow({ seg }) {
  const start = seg.socStart ?? 0;
  const end = seg.socEnd ?? 0;
  const limit = seg.limitMin || 270;
  // Average speed over this drive leg (km/h). With per-road posted limits
  // (Tier S) this varies leg-to-leg — autobahn legs near 80, legs through
  // towns/30-zones lower. The total still equals TomTom's measured time.
  const speedKph = seg.durationMin > 0 ? (seg.km || 0) / (seg.durationMin / 60) : 0;
  return (
    <div className="flex-1 rounded-control nx-card-inset px-3 py-2.5 transition-colors duration-snappy ease-nx-out hover:border-outline-variant/60">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-on-surface">
          Drive
          <span className="text-on-surface-variant font-normal">
            {" "}
            · {Math.round(seg.km || 0)} km · {fmtDur(seg.durationMin)}
          </span>
        </p>
        <div className="flex items-center gap-2 shrink-0 ml-2">
          {speedKph > 0 && (
            <span
              className="inline-flex items-center gap-1 rounded-pill px-2 py-0.5 text-[11px] font-semibold tabular-nums bg-secondary/10 text-secondary ring-1 ring-secondary/20"
              title="Average speed over this leg (varies by posted road limits)"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>speed</span>
              {Math.round(speedKph)} km/h
            </span>
          )}
          <span className="text-[11px] text-on-surface-variant tabular-nums">
            {fmtDur(seg.durationMin)} / {fmtDur(limit)}
          </span>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span className="text-[11px] tabular-nums" style={{ color: socColor(start) }}>
          {Math.round(start)}%
        </span>
        <div className="relative flex-1 h-1.5 rounded-full bg-surface overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 right-0 rounded-full"
            style={{ background: `linear-gradient(90deg, ${socColor(start)}, ${socColor(end)})` }}
          />
        </div>
        <span className="text-[11px] tabular-nums" style={{ color: socColor(end) }}>
          {Math.round(end)}%
        </span>
      </div>
    </div>
  );
}

function RestCard({ seg }) {
  return (
    <div className="flex-1 rounded-control bg-amber-500/10 border border-amber-500/30 px-3 py-2.5">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-amber-700">{seg.label || "Rest Break"}</p>
        <p className="text-[11px] text-amber-700/70 tabular-nums">{fmtDur(seg.durationMin)}</p>
      </div>
      <p className="text-[11px] text-amber-700/60 mt-0.5 tabular-nums">
        {fmtRange(seg.startTime, seg.endTime)}
      </p>
    </div>
  );
}

function ChargeCard({ seg }) {
  const start = seg.socStart ?? 0;
  const end = seg.socEnd ?? 0;
  return (
    <div className="flex-1 rounded-control bg-primary/8 border border-primary/25 px-3 py-2.5">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-primary truncate">
          {seg.station?.name || "Charging Stop"}
        </p>
        <p className="text-[11px] text-primary/70 tabular-nums shrink-0 ml-2">
          {fmtDur(seg.durationMin)}
        </p>
      </div>
      <p className="text-[11px] text-on-surface-variant mt-0.5 tabular-nums">
        {fmtRange(seg.startTime, seg.endTime)}
      </p>
      <div className="mt-2 flex items-center justify-between text-[11px]">
        <span className="text-on-surface-variant tabular-nums">
          <span style={{ color: socColor(start) }}>{Math.round(start)}%</span>
          <span className="material-symbols-outlined align-middle text-on-surface-variant" style={{ fontSize: "14px" }}>
            arrow_right_alt
          </span>
          <span style={{ color: socColor(end) }}>{Math.round(end)}%</span>
        </span>
        <span className="text-on-surface tabular-nums">
          {Math.round(seg.kWh || 0)} kWh
          {/* Cost only with a live per-kWh tariff; the flat estimate is hidden. */}
          {Number.isFinite(seg.pricePerKwh) && Number.isFinite(seg.costEur) && seg.costEur > 0 ? (
            <span className="text-on-surface-variant"> · ≈€{Math.round(seg.costEur)}</span>
          ) : null}
        </span>
      </div>
    </div>
  );
}

export default function TripTimeline({ segments = [] }) {
  if (!segments.length) {
    return <p className="text-sm text-on-surface-variant">No trip segments.</p>;
  }

  return (
    <ol className="space-y-2.5">
      {segments.map((seg, i) => {
        const isLast = i === segments.length - 1;
        let marker;
        let body;
        if (seg.type === "rest" || seg.type === "daily_rest") {
          marker = <Marker icon={seg.type === "daily_rest" ? "bedtime" : "hotel"} tint="#f59e0b" />;
          body = <RestCard seg={seg} />;
        } else if (seg.type === "unload") {
          marker = <Marker icon="package_2" tint="#0059bb" />;
          body = <RestCard seg={seg} />;
        } else if (seg.type === "charge") {
          marker = <Marker icon="ev_station" tint="#006d32" />;
          body = <ChargeCard seg={seg} />;
        } else {
          marker = <Marker icon="local_shipping" tint="#0059bb" />;
          body = <DriveRow seg={seg} />;
        }
        return (
          <li key={i} className="flex gap-3">
            <div className="flex flex-col items-center">
              {marker}
              {!isLast && <div className="flex-1 w-0.5 mt-1 rounded-pill bg-gradient-to-b from-outline-variant/60 to-outline-variant/25" />}
            </div>
            {body}
          </li>
        );
      })}
    </ol>
  );
}
