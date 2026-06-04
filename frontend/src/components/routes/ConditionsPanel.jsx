// Route conditions strip: avg temperature, avg wind, max gradient, total climb.
// Reads plan.conditions (enriched server-side from Open-Meteo).

function fmt(v, decimals = 1) {
  if (v === null || v === undefined || typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toFixed(decimals);
}

// Translate a wind direction (deg, meteorological "from") into a compass label.
function windCompass(deg) {
  if (deg === null || deg === undefined || !Number.isFinite(deg)) return null;
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round(((deg % 360) / 45)) % 8];
}

function Tile({ icon, value, unit, label, tint }) {
  return (
    <div className="group rounded-control nx-card-inset px-3 py-3 transition-colors duration-snappy ease-nx-out hover:border-outline-variant/60">
      <span
        className="material-symbols-outlined transition-transform duration-snappy ease-nx-out group-hover:scale-110"
        style={{ fontSize: "20px", color: tint }}
      >
        {icon}
      </span>
      <p className="mt-1 text-xl font-headline font-bold text-on-surface leading-none tabular-nums">
        {value}
        {unit ? <span className="text-sm font-medium text-on-surface-variant"> {unit}</span> : null}
      </p>
      <p className="text-[11px] text-on-surface-variant mt-1">{label}</p>
    </div>
  );
}

export default function ConditionsPanel({ conditions = {} }) {
  const c = conditions || {};
  const compass = windCompass(c.windDirDeg);

  // When Open-Meteo is unreachable / rate-limited, the planner fails fast to
  // seasonal-default weather and flat (0 m) terrain. Surface that honestly so the
  // dispatcher knows these figures are approximate.
  const wDeg = !!c.weatherDegraded;
  const eDeg = !!c.elevationDegraded;
  const degraded = wDeg || eDeg;
  const what = wDeg && eDeg ? "Weather & elevation" : wDeg ? "Weather" : "Elevation";

  return (
    <div className="nx-card p-5">
      <div className="flex items-center gap-2 mb-4">
        <span className="flex items-center justify-center w-8 h-8 rounded-control bg-secondary/10 text-secondary ring-1 ring-secondary/20">
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            partly_cloudy_day
          </span>
        </span>
        <h3 className="font-headline font-semibold text-lg text-on-surface">Route Conditions</h3>
        {degraded ? (
          <span className="text-[11px] font-medium ml-auto px-2 py-0.5 rounded-pill" style={{ background: "#f59e0b1f", color: "#b45309" }}>
            seasonal defaults
          </span>
        ) : (
          <span className="text-[11px] text-on-surface-variant ml-auto px-2 py-0.5 rounded-pill bg-surface ring-1 ring-outline-variant/40">via Open-Meteo</span>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
        <Tile icon="thermostat" value={fmt(c.avgTempC, 1)} unit="°C" label="Avg Temperature" tint="#f59e0b" />
        <Tile
          icon="air"
          value={fmt(c.avgWindMps, 1)}
          unit="m/s"
          label={compass ? `Avg Wind · ${compass}` : "Avg Wind"}
          tint="#0059bb"
        />
        <Tile icon="trending_up" value={fmt(c.maxGradientPct, 1)} unit="%" label="Max Gradient" tint="#ba1a1a" />
        <Tile icon="landscape" value={fmt(c.climbM, 0)} unit="m" label="Total Climb" tint="#006d32" />
      </div>
      {degraded && (
        <div className="mt-3 flex items-start gap-2 rounded-control border px-3 py-2 text-[12px]" style={{ background: "#f59e0b14", borderColor: "#f59e0b40" }}>
          <span className="material-symbols-outlined shrink-0" style={{ fontSize: "16px", color: "#d97706" }}>cloud_off</span>
          <span className="text-on-surface">
            <span className="font-semibold" style={{ color: "#d97706" }}>{what} data unavailable</span>{" "}
            (live provider rate-limited) — used seasonal/flat defaults, so energy and
            gradient figures are approximate. Try again shortly for live conditions.
          </span>
        </div>
      )}
    </div>
  );
}
