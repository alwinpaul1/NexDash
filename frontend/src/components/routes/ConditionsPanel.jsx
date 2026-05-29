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
    <div className="rounded-xl bg-surface-low border border-outline-variant/50 px-3 py-3">
      <span className="material-symbols-outlined" style={{ fontSize: "20px", color: tint }}>
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

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
      <div className="flex items-center gap-2 mb-3">
        <span className="material-symbols-outlined text-secondary" style={{ fontSize: "20px" }}>
          partly_cloudy_day
        </span>
        <h3 className="font-headline font-semibold text-lg text-on-surface">Route Conditions</h3>
        <span className="text-[11px] text-on-surface-variant ml-auto">via Open-Meteo</span>
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
    </div>
  );
}
