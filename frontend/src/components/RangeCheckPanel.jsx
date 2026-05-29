import { useState } from "react";
import { predictRange } from "../lib/api.js";

const FIELDS = [
  { id: "soc_pct", label: "State of Charge (%)", min: 0, max: 100, step: 1, default: "80" },
  { id: "distance_km", label: "Distance (km)", min: 0, max: 600, step: 1, default: "120" },
  { id: "payload_t", label: "Payload (t)", min: 0, max: 22, step: 0.5, default: "14" },
  { id: "speed_kph", label: "Speed (km/h)", min: 30, max: 90, step: 1, default: "75" },
  { id: "gradient_pct", label: "Gradient (%)", min: -6, max: 6, step: 0.5, default: "0" },
  { id: "temperature_c", label: "Temperature (°C)", min: -15, max: 40, step: 1, default: "12" },
];

const REQUIRED_LABELS = {
  soc_pct: "SOC %",
  distance_km: "Distance (km)",
  payload_t: "Payload (t)",
  speed_kph: "Speed (kph)",
  gradient_pct: "Gradient (%)",
  temperature_c: "Temperature (°C)",
};

const initialForm = {
  soc_pct: "80",
  distance_km: "120",
  payload_t: "14",
  speed_kph: "75",
  gradient_pct: "0",
  temperature_c: "12",
  wind_mps: "0",
};

/** Round a number to a fixed number of decimals, tolerating non-numbers. */
function fmt(value, decimals = 1) {
  if (value === null || value === undefined || typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

/** Read a numeric value from a raw string. Returns null when blank, NaN when invalid. */
function readNumber(raw) {
  const trimmed = String(raw).trim();
  if (trimmed === "") {
    return null;
  }
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : NaN;
}

/**
 * Collect form inputs into the /api/predict request body. Returns
 * { payload } on success or { error } when a required field is missing or
 * non-numeric.
 */
function collectPayload(form) {
  const payload = {};
  for (const id in REQUIRED_LABELS) {
    const value = readNumber(form[id]);
    if (value === null) {
      return { error: "Please fill in " + REQUIRED_LABELS[id] + "." };
    }
    if (Number.isNaN(value)) {
      return { error: REQUIRED_LABELS[id] + " must be a number." };
    }
    payload[id] = value;
  }

  // wind_mps is optional; include only when provided and valid.
  const wind = readNumber(form.wind_mps);
  if (wind !== null) {
    if (Number.isNaN(wind)) {
      return { error: "Wind (m/s) must be a number." };
    }
    payload.wind_mps = wind;
  }

  return { payload };
}

function MetricTile({ label, value, unit }) {
  return (
    <div className="rounded-lg p-3 bg-surface-low">
      <div className="text-xs uppercase tracking-wide text-on-surface-variant">{label}</div>
      <div className="text-lg font-semibold mt-1 text-on-surface">
        {value}
        {unit ? <span className="text-sm font-normal text-on-surface-variant"> {unit}</span> : null}
      </div>
    </div>
  );
}

function Verdict({ data }) {
  const reaches = Boolean(data.reaches);
  const icon = reaches ? "check_circle" : "cancel";
  const title = reaches ? "REACHES" : "WILL NOT REACH";
  const accentClass = reaches ? "text-accent" : "text-error";
  const borderClass = reaches ? "border-accent" : "border-error";
  const marginVal = typeof data.margin_kwh === "number" ? data.margin_kwh : null;
  const marginSign = marginVal !== null && marginVal >= 0 ? "+" : "";

  return (
    <div className={"rounded-xl border-2 p-5 bg-surface-low " + borderClass}>
      <div className="flex items-center gap-3">
        <span className={"material-symbols-outlined " + accentClass} style={{ fontSize: "32px" }}>
          {icon}
        </span>
        <div>
          <div className={"text-xl font-bold font-headline " + accentClass}>{title}</div>
          <div className="text-sm text-on-surface-variant">
            Margin {marginSign + fmt(marginVal, 1)} kWh
          </div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-3 mt-4">
        <MetricTile label="Energy needed" value={fmt(data.energy_needed_kwh, 1)} unit="kWh" />
        <MetricTile label="Margin" value={marginSign + fmt(marginVal, 1)} unit="kWh" />
        <MetricTile label="Remaining SOC" value={fmt(data.remaining_soc_pct, 1)} unit="%" />
        <MetricTile label="Remaining range" value={fmt(data.remaining_range_km, 0)} unit="km" />
      </div>
      {data.confidence_note ? (
        <div className="text-xs mt-4 flex items-start gap-1.5 text-on-surface-variant">
          <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
            info
          </span>
          <span>{data.confidence_note}</span>
        </div>
      ) : null}
    </div>
  );
}

export default function RangeCheckPanel() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  function handleChange(event) {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    const collected = collectPayload(form);
    if (collected.error) {
      setResult(null);
      setError(collected.error);
      return;
    }

    setError(null);
    setResult(null);
    setLoading(true);

    try {
      const data = await predictRange(collected.payload);
      setResult(data);
    } catch (err) {
      let msg = err && err.message ? err.message : String(err);
      // Network/CORS failures surface as a generic TypeError from fetch.
      if (err instanceof TypeError) {
        msg = "Could not reach the server. Is the API running on this host?";
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      id="range-check"
      className="bg-surface-lowest rounded-2xl border-2 border-primary/30 overflow-hidden flex flex-col"
    >
      <div className="px-5 py-4 bg-gradient-to-r from-primary to-accent text-on-primary">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined">battery_charging_full</span>
          <h2 className="font-headline font-semibold text-lg">Live Range Check</h2>
        </div>
        <p className="text-xs text-on-primary/85 mt-0.5">Real-time ML reachability for an eActros 600</p>
      </div>

      <div className="p-5 flex-1">
        <form id="range-form" className="space-y-4" onSubmit={handleSubmit}>
          <div className="grid grid-cols-2 gap-3">
            {FIELDS.map((field) => (
              <div key={field.id}>
                <label htmlFor={field.id} className="block text-xs font-medium text-on-surface-variant mb-1">
                  {field.label}
                </label>
                <input
                  id={field.id}
                  name={field.id}
                  type="number"
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={form[field.id]}
                  onChange={handleChange}
                  className="w-full px-3 py-2.5 rounded-xl bg-surface-low border border-outline-variant/60 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
                />
              </div>
            ))}
            {/* Wind (optional) */}
            <div className="col-span-2">
              <label htmlFor="wind_mps" className="block text-xs font-medium text-on-surface-variant mb-1">
                Headwind (m/s) <span className="text-on-surface-variant/60">· optional</span>
              </label>
              <input
                id="wind_mps"
                name="wind_mps"
                type="number"
                min="0"
                max="12"
                step="0.5"
                value={form.wind_mps}
                onChange={handleChange}
                className="w-full px-3 py-2.5 rounded-xl bg-surface-low border border-outline-variant/60 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-primary text-on-primary font-medium hover:bg-primary/90 active:scale-[0.99] transition focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-70"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
              bolt
            </span>
            Check Reachability
          </button>
        </form>

        {/* Results area */}
        <div id="range-result" className="mt-5">
          {loading ? (
            <div className="rounded-xl border border-outline-variant bg-surface-low text-on-surface-variant p-4 text-sm">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined animate-spin" style={{ fontSize: "20px" }}>
                  progress_activity
                </span>
                <span>Running range check…</span>
              </div>
            </div>
          ) : error ? (
            <div className="rounded-xl border border-error bg-error/5 text-error p-4">
              <div className="flex items-start gap-2">
                <span className="material-symbols-outlined" style={{ fontSize: "22px" }}>
                  error
                </span>
                <div>
                  <div className="font-semibold">Range check failed</div>
                  <div className="text-sm mt-1 text-on-surface-variant">{error}</div>
                </div>
              </div>
            </div>
          ) : result ? (
            <Verdict data={result} />
          ) : null}
        </div>
      </div>
    </div>
  );
}
