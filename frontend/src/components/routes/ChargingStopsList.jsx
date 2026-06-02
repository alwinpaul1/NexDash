// Compact "Charging Stops" list (light theme). Shows each REAL TomTom station's
// name, connectors, max power, live availability, kWh added, and estimated
// charge time. Rendered in the Routes results column; hidden when no stops.

// Friendly charge-duration label, e.g. 71 -> "71 min", 132 -> "2 h 12 min".
function fmtChargeTime(min) {
  if (!Number.isFinite(min) || min <= 0) return "";
  if (min < 90) return `${Math.round(min)} min`;
  const h = Math.floor(min / 60);
  const m = Math.round(min % 60);
  return m ? `${h} h ${m} min` : `${h} h`;
}

export default function ChargingStopsList({ stops = [] }) {
  const list = (stops || []).filter(Boolean);
  if (list.length === 0) return null;

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
      <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant mb-3">
        Charging Stops
      </h3>
      <ul className="space-y-2.5">
        {list.map((s, i) => (
          <li
            key={Number.isFinite(s?.lat) && Number.isFinite(s?.lng) ? `cs-${s.lat},${s.lng}` : `cs-${i}`}
            className="flex items-start gap-3 rounded-xl bg-surface-low border border-outline-variant/50 px-3 py-2.5"
          >
            <span
              className="flex shrink-0 items-center justify-center rounded-full text-white text-xs font-semibold"
              style={{ width: 24, height: 24, background: "#f59e0b" }}
            >
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-sm font-semibold text-on-surface truncate">
                {s.name || `Charging Stop ${i + 1}`}
              </p>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-on-surface-variant">
                {s.maxPowerKw ? (
                  <span className="inline-flex items-center gap-1">
                    <span className="material-symbols-outlined" style={{ fontSize: "13px", color: "#f59e0b" }}>
                      bolt
                    </span>
                    {s.maxPowerKw} kW
                  </span>
                ) : null}
                {s.connectors && s.connectors.length ? (
                  <span>{s.connectors.map((c) => c.label).join(", ")}</span>
                ) : null}
                {s.address ? <span className="truncate">· {s.address}</span> : null}
              </div>
              <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px]">
                {s.availability ? (
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ${
                      s.availability.available > 0
                        ? "bg-primary/10 text-primary"
                        : "bg-surface text-on-surface-variant ring-1 ring-outline-variant/60"
                    }`}
                  >
                    {s.availability.available} of {s.availability.total} free
                  </span>
                ) : null}
                {Number.isFinite(s.kWh) ? (
                  <span className="inline-flex items-center gap-1 rounded-full bg-surface text-on-surface-variant px-2 py-0.5 ring-1 ring-outline-variant/60">
                    +{Math.round(s.kWh)} kWh
                    {Number.isFinite(s.chargeMinutes) ? ` · ~${fmtChargeTime(s.chargeMinutes)}` : ""}
                  </span>
                ) : null}
                {Number.isFinite(s.costEur) && s.costEur > 0 ? (
                  <span
                    className="inline-flex items-center gap-1 rounded-full bg-surface text-on-surface-variant px-2 py-0.5 ring-1 ring-outline-variant/60"
                    title={
                      Number.isFinite(s.pricePerKwh)
                        ? `Estimated at €${s.pricePerKwh.toFixed(2)}/kWh`
                        : "Estimated at a flat €0.45/kWh (no live tariff for this site)"
                    }
                  >
                    ≈ €{Math.round(s.costEur)}
                    {Number.isFinite(s.pricePerKwh) ? ` · €${s.pricePerKwh.toFixed(2)}/kWh` : ""}
                  </span>
                ) : null}
                {s.openingHours ? (
                  <span className="text-on-surface-variant/80">{s.openingHours}</span>
                ) : null}
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
