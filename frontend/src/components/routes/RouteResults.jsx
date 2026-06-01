// Route RESULT view (light theme). Renders: SOC gauge, route-info cards,
// driver-hours bars + EU561 badge, trip timeline, energy overview.
import SocGauge from "./SocGauge.jsx";
import TripTimeline from "./TripTimeline.jsx";
import ChargingStopsList from "./ChargingStopsList.jsx";

function InfoCard({ icon, value, label, tint = "#006d32" }) {
  return (
    <div className="rounded-xl bg-surface-low border border-outline-variant/50 px-3 py-2.5">
      <span className="material-symbols-outlined" style={{ fontSize: "18px", color: tint }}>
        {icon}
      </span>
      <p className="mt-1 text-xl font-headline font-bold text-on-surface leading-none tabular-nums">{value}</p>
      <p className="text-[11px] text-on-surface-variant mt-1">{label}</p>
    </div>
  );
}

function ProgressBar({ value, max, tint }) {
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const over = value > max;
  const color = over ? "#ba1a1a" : tint;
  return (
    <div className="h-2 rounded-full bg-surface overflow-hidden">
      <div className="h-full rounded-full transition-all" style={{ width: `${frac * 100}%`, background: color }} />
    </div>
  );
}

// Format a duration given in MINUTES as a readable "Xh Ym" (e.g. 195 -> "3h 15m",
// 636 -> "10h 36m", 45 -> "45m"). Used for both total trip time and charging time.
function fmtHm(min) {
  if (min == null || !Number.isFinite(min)) return "0m";
  const total = Math.round(min);
  const h = Math.floor(total / 60);
  const m = total % 60;
  if (h && m) return `${h}h ${m}m`;
  return h ? `${h}h` : `${m}m`;
}

// Hours as a compact decimal (e.g. 7.4 -> "7.4", 9 -> "9") — used by the
// driver-hours bars, which read against whole-hour limits (9 h / 56 h).
function fmtH(h) {
  if (h == null) return "0";
  return Number.isInteger(h) ? String(h) : h.toFixed(1);
}

export default function RouteResults({ plan }) {
  if (!plan || !plan.summary) {
    return (
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-8 text-center">
        <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "36px" }}>
          route
        </span>
        <p className="text-sm text-on-surface-variant mt-2">No route result yet.</p>
        <p className="text-xs text-on-surface-variant/70 mt-1">
          Set an origin and at least one destination, then Optimize Route.
        </p>
      </div>
    );
  }

  const s = plan.summary;
  const d = s.driver || {};
  const stops = plan.stops || [];
  // Show the per-stop panel only when it adds information (multi-stop, or any
  // stop carries delivery data) — a lone final stop with no data is noise.
  const showStops =
    stops.length > 1 ||
    stops.some((st) => st.dropWeightKg > 0 || st.deliverBy || st.unloadMin > 0);
  const assumptions = s.assumptions || [];

  return (
    <div className="space-y-5">
      {/* SOC gauge + badges */}
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
        <div className="flex flex-col items-center">
          <SocGauge arrivalSoc={s.arrivalSoc} startSoc={s.startSoc} minSoc={s.minSoc} />
          <div className="flex items-center gap-2 mt-3">
            <span className="px-2.5 py-1 rounded-full bg-surface-low text-on-surface-variant text-xs font-medium ring-1 ring-outline-variant/60">
              eActros 600
            </span>
            <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-primary/10 text-primary text-xs font-medium ring-1 ring-primary/25">
              <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
                schedule
              </span>
              ETA {s.etaLabel || "--:--"}
            </span>
          </div>
        </div>

        {/* Route info cards */}
        <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant mb-2 mt-5">Route Info</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <InfoCard icon="ev_station" value={s.chargingStops ?? 0} label="Charging Stops" tint="#f59e0b" />
          <InfoCard icon="straighten" value={`${Math.round(s.distanceKm || 0)} km`} label="Total Distance" tint="#0059bb" />
          <InfoCard icon="schedule" value={fmtHm((s.totalTimeH || 0) * 60)} label="Total Time" />
          <InfoCard icon="bolt" value={fmtHm(s.chargingTimeMin || 0)} label="Charging Time" tint="#f59e0b" />
        </div>

        {/* Live traffic — TomTom routes around closures/congestion (fastest +
            traffic), so this delay is already baked into the ETA above. */}
        {(() => {
          const t = plan.traffic || {};
          const delayMin = Math.round((t.delayS || 0) / 60);
          const nInc = (t.incidents || []).length;
          if (delayMin <= 0 && nInc === 0) return null;
          return (
            <div className="mt-3 flex items-center gap-2 rounded-xl bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-900">
              <span className="material-symbols-outlined" style={{ fontSize: "18px", color: "#d97706" }}>
                traffic
              </span>
              <span>
                <span className="font-semibold">Live traffic:</span>{" "}
                {delayMin > 0 ? `+${delayMin} min delay` : "no significant delay"}
                {nInc > 0 ? ` · ${nInc} incident${nInc > 1 ? "s" : ""} on route` : ""}
                <span className="text-amber-700"> — already factored into ETA</span>
              </span>
            </div>
          );
        })()}
      </div>

      {/* Per-stop arrivals (per-leg simulation): arrival SOC, ETA, deliver-by */}
      {showStops && (
        <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
          <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant mb-3">
            Delivery Stops
          </h3>
          <div className="space-y-2">
            {stops.map((st) => (
              <div
                key={st.index}
                className="flex items-center gap-3 rounded-xl bg-surface-low border border-outline-variant/50 px-3 py-2"
              >
                <span className="flex items-center justify-center w-5 h-5 shrink-0 rounded-full bg-primary/15 text-primary text-[11px] font-semibold ring-1 ring-primary/30">
                  {st.index + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-on-surface truncate">
                    {st.label}
                    {st.isFinal ? " · destination" : ""}
                  </p>
                  <p className="text-[11px] text-on-surface-variant tabular-nums">
                    {Math.round(st.distKm)} km · ETA {st.etaLabel} · arrive {Math.round(st.arriveSoc)}% SOC
                    {st.dropWeightKg > 0 ? ` · drop ${(st.dropWeightKg / 1000).toFixed(1)} t` : ""}
                  </p>
                </div>
                {st.deliverBy && (
                  <span
                    className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
                      st.onTime
                        ? "bg-primary/10 text-primary ring-1 ring-primary/25"
                        : "bg-error/10 text-error ring-1 ring-error/25"
                    }`}
                    title={`Deliver by ${st.deliverBy}`}
                  >
                    <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>
                      {st.onTime ? "schedule" : "running_with_errors"}
                    </span>
                    {st.onTime ? "On time" : "Late"}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Driver hours */}
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant">Driver Hours</h3>
          {d.eu561ok == null ? (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-surface text-on-surface-variant ring-1 ring-outline-variant/50">
              <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>help</span>
              EU 561 — not evaluated offline
            </span>
          ) : (
            <span
              className={`flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium ${
                d.eu561ok
                  ? "bg-primary/10 text-primary ring-1 ring-primary/25"
                  : "bg-error/10 text-error ring-1 ring-error/25"
              }`}
            >
              <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>
                {d.eu561ok ? "verified" : "warning"}
              </span>
              EU 561 {d.eu561ok ? "Compliant" : "Violation"}
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-xl font-headline font-bold text-on-surface tabular-nums">{fmtH(d.drivingH)}h</p>
            <p className="text-[10px] text-on-surface-variant">Driving</p>
          </div>
          <div>
            <p className="text-xl font-headline font-bold text-on-surface tabular-nums">{d.breaks ?? 0}</p>
            <p className="text-[10px] text-on-surface-variant">Breaks</p>
          </div>
          <div>
            <p className="text-xl font-headline font-bold text-on-surface tabular-nums">{fmtH(d.totalH)}h</p>
            <p className="text-[10px] text-on-surface-variant">Total</p>
          </div>
        </div>

        <div className="space-y-2">
          <div>
            <div className="flex justify-between text-[11px] text-on-surface-variant mb-1">
              <span>Daily</span>
              <span className="tabular-nums">{fmtH(d.dailyH)} / {d.dailyMaxH ?? 9}h</span>
            </div>
            <ProgressBar value={d.dailyH ?? 0} max={d.dailyMaxH ?? 9} tint="#00d166" />
          </div>
          <div>
            <div className="flex justify-between text-[11px] text-on-surface-variant mb-1">
              <span>Weekly</span>
              <span className="tabular-nums">{fmtH(d.weeklyH)} / {d.weeklyMaxH ?? 56}h</span>
            </div>
            <ProgressBar value={d.weeklyH ?? 0} max={d.weeklyMaxH ?? 56} tint="#006d32" />
          </div>
        </div>

        {d.days > 1 && Array.isArray(d.perDay) && (
          <div className="pt-2 border-t border-outline-variant/30">
            <p className="text-[10px] uppercase tracking-wide text-on-surface-variant mb-1">
              {d.days}-day schedule · 11 h rest between days
            </p>
            <div className="space-y-0.5">
              {d.perDay.map((p, i) => (
                <div key={i} className="flex justify-between text-[11px] text-on-surface-variant tabular-nums">
                  <span>Day {p.day}{p.dateLabel ? ` · ${p.dateLabel}` : ""}</span>
                  <span>
                    {fmtH(p.drivingH)}h driving
                    {p.breaks ? ` · ${p.breaks} break${p.breaks > 1 ? "s" : ""}` : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Trip timeline */}
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
        <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant mb-3">Trip Timeline</h3>
        <TripTimeline segments={plan.segments} />
      </div>

      {/* Charging stops — real TomTom stations: connectors, power, availability */}
      <ChargingStopsList stops={plan.chargingStops} />

      {/* Energy overview */}
      <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-5">
        <h3 className="text-[11px] uppercase tracking-wide font-medium text-on-surface-variant mb-3">Energy Overview</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          <InfoCard icon="battery_charging_full" value={`${Math.round(s.energyKwh || 0)}`} label="Total kWh" />
          <InfoCard icon="speed" value={`${Math.round(s.kwhPer100 || 0)}`} label="kWh / 100km" tint="#0059bb" />
          <InfoCard icon="trending_up" value={`${Math.round(s.elevationGainM || 0)} m`} label="Elevation Gain" tint="#f59e0b" />
          <InfoCard icon="ev_station" value={`${s.chargingStops || 0}`} label="Charging Stops" />
        </div>
        <p className="mt-2 text-[10px] leading-tight text-on-surface-variant/70">
          Total kWh is field-calibrated to real-world consumption; charging and range
          margin use a higher conservative estimate (see modelling assumptions below).
        </p>
      </div>

      {/* Honest modelling assumptions — surfaced from the backend so the
          dispatcher sees the caveats, not just a confident number. */}
      {assumptions.length > 0 && (
        <details className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm p-4 text-sm text-on-surface-variant">
          <summary className="cursor-pointer font-medium text-on-surface flex items-center gap-1.5">
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              info
            </span>
            Modelling assumptions &amp; limitations
          </summary>
          <ul className="mt-2 list-disc pl-5 space-y-1">
            {assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
