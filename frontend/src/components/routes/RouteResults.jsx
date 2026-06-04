// Route RESULT view (light theme). Renders: SOC gauge, route-info cards,
// driver-hours bars + EU561 badge, trip timeline, energy overview.
import SocGauge from "./SocGauge.jsx";
import TripTimeline from "./TripTimeline.jsx";
import ChargingStopsList from "./ChargingStopsList.jsx";
import { to12h } from "../../lib/time.js";

function InfoCard({ icon, value, label, tint = "#006d32" }) {
  return (
    <div className="group rounded-control nx-card-inset px-3 py-2.5 transition-colors duration-snappy ease-nx-out hover:border-outline-variant/60">
      <span
        className="material-symbols-outlined transition-transform duration-snappy ease-nx-out group-hover:scale-110"
        style={{ fontSize: "18px", color: tint }}
      >
        {icon}
      </span>
      <p className="ck-num mt-1.5 text-xl font-bold text-on-surface leading-none">{value}</p>
      <p className="ck-label text-[9px] text-on-surface-variant mt-1.5">{label}</p>
    </div>
  );
}

function ProgressBar({ value, max, tint }) {
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const over = value > max;
  const color = over ? "#ba1a1a" : tint;
  return (
    <div className="h-2 rounded-pill bg-surface overflow-hidden ring-1 ring-inset ring-outline-variant/30">
      <div
        className="h-full rounded-pill transition-[width] duration-smooth ease-nx-out"
        style={{ width: `${frac * 100}%`, background: color }}
      />
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
      <div className="nx-card p-10 text-center">
        <span className="inline-flex items-center justify-center w-14 h-14 rounded-pill bg-surface-low text-on-surface-variant ring-1 ring-outline-variant/40">
          <span className="material-symbols-outlined" style={{ fontSize: "30px" }}>
            route
          </span>
        </span>
        <p className="text-sm font-medium text-on-surface mt-3">No route result yet.</p>
        <p className="text-xs text-on-surface-variant/80 mt-1">
          Set an origin and at least one destination, then Optimize Route.
        </p>
      </div>
    );
  }

  const s = plan.summary;
  const d = s.driver || {};
  const stops = plan.stops || [];
  // Average DRIVING speed (excludes breaks/charging) — the truck's effective
  // cruising speed, comparable to the figure NexOS shows (~75 km/h).
  const driveH = s.drivingTimeH ?? d.drivingH ?? 0;
  const avgSpeed = driveH > 0 ? (s.distanceKm || 0) / driveH : 0;
  // Show the per-stop panel only when it adds information (multi-stop, or any
  // stop carries delivery data) — a lone final stop with no data is noise.
  const showStops =
    stops.length > 1 ||
    stops.some((st) => st.dropWeightKg > 0 || st.deliverBy || st.unloadMin > 0);
  const assumptions = s.assumptions || [];
  // Live-traffic totals — TomTom routes around congestion (fastest + traffic),
  // so this delay is already baked into the ETA. Surfaced both as a dashboard
  // readout and the explanatory banner below it.
  const traffic = plan.traffic || {};
  const delayMin = Math.round((traffic.delayS || 0) / 60);
  const nInc = (traffic.incidents || []).length;

  return (
    <div className="space-y-5">
      {/* Instrument dashboard — SOC speedometer + the key energy/route readouts,
          up top so it's the first thing the dispatcher sees. */}
      <div className="nx-card p-6">
        <p className="ck-label text-[10px] text-on-surface-variant/70 mb-4">Energy · Route Dashboard</p>
        <div className="flex flex-col lg:flex-row lg:items-center gap-6">
          {/* Speedometer */}
          <div className="flex flex-col items-center shrink-0">
            <SocGauge arrivalSoc={s.arrivalSoc} startSoc={s.startSoc} minSoc={s.minSocFloor ?? s.minSoc} />
            <div className="flex flex-wrap items-center justify-center gap-2 mt-4">
              <span className="px-2.5 py-1 rounded-pill bg-surface text-on-surface-variant text-xs font-medium ring-1 ring-outline-variant/50">
                eActros 600
              </span>
              <span className="flex items-center gap-1.5 px-2.5 py-1 rounded-pill bg-primary/10 text-primary text-xs font-semibold ring-1 ring-primary/25">
                <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>schedule</span>
                ETA {s.etaLabel ? to12h(s.etaLabel) : "--:--"}
              </span>
            </div>
          </div>
          {/* Readout cluster */}
          <div className="flex-1 w-full grid grid-cols-2 sm:grid-cols-4 gap-2">
            <InfoCard icon="battery_charging_full" value={`${Math.round(s.energyKwh || 0)}`} label="Total kWh" />
            <InfoCard icon="speed" value={`${Math.round(s.kwhPer100 || 0)}`} label="kWh / 100km" tint="#0059bb" />
            <InfoCard icon="straighten" value={`${Math.round(s.distanceKm || 0)} km`} label="Distance" tint="#0059bb" />
            <InfoCard icon="schedule" value={fmtHm((s.totalTimeH || 0) * 60)} label="Total Time" />
            <InfoCard icon="ev_station" value={`${s.chargingStops ?? 0}`} label="Charging Stops" tint="#f59e0b" />
            <InfoCard icon="bolt" value={fmtHm(s.chargingTimeMin || 0)} label="Charging Time" tint="#f59e0b" />
            <InfoCard icon="trending_up" value={`${Math.round(s.elevationGainM || 0)} m`} label="Elevation" tint="#f59e0b" />
            <InfoCard
              icon="traffic"
              value={delayMin > 0 ? `+${delayMin} min` : "0 min"}
              label={`Traffic${nInc ? ` · ${nInc} inc` : ""}`}
              tint="#d97706"
            />
          </div>
        </div>
        <p className="mt-3 text-[10px] leading-tight text-on-surface-variant/70">
          Total kWh is field-calibrated to real-world consumption; charging and range
          margin use a higher conservative estimate (see modelling assumptions below).
        </p>

        {/* Live traffic — already baked into the ETA; the banner explains the readout above. */}
        {(delayMin > 0 || nInc > 0) && (
          <div className="mt-4 flex items-center gap-2.5 rounded-control border px-3 py-2.5 text-sm" style={{ background: "#f59e0b14", borderColor: "#f59e0b40", color: "#b45309" }}>
            <span className="material-symbols-outlined shrink-0" style={{ fontSize: "18px", color: "#d97706" }}>
              traffic
            </span>
            <span className="text-on-surface">
              <span className="font-semibold" style={{ color: "#d97706" }}>Live traffic:</span>{" "}
              {delayMin > 0 ? `+${delayMin} min total delay` : "no significant delay"}
              {nInc > 0 ? ` · ${nInc} incident${nInc > 1 ? "s" : ""} on route` : ""}
              <span className="text-on-surface-variant"> — already factored into ETA</span>
            </span>
          </div>
        )}
      </div>

      {/* Per-stop arrivals (per-leg simulation): arrival SOC, ETA, deliver-by */}
      {showStops && (
        <div className="nx-card p-5">
          <h3 className="text-[11px] uppercase tracking-[0.1em] font-semibold text-on-surface-variant mb-3">
            Delivery Stops
          </h3>
          <div className="space-y-2">
            {stops.map((st) => (
              <div
                key={st.index}
                className="flex items-center gap-3 rounded-control nx-card-inset px-3 py-2.5 transition-colors duration-snappy ease-nx-out hover:border-outline-variant/60"
              >
                <span className="flex items-center justify-center w-6 h-6 shrink-0 rounded-pill bg-primary/15 text-primary text-[11px] font-semibold ring-1 ring-primary/30">
                  {st.index + 1}
                </span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-on-surface truncate">
                    {st.label}
                    {st.isFinal ? " · destination" : ""}
                  </p>
                  <p className="text-[11px] text-on-surface-variant tabular-nums">
                    {Math.round(st.distKm)} km · ETA {to12h(st.etaLabel)} · arrive {Math.round(st.arriveSoc)}% SOC
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
      <div className="nx-card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-[11px] uppercase tracking-[0.1em] font-semibold text-on-surface-variant">Driver Hours</h3>
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

        <div className="grid grid-cols-3 gap-2 text-center nx-card-inset px-2 py-3">
          <div className="border-r border-outline-variant/30">
            <p className="ck-num text-xl font-bold text-on-surface">{fmtH(d.drivingH)}h</p>
            <p className="ck-label text-[9px] text-on-surface-variant mt-1">Driving</p>
          </div>
          <div className="border-r border-outline-variant/30">
            <p className="ck-num text-xl font-bold text-on-surface">{d.breaks ?? 0}</p>
            <p className="ck-label text-[9px] text-on-surface-variant mt-1">Breaks</p>
          </div>
          <div>
            <p className="ck-num text-xl font-bold text-on-surface">{fmtH(d.totalH)}h</p>
            <p className="ck-label text-[9px] text-on-surface-variant mt-1">Total</p>
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
          <div className="pt-3 nx-divider">
            <p className="text-[10px] uppercase tracking-wide text-on-surface-variant mb-1 mt-3">
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
      <div className="nx-card p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-[11px] uppercase tracking-[0.1em] font-semibold text-on-surface-variant">Trip Timeline</h3>
          {avgSpeed > 0 && (
            <span
              className="inline-flex items-center gap-1 rounded-pill px-2 py-0.5 text-[11px] font-semibold tabular-nums bg-secondary/10 text-secondary ring-1 ring-secondary/20"
              title="Average driving speed (excludes breaks &amp; charging)"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>speed</span>
              Ø {Math.round(avgSpeed)} km/h
            </span>
          )}
        </div>
        <TripTimeline segments={plan.segments} />
      </div>

      {/* Charging stops — real TomTom stations: connectors, power, availability */}
      <ChargingStopsList stops={plan.chargingStops} />

      {/* Honest modelling assumptions — surfaced from the backend so the
          dispatcher sees the caveats, not just a confident number. */}
      {assumptions.length > 0 && (
        <details className="nx-card p-4 text-sm text-on-surface-variant group">
          <summary className="cursor-pointer font-medium text-on-surface flex items-center gap-1.5 nx-focus rounded-control -m-1 p-1">
            <span className="material-symbols-outlined text-on-surface-variant transition-transform duration-snappy ease-nx-out group-open:rotate-90" style={{ fontSize: "16px" }}>
              chevron_right
            </span>
            Modelling assumptions &amp; limitations
          </summary>
          <ul className="mt-3 list-disc pl-5 space-y-1.5 text-on-surface-variant">
            {assumptions.map((a, i) => (
              <li key={i}>{a}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}
