import { useEffect, useState } from "react";

/* KpiStrip — a slim row of REAL operations KPIs for the dispatcher console.
 *
 * Every tile is backed by live API output, never a vanity constant:
 *   - Active trucks      -> /api/fleet counts (in_transit + available)
 *   - Trucks at-risk     -> /api/fleet counts.atRisk (real model verdicts)
 *   - Model MAE (kWh)    -> /api/model-info mae_kwh (trained model metric)
 *   - Range error        -> /api/model-info pct_range_error (% of a full charge)
 *
 * Fail-soft: a failed/slow fetch shows "—" rather than crashing the console.
 */

function Tile({ icon, label, value, unit, tone = "default", loading }) {
  const toneClass =
    tone === "risk"
      ? "text-error"
      : tone === "good"
      ? "text-primary"
      : "text-on-surface";
  const iconBg =
    tone === "risk"
      ? "bg-error/10 text-error"
      : tone === "good"
      ? "bg-primary/10 text-primary"
      : "bg-secondary/10 text-secondary";

  return (
    <div className="flex items-center gap-3 bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm px-4 py-3.5">
      <div className={"flex items-center justify-center w-10 h-10 rounded-xl shrink-0 " + iconBg}>
        <span className="material-symbols-outlined" style={{ fontSize: "22px" }}>
          {icon}
        </span>
      </div>
      <div className="min-w-0">
        <p className="text-[11px] uppercase tracking-wide text-on-surface-variant truncate">{label}</p>
        <p className={"font-headline font-bold text-xl leading-tight " + toneClass}>
          {loading ? (
            <span className="inline-block w-10 h-5 rounded bg-surface-low animate-pulse align-middle" />
          ) : (
            <>
              {value}
              {unit ? <span className="text-sm font-medium text-on-surface-variant"> {unit}</span> : null}
            </>
          )}
        </p>
      </div>
    </div>
  );
}

function fmt(value, decimals = 1) {
  if (value === null || value === undefined || typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

export default function KpiStrip() {
  const [counts, setCounts] = useState(null);
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const [fleetRes, modelRes] = await Promise.all([
          fetch("/api/fleet"),
          fetch("/api/model-info"),
        ]);
        const fleetBody = fleetRes.ok ? await fleetRes.json() : null;
        const modelBody = modelRes.ok ? await modelRes.json() : null;
        if (!active) return;
        setCounts(fleetBody && fleetBody.counts ? fleetBody.counts : null);
        setModel(modelBody && typeof modelBody === "object" ? modelBody : null);
      } catch {
        // Network/CORS failure — fail soft, tiles render em-dashes.
        if (active) {
          setCounts(null);
          setModel(null);
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  const active =
    counts && Number.isFinite(counts.inTransit) && Number.isFinite(counts.available)
      ? counts.inTransit + counts.available
      : null;
  const atRisk = counts && Number.isFinite(counts.atRisk) ? counts.atRisk : null;
  const mae = model ? model.mae_kwh : null;
  const rangeErr = model ? model.pct_range_error : null;

  return (
    <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      <Tile
        icon="local_shipping"
        label="Active trucks"
        value={active === null ? "—" : active}
        unit="driving"
        tone="good"
        loading={loading}
      />
      <Tile
        icon="warning"
        label="Trucks at-risk"
        value={atRisk === null ? "—" : atRisk}
        unit="leg"
        tone={atRisk && atRisk > 0 ? "risk" : "default"}
        loading={loading}
      />
      <Tile
        icon="target"
        label="Model MAE"
        value={fmt(mae, 1)}
        unit="kWh"
        loading={loading}
      />
      <Tile
        icon="percent"
        label="Range error"
        value={fmt(rangeErr, 2)}
        unit="% of charge"
        loading={loading}
      />
    </section>
  );
}
