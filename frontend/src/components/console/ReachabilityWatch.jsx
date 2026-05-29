import { useEffect, useState } from "react";

/* ReachabilityWatch — the dispatcher's answer board.
 *
 * For every truck in /api/fleet it shows the one thing dispatch cares about:
 * "will this truck make it to its next stop, and how much charge left?".
 * SOC, next-stop label + distance, the reachable/at-risk verdict and the
 * energy margin all come straight from the real model (via the fleet helper),
 * not from invented numbers.
 *
 * Driving trucks (in_transit / available) carry a model verdict. Parked trucks
 * (charging / maintenance) have no leg in flight, so they show a neutral
 * "parked" badge instead of a reachability verdict.
 *
 * Fail-soft: loading skeleton, then a clear error row if the fleet API is down.
 */

const STATUS_META = {
  in_transit: { label: "In transit", icon: "near_me", dot: "bg-accent" },
  available: { label: "Available", icon: "check_circle", dot: "bg-primary" },
  charging: { label: "Charging", icon: "bolt", dot: "bg-secondary" },
  maintenance: { label: "Maintenance", icon: "build", dot: "bg-on-surface-variant" },
};

function socTone(soc) {
  if (typeof soc !== "number") return "text-on-surface";
  if (soc < 20) return "text-error";
  if (soc < 40) return "text-secondary";
  return "text-on-surface";
}

function fmt(value, decimals = 0) {
  if (value === null || value === undefined || typeof value !== "number" || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

function Badge({ truck }) {
  // Parked trucks have no active leg -> neutral badge, no verdict.
  if (truck.status === "charging" || truck.status === "maintenance") {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-surface-low text-on-surface-variant border border-outline-variant/50">
        <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>pause_circle</span>
        Parked
      </span>
    );
  }

  // Model verdict unavailable (model artifact missing) -> unknown.
  if (truck.reachable === null || truck.reachable === undefined) {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-surface-low text-on-surface-variant border border-outline-variant/50">
        <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>help</span>
        No model
      </span>
    );
  }

  if (truck.atRisk) {
    const label = truck.reachable ? "At risk" : "Won't reach";
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-error/10 text-error border border-error/30">
        <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>warning</span>
        {label}
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold bg-accent/15 text-primary border border-accent/40">
      <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>check_circle</span>
      Reachable
    </span>
  );
}

function Row({ truck }) {
  const meta = STATUS_META[truck.status] || STATUS_META.available;
  const stop = truck.nextStop || {};
  const margin = truck.marginKwh;
  const hasMargin = typeof margin === "number" && Number.isFinite(margin);
  const marginSign = hasMargin && margin >= 0 ? "+" : "";
  const marginTone = !hasMargin ? "text-on-surface-variant" : margin < 0 ? "text-error" : truck.atRisk ? "text-secondary" : "text-primary";

  return (
    <div className="flex items-center gap-3 px-4 py-3 border-b border-outline-variant/30 last:border-0 hover:bg-surface-low/60 transition-colors">
      {/* Identity */}
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className={"w-2 h-2 rounded-full shrink-0 " + meta.dot} />
          <p className="font-medium text-sm text-on-surface truncate">{truck.name}</p>
        </div>
        <p className="text-xs text-on-surface-variant mt-0.5 flex items-center gap-1">
          <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>place</span>
          {stop.label || "—"} · {fmt(stop.distanceKm, 0)} km
        </p>
      </div>

      {/* SOC */}
      <div className="text-right w-14 shrink-0">
        <p className={"font-headline font-semibold text-sm " + socTone(truck.soc)}>{fmt(truck.soc, 0)}%</p>
        <p className="text-[10px] uppercase tracking-wide text-on-surface-variant">SOC</p>
      </div>

      {/* Margin (real model output) */}
      <div className="text-right w-20 shrink-0 hidden sm:block">
        <p className={"font-headline font-semibold text-sm " + marginTone}>
          {hasMargin ? marginSign + fmt(margin, 1) : "—"}
        </p>
        <p className="text-[10px] uppercase tracking-wide text-on-surface-variant">kWh margin</p>
      </div>

      {/* Verdict */}
      <div className="w-28 shrink-0 flex justify-end">
        <Badge truck={truck} />
      </div>
    </div>
  );
}

export default function ReachabilityWatch() {
  const [trucks, setTrucks] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const res = await fetch("/api/fleet");
        if (!res.ok) throw new Error("Server returned HTTP " + res.status + ".");
        const body = await res.json();
        if (!active) return;
        setTrucks(Array.isArray(body.trucks) ? body.trucks : []);
        setError(null);
      } catch (err) {
        if (!active) return;
        setError("Could not load the fleet. Is the API running on this host?");
      } finally {
        if (active) setLoading(false);
      }
    }

    load();
    return () => {
      active = false;
    };
  }, []);

  // Surface at-risk trucks first — that's what a dispatcher scans for.
  const ordered = [...trucks].sort((a, b) => {
    const ra = a.atRisk ? 1 : 0;
    const rb = b.atRisk ? 1 : 0;
    if (ra !== rb) return rb - ra;
    return (a.soc ?? 100) - (b.soc ?? 100);
  });

  const atRiskCount = trucks.filter((t) => t.atRisk).length;

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm overflow-hidden flex flex-col">
      <div className="flex items-center justify-between px-5 pt-5 pb-3">
        <div>
          <h2 className="font-headline font-semibold text-lg text-on-surface">Reachability Watch</h2>
          <p className="text-xs text-on-surface-variant">Will each truck make its next stop?</p>
        </div>
        {!loading && !error ? (
          <span
            className={
              "inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-semibold " +
              (atRiskCount > 0
                ? "bg-error/10 text-error border border-error/30"
                : "bg-accent/15 text-primary border border-accent/40")
            }
          >
            <span className="material-symbols-outlined" style={{ fontSize: "14px" }}>
              {atRiskCount > 0 ? "warning" : "verified"}
            </span>
            {atRiskCount} at-risk
          </span>
        ) : null}
      </div>

      <div className="flex-1 overflow-y-auto max-h-[26rem]">
        {loading ? (
          <div className="px-4 py-3 space-y-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-10 rounded-lg bg-surface-low animate-pulse" />
            ))}
          </div>
        ) : error ? (
          <div className="m-5 rounded-xl border border-error bg-error/5 text-error p-4">
            <div className="flex items-start gap-2">
              <span className="material-symbols-outlined" style={{ fontSize: "22px" }}>error</span>
              <div>
                <div className="font-semibold">Fleet unavailable</div>
                <div className="text-sm mt-1 text-on-surface-variant">{error}</div>
              </div>
            </div>
          </div>
        ) : ordered.length === 0 ? (
          <div className="m-5 text-sm text-on-surface-variant">No trucks in the roster.</div>
        ) : (
          ordered.map((truck) => <Row key={truck.id} truck={truck} />)
        )}
      </div>
    </div>
  );
}
