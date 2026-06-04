import { useState } from "react";

/* TruckCard — the eActros 600 with a pseudo-3D tilt and a specs popover.
 *
 * The truck image tilts toward the cursor (perspective + rotateX/rotateY) for a
 * "3D" feel. A true 360° spin would need an actual 3D model (.glb) rendered with
 * three.js / <model-viewer>; with a single side-view PNG we do an interactive
 * tilt. Drop a .glb in and this can be swapped for real 3D.
 */

const SPECS = [
  { icon: "battery_full", label: "Battery", value: "621 kWh (600 usable)" },
  { icon: "bolt", label: "Max Charging", value: "400 kW (CCS)" },
  { icon: "package_2", label: "Max Payload", value: "22 t (22,000 kg)" },
  { icon: "scale", label: "GCW (laden)", value: "40 t (40,000 kg)" },
  { icon: "straighten", label: "L × W × H", value: "16.5 × 2.55 × 4.0 m" },
  { icon: "trip_origin", label: "Axles", value: "5 (artic)" },
];

export default function TruckCard() {
  const [showSpecs, setShowSpecs] = useState(false);

  return (
    <div className="nx-card nx-hover-lift relative p-4">
      {/* Framed render box — truck on a light "render bay" with the signature
          44px HUD grid backdrop and a lime CCS power tag. */}
      <div className="relative overflow-hidden rounded-control border border-outline-variant/50 bg-gradient-to-br from-white to-slate-200 p-2">
        <div aria-hidden="true" className="pointer-events-none absolute inset-0 ck-grid-bg" />
        <span className="absolute top-2 right-2 z-10 ck-label text-[10px] font-semibold px-2.5 py-1 rounded-pill bg-ck-lime/15 text-ck-lime-ink ring-1 ring-ck-lime/40">
          400 kW CCS
        </span>
        <img
          src="/eactros-600.png"
          alt="Mercedes-Benz eActros 600"
          className="relative w-full h-44 object-contain"
          style={{ mixBlendMode: "multiply" }}
        />
      </div>

      <div className="mt-3 flex items-center justify-between">
        <div className="min-w-0">
          <h3 className="font-headline font-bold text-xl tracking-tight text-on-surface">eActros 600</h3>
          <p className="ck-num text-[11px] text-on-surface-variant mt-0.5">621 kWh · 500 km · 22 t</p>
        </div>
        <button
          type="button"
          onClick={() => setShowSpecs((s) => !s)}
          aria-label="Vehicle specifications"
          aria-pressed={showSpecs}
          className={`w-9 h-9 shrink-0 rounded-full border flex items-center justify-center transition-all duration-snappy ease-nx-out active:scale-95 ${
            showSpecs
              ? "border-primary/40 bg-primary/10 text-primary"
              : "border-outline-variant/50 bg-surface-lowest text-on-surface-variant hover:text-primary hover:border-primary/30 hover:bg-primary/10"
          }`}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            info
          </span>
        </button>
      </div>

      {showSpecs && (
        <div className="mt-3 rounded-control border border-outline-variant/50 bg-surface-low/50 divide-y divide-outline-variant/30 overflow-hidden">
          {SPECS.map((s) => (
            <div
              key={s.label}
              className="flex items-center justify-between gap-3 px-3 py-2 text-sm transition-colors duration-snappy hover:bg-surface-low"
            >
              <span className="flex items-center gap-2 text-on-surface-variant">
                <span className="material-symbols-outlined text-on-surface-variant/80" style={{ fontSize: "18px" }}>
                  {s.icon}
                </span>
                {s.label}
              </span>
              <span className="font-semibold tabular-nums text-on-surface text-right">{s.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
