import { useState } from "react";

/* TruckCard — vehicle selector (dropdown) + specs popover.
 *
 * One vehicle for now (the eActros 600), presented as a dropdown so the user
 * is "choosing" the truck and more models can be added by extending VEHICLES.
 * The info button toggles the spec sheet.
 */

const SPECS = [
  { icon: "battery_full", label: "Battery", value: "621 kWh (600 usable)" },
  { icon: "bolt", label: "Max Charging", value: "400 kW (CCS)" },
  { icon: "package_2", label: "Max Payload", value: "22 t (22,000 kg)" },
  { icon: "scale", label: "GCW (laden)", value: "40 t (40,000 kg)" },
  { icon: "straighten", label: "L × W × H", value: "16.5 × 2.55 × 4.0 m" },
  { icon: "trip_origin", label: "Axles", value: "5" },
];

const VEHICLES = [{ id: "eactros-600", name: "Mercedes-Benz eActros 600" }];

export default function TruckCard() {
  const [showSpecs, setShowSpecs] = useState(false);
  const [vehicle, setVehicle] = useState(VEHICLES[0].id);

  return (
    <div className="nx-card relative p-4">
      <label className="flex items-center gap-1.5 text-[11px] font-semibold text-on-surface-variant uppercase tracking-[0.08em] mb-2">
        <span className="material-symbols-outlined text-on-surface-variant/80" style={{ fontSize: "16px" }}>
          local_shipping
        </span>
        Vehicle
      </label>

      <div className="flex items-center gap-2">
        <div className="relative flex-1 min-w-0">
          <select
            value={vehicle}
            onChange={(e) => setVehicle(e.target.value)}
            aria-label="Select vehicle"
            className="w-full appearance-none rounded-control bg-surface-lowest border border-outline-variant/50 pl-3 pr-9 py-2.5 text-sm font-medium text-on-surface outline-none hover:border-outline-variant/70 focus:border-primary/50 focus:ring-2 focus:ring-primary/30 transition duration-snappy cursor-pointer"
          >
            {VEHICLES.map((v) => (
              <option key={v.id} value={v.id}>
                {v.name}
              </option>
            ))}
          </select>
          <span
            aria-hidden="true"
            className="material-symbols-outlined absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none text-on-surface-variant"
            style={{ fontSize: "20px" }}
          >
            expand_more
          </span>
        </div>

        <button
          type="button"
          onClick={() => setShowSpecs((s) => !s)}
          aria-label="Vehicle specifications"
          aria-pressed={showSpecs}
          className={`w-10 h-10 shrink-0 rounded-control border flex items-center justify-center transition-all duration-snappy ease-nx-out active:scale-95 ${
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
