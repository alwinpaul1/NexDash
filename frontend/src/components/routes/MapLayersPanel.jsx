import { useState } from "react";

// Layer toggle panel (light theme). Controlled by RouteMap via `layers` + `onToggle`.
// Collapsible so it doesn't crowd the map; defaults open.
const ROWS = [
  { key: "route", icon: "route", label: "Route", color: "#006d32" },
  { key: "charging", icon: "ev_station", label: "Charging Stations", color: "#f59e0b" },
  { key: "stops", icon: "flag", label: "Planned Stops", color: "#0059bb" },
  { key: "drain", icon: "battery_charging_full", label: "Battery Drain", color: "#00d166" },
  { key: "incidents", icon: "warning", label: "Traffic Incidents", color: "#ba1a1a" },
];

export default function MapLayersPanel({ layers, onToggle }) {
  const [open, setOpen] = useState(true);

  return (
    <div className="absolute top-4 right-4 z-[1000] w-52">
      <div className="rounded-xl bg-surface-lowest/95 backdrop-blur border border-outline-variant/60 shadow-sm overflow-hidden">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between px-3 py-2 text-left"
        >
          <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-on-surface-variant">
            <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
              layers
            </span>
            Layers
          </span>
          <span
            className="material-symbols-outlined text-on-surface-variant"
            style={{ fontSize: "18px" }}
          >
            {open ? "expand_less" : "expand_more"}
          </span>
        </button>

        {open && (
          <ul className="border-t border-outline-variant/40 px-1 py-1">
            {ROWS.map((r) => {
              const on = !!layers[r.key];
              return (
                <li key={r.key}>
                  <button
                    type="button"
                    onClick={() => onToggle(r.key)}
                    className="flex w-full items-center gap-2.5 rounded-lg px-2 py-1.5 text-left text-sm text-on-surface hover:bg-surface-low transition-colors"
                  >
                    <span
                      className="material-symbols-outlined shrink-0"
                      style={{ fontSize: "18px", color: on ? r.color : "#9aa0a6" }}
                    >
                      {r.icon}
                    </span>
                    <span className={on ? "text-on-surface" : "text-on-surface-variant"}>
                      {r.label}
                    </span>
                    <span
                      role="switch"
                      aria-checked={on}
                      className={`ml-auto relative inline-flex h-4 w-7 shrink-0 items-center rounded-full transition-colors ${
                        on ? "bg-primary" : "bg-outline-variant"
                      }`}
                    >
                      <span
                        className={`inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform ${
                          on ? "translate-x-3.5" : "translate-x-0.5"
                        }`}
                      />
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
