import { useState, useRef } from "react";
import LocationSearch from "./LocationSearch.jsx";
import TruckCard from "./TruckCard.jsx";

// The eActros 600's maximum payload (22 t). The payload slider is capped here so
// the dispatcher can't set a load the truck physically can't carry.
const MAX_PAYLOAD_KG = 22000;

function nowLocalISO() {
  const d = new Date();
  const off = d.getTimezoneOffset();
  return new Date(d.getTime() - off * 60000).toISOString().slice(0, 16);
}

// Emerald range slider with a filled track that follows the value. Accepts an
// accessible name + human-readable value text so screen readers announce e.g.
// "Starting battery, 80 percent" instead of an unnamed slider reading "80".
function Slider({ value, min, max, step = 1, onChange, ariaLabel, ariaValueText }) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      aria-label={ariaLabel}
      aria-valuetext={ariaValueText ?? String(value)}
      className="route-slider w-full"
      style={{
        background: `linear-gradient(to right, #006d32 0%, #00d166 ${pct}%, #e5eeff ${pct}%, #e5eeff 100%)`,
      }}
    />
  );
}

function FieldLabel({ icon, children, hint }) {
  return (
    <div className="flex items-center justify-between mb-2">
      <label className="flex items-center gap-1.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide">
        {icon && (
          <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "16px" }}>
            {icon}
          </span>
        )}
        {children}
      </label>
      {hint != null && <span className="text-xs text-on-surface-variant">{hint}</span>}
    </div>
  );
}

// A single numbered, removable, drag-to-reorder destination row, backed by a
// LocationSearch geocode and an expandable panel of per-stop delivery options
// (drop-off weight, unloading dwell, deliver-by) that the backend per-leg
// simulation actually consumes (payload decay, ETA dwell, deadline feasibility).
function DestinationRow({
  dest,
  index,
  count,
  onUpdate,
  onRemove,
  onDragStartRow,
  onDropRow,
  onMove,
}) {
  const [dragOver, setDragOver] = useState(false);
  const [open, setOpen] = useState(false);

  return (
    <div
      className={`rounded-xl border bg-surface-low/60 transition-colors ${
        dragOver ? "border-primary ring-1 ring-primary/40" : "border-outline-variant/60"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        onDropRow(index);
      }}
    >
      {/* Header row: drag handle + number + search + options + remove */}
      <div className="flex items-center gap-2 px-2.5 py-2">
        <span
          draggable
          onDragStart={(e) => {
            e.dataTransfer.effectAllowed = "move";
            onDragStartRow(index);
          }}
          aria-label={`Drag to reorder stop ${index + 1}`}
          title="Drag to reorder"
          className="flex items-center justify-center w-5 shrink-0 text-on-surface-variant hover:text-primary cursor-grab active:cursor-grabbing"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            drag_indicator
          </span>
        </span>
        <span className="flex items-center justify-center w-5 h-5 shrink-0 rounded-full bg-primary/15 text-primary text-[11px] font-semibold ring-1 ring-primary/30">
          {index + 1}
        </span>
        <div className="flex-1 min-w-0">
          <LocationSearch
            value={dest.label}
            placeholder="Add a destination…"
            icon="flag"
            onSelect={(r) => onUpdate(dest.id, { label: r.label, lat: r.lat, lng: r.lng })}
            onClear={() => onUpdate(dest.id, { label: "", lat: null, lng: null })}
          />
        </div>
        {/* Keyboard/single-pointer reorder alternative to drag (WCAG 2.5.7). */}
        <button
          type="button"
          onClick={() => onMove?.(index, -1)}
          disabled={index === 0}
          aria-label={`Move stop ${index + 1} up`}
          className="flex items-center justify-center w-6 h-6 shrink-0 rounded-lg text-on-surface-variant hover:text-primary hover:bg-surface disabled:opacity-30 disabled:pointer-events-none transition-colors"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            keyboard_arrow_up
          </span>
        </button>
        <button
          type="button"
          onClick={() => onMove?.(index, 1)}
          disabled={index === (count ?? 1) - 1}
          aria-label={`Move stop ${index + 1} down`}
          className="flex items-center justify-center w-6 h-6 shrink-0 rounded-lg text-on-surface-variant hover:text-primary hover:bg-surface disabled:opacity-30 disabled:pointer-events-none transition-colors"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            keyboard_arrow_down
          </span>
        </button>
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-label={`${open ? "Hide" : "Show"} delivery options for stop ${index + 1}`}
          aria-expanded={open}
          className={`flex items-center justify-center w-6 h-6 shrink-0 rounded-lg transition-colors ${
            open ? "text-primary bg-primary/10" : "text-on-surface-variant hover:text-on-surface hover:bg-surface"
          }`}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            tune
          </span>
        </button>
        <button
          type="button"
          onClick={() => onRemove(dest.id)}
          aria-label={`Remove stop ${index + 1}`}
          className="flex items-center justify-center w-6 h-6 shrink-0 rounded-lg text-on-surface-variant hover:text-error hover:bg-error/10 transition-colors"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
            close
          </span>
        </button>
      </div>

      {/* Per-stop delivery options — these feed the backend per-leg simulation. */}
      {open && (
        <div className="px-3 pb-3 pt-1 space-y-3 border-t border-outline-variant/50 bg-surface-lowest/40">
          {/* Drop-off Weight: editable kg box + slider. The backend subtracts this
              from the payload for every leg AFTER this stop (payload decay). */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="flex items-center gap-1.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide">
                <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "16px" }}>
                  package_2
                </span>
                Drop-off Weight
              </label>
              <div className="flex items-center gap-1.5">
                <input
                  type="number"
                  min={0}
                  max={MAX_PAYLOAD_KG}
                  step={500}
                  value={dest.dropWeightKg}
                  aria-label={`Drop-off weight at stop ${index + 1} in kilograms`}
                  onChange={(e) =>
                    onUpdate(dest.id, {
                      dropWeightKg: Math.max(0, Math.min(MAX_PAYLOAD_KG, Math.round(Number(e.target.value) || 0))),
                    })
                  }
                  className="w-24 px-2 py-1 rounded-lg bg-surface-low border border-outline-variant/60 text-sm text-on-surface text-right outline-none focus:ring-2 focus:ring-primary/40 transition"
                />
                <span className="text-xs text-on-surface-variant">kg</span>
              </div>
            </div>
            <Slider
              value={dest.dropWeightKg}
              min={0}
              max={MAX_PAYLOAD_KG}
              step={250}
              ariaLabel={`Drop-off weight at stop ${index + 1}`}
              ariaValueText={`${(dest.dropWeightKg / 1000).toFixed(1)} tonnes dropped`}
              onChange={(v) => onUpdate(dest.id, { dropWeightKg: v })}
            />
          </div>
          {/* Unloading Time: editable min box. */}
          <div className="flex items-center justify-between">
            <label className="flex items-center gap-1.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide">
              <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "16px" }}>
                timer
              </span>
              Unloading Time
            </label>
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                min={0}
                step={5}
                value={dest.unloadMin}
                aria-label={`Unloading minutes at stop ${index + 1}`}
                onChange={(e) => onUpdate(dest.id, { unloadMin: Math.max(0, Math.round(Number(e.target.value) || 0)) })}
                className="w-24 px-2 py-1 rounded-lg bg-surface-low border border-outline-variant/60 text-sm text-on-surface text-right outline-none focus:ring-2 focus:ring-primary/40 transition"
              />
              <span className="text-xs text-on-surface-variant">min</span>
            </div>
          </div>
          {/* Deliver by. */}
          <div>
            <FieldLabel icon="event_available">Deliver by</FieldLabel>
            <input
              type="datetime-local"
              value={dest.deliverBy}
              aria-label={`Deliver-by deadline for stop ${index + 1}`}
              onChange={(e) => onUpdate(dest.id, { deliverBy: e.target.value })}
              className="w-full px-2 py-2 rounded-xl bg-surface-low border border-outline-variant/60 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/40 transition"
            />
          </div>
        </div>
      )}
    </div>
  );
}

export default function PlannerForm({
  planner,
  status,
  error,
  onStartSoc,
  onMinSoc,
  onPayloadKg,
  onReservePct,
  onMaxDetourKm,
  onMaxChargeKw,
  onMinChargerKw,
  onSetOrigin,
  onAddDestination,
  onUpdateDestination,
  onRemoveDestination,
  onReorderDestination,
  onDeparture,
  onOptimize,
  onReset,
}) {
  const [moreOpen, setMoreOpen] = useState(false);

  // Drag-to-reorder destinations (grip handle is the drag source).
  const dragFrom = useRef(null);
  const handleDragStart = (i) => {
    dragFrom.current = i;
  };
  const handleDrop = (i) => {
    const from = dragFrom.current;
    dragFrom.current = null;
    if (from != null && from !== i) onReorderDestination?.(from, i);
  };

  const hasOrigin = !!(planner.origin && planner.origin.lat != null);
  const validDestinations = planner.destinations.filter((d) => d.lat != null).length;
  const computing = status === "computing";
  const canOptimize = hasOrigin && validDestinations >= 1 && !computing;

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm overflow-hidden">
      <div className="px-5 py-4 bg-gradient-to-r from-primary to-accent text-on-primary">
        <div className="flex items-center gap-2">
          <span className="material-symbols-outlined">route</span>
          <h2 className="font-headline font-semibold text-lg">Route Planner</h2>
        </div>
        <p className="text-xs text-on-primary/85 mt-0.5">Optimize an eActros 600 long-haul trip</p>
      </div>

      <div className="p-5 space-y-4">
        {/* Vehicle */}
        <TruckCard />

        {/* Starting Battery */}
        <div>
          <FieldLabel icon="battery_charging_full" hint={`${planner.startSoc}%`}>
            Starting Battery
          </FieldLabel>
          <Slider value={planner.startSoc} min={0} max={100} onChange={onStartSoc}
            ariaLabel="Starting battery" ariaValueText={`${planner.startSoc} percent`} />
        </div>

        {/* Origin */}
        <div>
          <FieldLabel icon="my_location">Origin</FieldLabel>
          <LocationSearch
            value={planner.origin?.label || ""}
            placeholder="Where does the trip start?"
            icon="trip_origin"
            onSelect={(r) => onSetOrigin({ label: r.label, lat: r.lat, lng: r.lng })}
            onClear={() => onSetOrigin(null)}
          />
        </div>

        {/* Destinations */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <label className="flex items-center gap-1.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide">
              <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "16px" }}>
                flag
              </span>
              Destinations
            </label>
            <button
              type="button"
              onClick={onAddDestination}
              className="flex items-center gap-1 px-2 py-1 rounded-lg bg-primary/10 text-primary text-xs font-medium ring-1 ring-primary/25 hover:bg-primary/15 transition-colors"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                add
              </span>
              Add Stop
            </button>
          </div>
          <div className="space-y-2">
            {planner.destinations.map((d, i) => (
              <DestinationRow
                key={d.id}
                dest={d}
                index={i}
                count={planner.destinations.length}
                onUpdate={onUpdateDestination}
                onRemove={onRemoveDestination}
                onDragStartRow={handleDragStart}
                onDropRow={handleDrop}
                onMove={(idx, dir) => onReorderDestination?.(idx, idx + dir)}
              />
            ))}
            {planner.destinations.length === 0 && (
              <p className="text-xs text-on-surface-variant px-1">No stops yet. Add at least one destination.</p>
            )}
          </div>
        </div>

        {/* Departure */}
        <div>
          <FieldLabel icon="departure_board">Departure</FieldLabel>
          <div className="flex items-center gap-2">
            <input
              type="datetime-local"
              value={planner.departure}
              onChange={(e) => onDeparture(e.target.value)}
              className="flex-1 px-3 py-2.5 rounded-xl bg-surface-low border border-outline-variant/60 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/40 transition"
            />
            <button
              type="button"
              onClick={() => onDeparture(nowLocalISO())}
              className="px-3 py-2.5 rounded-xl bg-surface-low text-on-surface-variant text-xs font-semibold hover:bg-surface transition-colors"
            >
              NOW
            </button>
          </div>
        </div>

        {/* More Options (collapsible) */}
        <div className="rounded-xl border border-outline-variant/60 overflow-hidden">
          <button
            type="button"
            onClick={() => setMoreOpen((o) => !o)}
            aria-expanded={moreOpen}
            className="flex w-full items-center justify-between px-3 py-2.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide hover:bg-surface-low transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <span className="material-symbols-outlined" style={{ fontSize: "16px" }}>
                settings
              </span>
              More Options
            </span>
            <span
              className="material-symbols-outlined transition-transform"
              style={{ fontSize: "18px", transform: moreOpen ? "rotate(180deg)" : "none" }}
            >
              expand_more
            </span>
          </button>
          {moreOpen && (
            <div className="px-3 pb-3 pt-1 space-y-3 border-t border-outline-variant/50">
              <div>
                <FieldLabel icon="target" hint={`${planner.minSoc}%`}>
                  Arrive with at least
                </FieldLabel>
                <Slider value={planner.minSoc} min={0} max={50} onChange={onMinSoc}
                  ariaLabel="Minimum SOC floor" ariaValueText={`${planner.minSoc} percent`} />
                <p className="mt-1 text-[10px] leading-snug text-on-surface-variant/70">
                  Charge left in the battery when you reach the destination.
                </p>
              </div>
              <div>
                <FieldLabel icon="shield" hint={`${planner.reservePct}%`}>
                  Safety Reserve
                </FieldLabel>
                <Slider value={planner.reservePct} min={0} max={25} onChange={onReservePct}
                  ariaLabel="Safety reserve" ariaValueText={`${planner.reservePct} percent`} />
                <p className="mt-1 text-[10px] leading-snug text-on-surface-variant/70">
                  The floor the truck never dips below at any point en route.
                </p>
                <p className="mt-1 flex items-start gap-1 text-[10px] leading-snug text-on-surface-variant/60">
                  <span className="material-symbols-outlined shrink-0" style={{ fontSize: "12px", lineHeight: "1.3" }}>
                    info
                  </span>
                  <span>
                    Your battery never drops below the higher of the two —{" "}
                    <span className="font-medium">{Math.max(planner.minSoc ?? 0, planner.reservePct ?? 0)}% now</span>{" "}
                    — they aren't added together.
                  </span>
                </p>
              </div>
              <div>
                <FieldLabel icon="alt_route" hint={`${planner.maxDetourKm} km`}>
                  Max Charging Detour
                </FieldLabel>
                <Slider value={planner.maxDetourKm} min={5} max={100} step={5} onChange={onMaxDetourKm}
                  ariaLabel="Maximum charging detour" ariaValueText={`${planner.maxDetourKm} kilometres`} />
              </div>
              <div>
                <FieldLabel icon="ev_station" hint={`${planner.minChargerKw} kW`}>
                  Min Charger Speed
                </FieldLabel>
                <Slider value={planner.minChargerKw ?? 150} min={50} max={350} step={50} onChange={onMinChargerKw}
                  ariaLabel="Minimum charger power" ariaValueText={`${planner.minChargerKw ?? 150} kilowatts`} />
                <p className="mt-1 text-[10px] leading-snug text-on-surface-variant/70">
                  Skip chargers slower than this — a 40 t truck wastes hours on a low-power post.
                  Faster stations are always allowed.
                </p>
              </div>
              <div>
                <FieldLabel icon="bolt" hint={`${planner.maxChargeKw} kW`}>
                  Max Charging Speed
                </FieldLabel>
                <Slider value={planner.maxChargeKw} min={100} max={400} step={10} onChange={onMaxChargeKw} />
                <p className="mt-1 text-[10px] leading-snug text-on-surface-variant/70">
                  How hard the truck pulls — not a station filter. The eActros 600 charges up to
                  400 kW; lower this only to charge gentler (battery care). A faster station is
                  still used — the truck simply draws no more than this.
                </p>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="flex items-center gap-1.5 text-[11px] font-medium text-on-surface-variant uppercase tracking-wide">
                    <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "16px" }}>
                      scale
                    </span>
                    Payload
                  </label>
                  <div className="flex items-center gap-1.5">
                    <input
                      type="number"
                      min={0}
                      max={MAX_PAYLOAD_KG}
                      step={500}
                      value={planner.payloadKg}
                      aria-label="Payload in kilograms"
                      onChange={(e) =>
                        onPayloadKg(
                          Math.max(0, Math.min(MAX_PAYLOAD_KG, Math.round(Number(e.target.value) || 0)))
                        )
                      }
                      className="w-24 px-2 py-1 rounded-lg bg-surface-low border border-outline-variant/60 text-sm text-on-surface text-right outline-none focus:ring-2 focus:ring-primary/40 transition"
                    />
                    <span className="text-xs text-on-surface-variant">kg</span>
                  </div>
                </div>
                <Slider
                  value={planner.payloadKg}
                  min={0}
                  max={MAX_PAYLOAD_KG}
                  step={250}
                  onChange={onPayloadKg}
                />
                <p className="mt-1 text-[10px] text-on-surface-variant/70">
                  {(planner.payloadKg / 1000).toFixed(1)} t / {MAX_PAYLOAD_KG / 1000} t max
                </p>
              </div>
            </div>
          )}
        </div>

        {error && <p className="text-xs text-error px-1">{error}</p>}

        {/* Actions */}
        <div className="space-y-2 pt-1">
          <button
            type="button"
            onClick={onOptimize}
            disabled={!canOptimize}
            className="flex w-full items-center justify-center gap-2 px-4 py-3 rounded-xl bg-primary text-on-primary text-sm font-semibold hover:bg-primary/90 active:scale-[0.99] transition disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {computing ? (
              <>
                <span className="material-symbols-outlined animate-spin" style={{ fontSize: "18px" }}>
                  progress_activity
                </span>
                Optimizing…
              </>
            ) : (
              <>
                <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                  route
                </span>
                Optimize Route
              </>
            )}
          </button>
          <button
            type="button"
            onClick={onReset}
            className="block w-full text-center text-xs text-on-surface-variant hover:text-on-surface transition-colors"
          >
            Reset
          </button>
        </div>
      </div>
    </div>
  );
}
