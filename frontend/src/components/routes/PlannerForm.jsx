import { useState, useRef } from "react";
import LocationSearch from "./LocationSearch.jsx";
import TruckCard from "./TruckCard.jsx";

function nowLocalISO() {
  const d = new Date();
  const off = d.getTimezoneOffset();
  return new Date(d.getTime() - off * 60000).toISOString().slice(0, 16);
}

// Emerald range slider with a filled track that follows the value.
function Slider({ value, min, max, step = 1, onChange }) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0;
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
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

// A single numbered, removable, reorderable destination row that expands to
// reveal per-stop delivery options (drop-off weight, unloading time, deliver-by).
function DestinationRow({
  dest,
  index,
  onUpdate,
  onRemove,
  onDragStartRow,
  onDropRow,
}) {
  const [open, setOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);

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
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-label={`${open ? "Hide" : "Show"} options for stop ${index + 1}`}
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

      {/* Expanded per-stop options */}
      {open && (
        <div className="px-3 pb-3 pt-1 space-y-3 border-t border-outline-variant/50 bg-surface-lowest/40">
          <div>
            <FieldLabel icon="package_2" hint={`${dest.dropWeightKg} kg`}>
              Drop-off Weight
            </FieldLabel>
            <Slider
              value={dest.dropWeightKg}
              min={0}
              max={26000}
              step={250}
              onChange={(v) => onUpdate(dest.id, { dropWeightKg: v })}
            />
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <FieldLabel icon="timer">Unloading (min)</FieldLabel>
              <input
                type="number"
                min={0}
                step={5}
                value={dest.unloadMin}
                onChange={(e) => onUpdate(dest.id, { unloadMin: Number(e.target.value) })}
                className="w-full px-3 py-2 rounded-xl bg-surface-low border border-outline-variant/60 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/40 transition"
              />
            </div>
            <div>
              <FieldLabel icon="event_available">Deliver by</FieldLabel>
              <input
                type="datetime-local"
                value={dest.deliverBy}
                onChange={(e) => onUpdate(dest.id, { deliverBy: e.target.value })}
                className="w-full px-2 py-2 rounded-xl bg-surface-low border border-outline-variant/60 text-sm text-on-surface outline-none focus:ring-2 focus:ring-primary/40 transition"
              />
            </div>
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
      <style>{`
        .route-slider { -webkit-appearance: none; appearance: none; height: 6px; border-radius: 9999px; outline: none; cursor: pointer; }
        .route-slider::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 16px; height: 16px; border-radius: 9999px; background: #006d32; border: 2px solid #fff; box-shadow: 0 0 0 2px rgba(0,109,50,0.3); cursor: pointer; }
        .route-slider::-moz-range-thumb { width: 16px; height: 16px; border-radius: 9999px; background: #006d32; border: 2px solid #fff; box-shadow: 0 0 0 2px rgba(0,109,50,0.3); cursor: pointer; }
      `}</style>

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
          <Slider value={planner.startSoc} min={0} max={100} onChange={onStartSoc} />
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
                onUpdate={onUpdateDestination}
                onRemove={onRemoveDestination}
                onDragStartRow={handleDragStart}
                onDropRow={handleDrop}
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
                <Slider value={planner.minSoc} min={0} max={50} onChange={onMinSoc} />
              </div>
              <div>
                <FieldLabel icon="shield" hint={`${planner.reservePct}%`}>
                  Safety Reserve
                </FieldLabel>
                <Slider value={planner.reservePct} min={0} max={25} onChange={onReservePct} />
              </div>
              <div>
                <FieldLabel icon="alt_route" hint={`${planner.maxDetourKm} km`}>
                  Max Charging Detour
                </FieldLabel>
                <Slider value={planner.maxDetourKm} min={5} max={100} step={5} onChange={onMaxDetourKm} />
              </div>
              <div>
                <FieldLabel icon="bolt" hint={`${planner.maxChargeKw} kW`}>
                  Max Charging Speed
                </FieldLabel>
                <Slider value={planner.maxChargeKw} min={100} max={400} step={10} onChange={onMaxChargeKw} />
              </div>
              <div>
                <FieldLabel icon="scale" hint={`${planner.payloadKg} kg`}>
                  Payload
                </FieldLabel>
                <Slider
                  value={planner.payloadKg}
                  min={0}
                  max={26000}
                  step={250}
                  onChange={onPayloadKg}
                />
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
