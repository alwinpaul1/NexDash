import { useState, useCallback } from "react";
import { optimizeRoute } from "../../lib/routePlanner.js";
import PlannerForm from "./PlannerForm.jsx";
import RouteMap from "./RouteMap.jsx";
import RouteResults from "./RouteResults.jsx";
import ElevationProfile from "./ElevationProfile.jsx";
import ConditionsPanel from "./ConditionsPanel.jsx";

function nowLocalISO() {
  const d = new Date();
  const off = d.getTimezoneOffset();
  return new Date(d.getTime() - off * 60000).toISOString().slice(0, 16);
}

let destSeq = 1;
function newDest() {
  return {
    id: `d${destSeq++}`,
    label: "",
    lat: null,
    lng: null,
    dropWeightKg: 0,
    unloadMin: 30,
    deliverBy: "",
  };
}

const initialPlanner = {
  startSoc: 100,
  minSoc: 15,
  payloadKg: 0,
  reservePct: 10,
  maxDetourKm: 50,
  maxChargeKw: 400,
  origin: null,
  destinations: [newDest()],
  departure: nowLocalISO(),
};

export default function RoutesView() {
  const [planner, setPlanner] = useState(initialPlanner);
  const [status, setStatus] = useState("idle"); // idle | computing | done | error
  const [error, setError] = useState(null);
  const [plan, setPlan] = useState(null);

  const setStartSoc = useCallback((v) => setPlanner((p) => ({ ...p, startSoc: v })), []);
  const setMinSoc = useCallback((v) => setPlanner((p) => ({ ...p, minSoc: v })), []);
  const setPayloadKg = useCallback((v) => setPlanner((p) => ({ ...p, payloadKg: v })), []);
  const setReservePct = useCallback((v) => setPlanner((p) => ({ ...p, reservePct: v })), []);
  const setMaxDetourKm = useCallback((v) => setPlanner((p) => ({ ...p, maxDetourKm: v })), []);
  const setMaxChargeKw = useCallback((v) => setPlanner((p) => ({ ...p, maxChargeKw: v })), []);
  const setOrigin = useCallback(
    (o) => setPlanner((p) => ({ ...p, origin: o })),
    []
  );
  const addDestination = useCallback(
    () => setPlanner((p) => ({ ...p, destinations: [...p.destinations, newDest()] })),
    []
  );
  const updateDestination = useCallback(
    (id, patch) =>
      setPlanner((p) => ({
        ...p,
        destinations: p.destinations.map((d) => (d.id === id ? { ...d, ...patch } : d)),
      })),
    []
  );
  const removeDestination = useCallback(
    (id) =>
      setPlanner((p) => {
        const next = p.destinations.filter((d) => d.id !== id);
        return { ...p, destinations: next.length ? next : [newDest()] };
      }),
    []
  );
  const moveDestination = useCallback(
    (id, dir) =>
      setPlanner((p) => {
        const idx = p.destinations.findIndex((d) => d.id === id);
        if (idx < 0) return p;
        const swap = idx + dir;
        if (swap < 0 || swap >= p.destinations.length) return p;
        const next = p.destinations.slice();
        [next[idx], next[swap]] = [next[swap], next[idx]];
        return { ...p, destinations: next };
      }),
    []
  );
  const reorderDestination = useCallback(
    (from, to) =>
      setPlanner((p) => {
        const next = p.destinations.slice();
        if (from < 0 || from >= next.length || to < 0 || to >= next.length) return p;
        const [moved] = next.splice(from, 1);
        next.splice(to, 0, moved);
        return { ...p, destinations: next };
      }),
    []
  );
  const setDeparture = useCallback((v) => setPlanner((p) => ({ ...p, departure: v })), []);

  const reset = useCallback(() => {
    setPlanner({ ...initialPlanner, destinations: [newDest()], departure: nowLocalISO() });
    setPlan(null);
    setStatus("idle");
    setError(null);
  }, []);

  const onOptimize = useCallback(async () => {
    setStatus("computing");
    setError(null);
    try {
      const result = await optimizeRoute(planner);
      setPlan(result);
      setStatus("done");
    } catch (err) {
      setError(err?.message || "Could not plan the route.");
      setStatus("error");
    }
  }, [planner]);

  // Waypoints (origin + valid destinations) for the map pins.
  const waypoints = [];
  if (planner.origin?.lat != null) {
    waypoints.push({ label: planner.origin.label || "Origin", lat: planner.origin.lat, lng: planner.origin.lng });
  }
  for (const d of planner.destinations) {
    if (d.lat != null) waypoints.push({ label: d.label || "Destination", lat: d.lat, lng: d.lng });
  }

  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="font-headline font-bold text-2xl text-on-surface">Route Planning</h1>
        <p className="text-sm text-on-surface-variant">
          ML-driven SOC simulation with real elevation, gradient and weather along the road.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: planner form */}
        <div className="xl:col-span-1 space-y-6">
          <PlannerForm
            planner={planner}
            status={status}
            error={error}
            onStartSoc={setStartSoc}
            onMinSoc={setMinSoc}
            onPayloadKg={setPayloadKg}
            onReservePct={setReservePct}
            onMaxDetourKm={setMaxDetourKm}
            onMaxChargeKw={setMaxChargeKw}
            onSetOrigin={setOrigin}
            onAddDestination={addDestination}
            onUpdateDestination={updateDestination}
            onRemoveDestination={removeDestination}
            onReorderDestination={reorderDestination}
            onDeparture={setDeparture}
            onOptimize={onOptimize}
            onReset={reset}
          />
        </div>

        {/* Right: map + results */}
        <div className="xl:col-span-2 space-y-6">
          <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm overflow-hidden">
            <div className="h-[520px]">
              <RouteMap plan={plan} waypoints={waypoints} />
            </div>
          </div>

          {plan && (
            <>
              <ConditionsPanel conditions={plan.conditions} />
              <ElevationProfile profile={plan.elevationProfile} chargingStops={plan.chargingStops} />
            </>
          )}

          <RouteResults plan={plan} />
        </div>
      </div>
    </div>
  );
}
