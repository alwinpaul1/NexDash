import { useState, useCallback, useEffect } from "react";
import { optimizeRoute } from "../../lib/routePlanner.js";
import { useChat } from "../../context/ChatContext.jsx";
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
    // Per-stop delivery data — wired through to the backend per-leg simulation
    // (payload decay after this drop, unload dwell in the ETA, deliver-by check).
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
  minChargerKw: 150,
  origin: null,
  destinations: [newDest()],
  departure: nowLocalISO(),
};

// A plan is the offline client-side fallback (backend SOC simulator unreachable)
// when its first assumption is the fallback notice. We only narrate REAL plans.
function isFallbackPlan(p) {
  const a = p?.summary?.assumptions;
  return Array.isArray(a) && typeof a[0] === "string" && a[0].startsWith("Client-side fallback estimate");
}

export default function RoutesView() {
  const [planner, setPlanner] = useState(initialPlanner);
  const [status, setStatus] = useState("idle"); // idle | computing | done | error
  const [error, setError] = useState(null);
  const [plan, setPlan] = useState(null);
  const { registerPlanRequestHandler } = useChat();

  const setStartSoc = useCallback((v) => setPlanner((p) => ({ ...p, startSoc: v })), []);
  const setMinSoc = useCallback((v) => setPlanner((p) => ({ ...p, minSoc: v })), []);
  const setPayloadKg = useCallback((v) => setPlanner((p) => ({ ...p, payloadKg: v })), []);
  const setReservePct = useCallback((v) => setPlanner((p) => ({ ...p, reservePct: v })), []);
  const setMaxDetourKm = useCallback((v) => setPlanner((p) => ({ ...p, maxDetourKm: v })), []);
  const setMaxChargeKw = useCallback((v) => setPlanner((p) => ({ ...p, maxChargeKw: v })), []);
  const setMinChargerKw = useCallback((v) => setPlanner((p) => ({ ...p, minChargerKw: v })), []);
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

  // Runs the full optimize pipeline and populates RouteResults + RouteMap.
  //   plannerOverride — use this snapshot instead of the closed-over `planner`
  //     (a chat-initiated plan sets state and passes the same object here, so
  //     it need not wait a render for `planner` to update).
  //   skipChatSummary — true for chat-initiated plans: the agent already
  //     narrated the plan in chat, so we suppress the Optimize->chat post and
  //     thereby also avoid any chat<->optimize loop.
  const onOptimize = useCallback(
    async (plannerOverride, skipChatSummary = false) => {
      // Only honor an explicit planner snapshot — the button wires
      // onClick={onOptimize}, which passes a SyntheticEvent (also an object),
      // so we require the planner-shaped `destinations` array to disambiguate.
      const isPlanner =
        plannerOverride &&
        typeof plannerOverride === "object" &&
        Array.isArray(plannerOverride.destinations);
      const activePlanner = isPlanner ? plannerOverride : planner;
      setStatus("computing");
      setError(null);
      try {
        const result = await optimizeRoute(activePlanner);
        setPlan(result);
        setStatus("done");
        // NOTE: the Optimize button does NOT post to the chat. It only fills the
        // result panel. Natural-language planning lives entirely in the chat,
        // where the agent narrates its own reply.
      } catch (err) {
        setError(err?.message || "Could not plan the route.");
        setStatus("error");
      }
    },
    [planner]
  );

  // Apply a chat-initiated planRequest: fill the planner from the agent's
  // resolved args (coords are already embedded — no geocode needed), then run
  // Optimize with that same snapshot so RouteResults + RouteMap populate exactly
  // like the button. We pass skipChatSummary=true so the chat-initiated plan
  // does NOT post a fresh summary back into chat (no infinite loop).
  const applyPlanRequest = useCallback(
    (req) => {
      if (!req || typeof req !== "object" || !req.origin || !req.destination) return;

      const origin = {
        label: req.origin.label || "",
        lat: req.origin.lat ?? null,
        lng: req.origin.lng ?? null,
      };
      const destination = {
        id: `d${destSeq++}`,
        label: req.destination.label || "",
        lat: req.destination.lat ?? null,
        lng: req.destination.lng ?? null,
        dropWeightKg: 0,
        unloadMin: 30,
        deliverBy: req.deliverBy || "",
      };

      const num = (v, fallback) => (Number.isFinite(Number(v)) ? Number(v) : fallback);

      const nextPlanner = {
        ...initialPlanner,
        origin,
        destinations: [destination],
        payloadKg: num(req.payloadKg, initialPlanner.payloadKg),
        startSoc: num(req.startSoc, initialPlanner.startSoc),
        minSoc: num(req.minSoc, initialPlanner.minSoc),
        reservePct: num(req.reservePct, initialPlanner.reservePct),
        maxChargeKw: num(req.maxChargeKw, initialPlanner.maxChargeKw),
        temperatureC: req.temperatureC ?? initialPlanner.temperatureC,
        departure: req.departure || initialPlanner.departure || nowLocalISO(),
      };

      setPlanner(nextPlanner);
      onOptimize(nextPlanner, true);
    },
    [onOptimize]
  );

  // Register the handler so ChatContext can drive an optimize from a chat reply.
  useEffect(() => {
    const unsubscribe = registerPlanRequestHandler(applyPlanRequest);
    return unsubscribe;
  }, [registerPlanRequestHandler, applyPlanRequest]);

  // Waypoints (origin + valid destinations) for the map pins.
  const waypoints = [];
  if (planner.origin?.lat != null) {
    waypoints.push({ kind: "origin", label: planner.origin.label || "Origin", lat: planner.origin.lat, lng: planner.origin.lng });
  }
  for (const d of planner.destinations) {
    if (d.lat != null) waypoints.push({ kind: "dest", label: d.label || "Destination", lat: d.lat, lng: d.lng });
  }

  return (
    <div className="p-6 space-y-6">
      {/* Screen-reader status: persistent (always mounted) live regions so AT
          announces the plan lifecycle, which is otherwise conveyed only visually. */}
      <div className="sr-only" role="status" aria-live="polite">
        {status === "computing"
          ? "Computing route…"
          : status === "done" && plan?.summary
            ? `Route ready. Arrival ${plan.summary.etaLabel ?? ""}, arrival battery ${Math.round(plan.summary.arrivalSoc ?? 0)} percent, ${plan.summary.chargingStops ?? 0} charging stops.`
            : ""}
      </div>
      <div className="sr-only" role="alert" aria-live="assertive">
        {status === "error" ? error || "Could not plan the route." : ""}
      </div>

      <div className="flex items-start gap-3">
        <span
          aria-hidden="true"
          className="mt-1 hidden sm:flex h-10 w-10 shrink-0 items-center justify-center rounded-control bg-primary/10 text-primary ring-1 ring-primary/20"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "22px" }}>
            route
          </span>
        </span>
        <div>
          <h1 className="font-headline font-bold text-2xl text-on-surface tracking-tight">
            Route Planning
          </h1>
          <p className="text-sm text-on-surface-variant mt-0.5">
            ML-driven SOC simulation with real elevation, gradient and weather along the road.
          </p>
        </div>
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
            onMinChargerKw={setMinChargerKw}
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
          <div className="nx-card overflow-hidden">
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
