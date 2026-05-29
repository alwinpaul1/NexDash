// TomTom Routing API helper. Builds a road-following polyline from waypoints,
// caches results in-memory, and falls back to the straight waypoint line on
// any failure so the UI always has something to draw.

const KEY = import.meta.env.VITE_TOMTOM_API_KEY;

// In-memory cache keyed by the waypoint string.
const cache = new Map();

function keyFor(waypoints) {
  return waypoints.map(([lat, lng]) => `${lat},${lng}`).join(":");
}

/**
 * Resolve a road-following polyline through the given waypoints.
 * @param {Array<[number,number]>} waypoints - 2+ [lat, lng] pairs.
 * @param {object} [opts]
 * @returns {Promise<Array<[number,number]>>} road polyline, or the straight
 *   waypoints array on any error. Never throws.
 */
export async function getRoute(waypoints, opts = {}) {
  if (!Array.isArray(waypoints) || waypoints.length < 2) {
    return waypoints || [];
  }

  const cacheKey = keyFor(waypoints);
  if (cache.has(cacheKey)) return cache.get(cacheKey);

  if (!KEY) return waypoints;

  try {
    const locs = waypoints.map(([lat, lng]) => `${lat},${lng}`).join(":");
    const url =
      `https://api.tomtom.com/routing/1/calculateRoute/${locs}/json` +
      `?key=${KEY}&travelMode=truck&routeType=fastest&traffic=false`;

    const res = await fetch(url, opts.fetchOpts);
    if (!res.ok) return waypoints;

    const data = await res.json();
    const route = data?.routes?.[0];
    if (!route?.legs?.length) return waypoints;

    const points = [];
    for (const leg of route.legs) {
      if (!leg?.points) continue;
      for (const p of leg.points) {
        if (typeof p?.latitude === "number" && typeof p?.longitude === "number") {
          points.push([p.latitude, p.longitude]);
        }
      }
    }

    if (points.length < 2) return waypoints;

    cache.set(cacheKey, points);
    return points;
  } catch {
    return waypoints;
  }
}
