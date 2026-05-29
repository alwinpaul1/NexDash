import { useEffect } from "react";
import { useMap } from "react-leaflet";
import L from "leaflet";

// Haversine distance in meters between two [lat, lng] points.
function haversine(a, b) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(b[0] - a[0]);
  const dLng = toRad(b[1] - a[1]);
  const lat1 = toRad(a[0]);
  const lat2 = toRad(b[0]);
  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// Initial-bearing in degrees (0 = north, clockwise) from point a to point b.
function bearing(a, b) {
  const toRad = (d) => (d * Math.PI) / 180;
  const toDeg = (r) => (r * 180) / Math.PI;
  const lat1 = toRad(a[0]);
  const lat2 = toRad(b[0]);
  const dLng = toRad(b[1] - a[1]);
  const y = Math.sin(dLng) * Math.cos(lat2);
  const x =
    Math.cos(lat1) * Math.sin(lat2) -
    Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

// Top-down truck (cab + box trailer). Drawn pointing UP/north so a CSS
// rotate(bearingDeg) aligns its nose with the travel direction.
function truckSvg(color, size) {
  return `
    <div class="moving-truck-rot" style="width:${size}px;height:${size}px;will-change:transform;transform:rotate(0deg);filter:drop-shadow(0 1px 2px rgba(0,0,0,0.45));">
      <svg viewBox="0 0 24 24" width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
        <!-- box trailer (rear) -->
        <rect x="6" y="8.5" width="12" height="13" rx="1.6"
              fill="${color}" stroke="#ffffff" stroke-width="1.4"/>
        <!-- trailer roof seam -->
        <line x1="12" y1="9.5" x2="12" y2="20.5" stroke="#ffffff" stroke-width="0.7" opacity="0.55"/>
        <!-- cab (front, pointing up) -->
        <rect x="7.5" y="2.5" width="9" height="6.5" rx="1.8"
              fill="${color}" stroke="#ffffff" stroke-width="1.4"/>
        <!-- windshield -->
        <rect x="8.9" y="3.3" width="6.2" height="2.4" rx="0.8"
              fill="#ffffff" opacity="0.92"/>
      </svg>
    </div>`;
}

export default function MovingTruck({
  path,
  color = "#006d32",
  durationMs = 18000,
  size = 30,
}) {
  const map = useMap();

  useEffect(() => {
    if (!map || !Array.isArray(path) || path.length < 2) return;

    // Precompute cumulative arc-length so progress maps to a point.
    const cum = [0];
    for (let i = 1; i < path.length; i++) {
      cum[i] = cum[i - 1] + haversine(path[i - 1], path[i]);
    }
    const total = cum[cum.length - 1];
    if (total <= 0) return;

    const icon = L.divIcon({
      className: "moving-truck-icon",
      html: truckSvg(color, size),
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });

    const marker = L.marker(path[0], {
      icon,
      interactive: false,
      keyboard: false,
      zIndexOffset: 1000,
    });
    marker.addTo(map);

    let rafId = null;
    let start = null;

    const frame = (ts) => {
      if (start === null) start = ts;
      const elapsed = ts - start;
      const progress = (elapsed % durationMs) / durationMs;
      const target = progress * total;

      // Find the segment containing the target arc-length.
      let seg = 1;
      while (seg < cum.length - 1 && cum[seg] < target) seg++;
      const segStart = cum[seg - 1];
      const segLen = cum[seg] - segStart || 1;
      const t = (target - segStart) / segLen;

      const a = path[seg - 1];
      const b = path[seg];
      const pos = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
      marker.setLatLng(pos);

      // Rotate the inner SVG to face travel direction (0deg = north).
      const el = marker.getElement();
      const rot = el && el.querySelector(".moving-truck-rot");
      if (rot) rot.style.transform = `rotate(${bearing(a, b)}deg)`;

      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      map.removeLayer(marker);
    };
  }, [map, path, color, durationMs, size]);

  return null;
}
