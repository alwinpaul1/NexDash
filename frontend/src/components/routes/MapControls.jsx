import { useMap } from "react-leaflet";

// Light-theme map control cluster: zoom +/-, locate, fullscreen.
// Rendered as an absolutely-positioned overlay (outside MapContainer it would
// have no map ctx for zoom; the zoom buttons live inside via useMap, so this
// whole component MUST be a child of <MapContainer>).
export default function MapControls({ isFullscreen, onToggleFullscreen, onLocated }) {
  const map = useMap();

  const locate = () => {
    if (!navigator.geolocation) {
      window.alert("Location is not available in this browser.");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude } = pos.coords;
        // Fly close in (street level) and drop the pulsing GPS marker.
        map.flyTo([latitude, longitude], 16, { duration: 1.2 });
        onLocated?.({ lat: latitude, lng: longitude });
      },
      (err) => {
        window.alert(
          err.code === err.PERMISSION_DENIED
            ? "Location permission was denied. Enable it in your browser to use GPS."
            : "Couldn't get your location. Try again."
        );
      },
      { enableHighAccuracy: true, timeout: 8000, maximumAge: 10000 }
    );
  };

  const btn =
    "w-9 h-9 flex items-center justify-center bg-surface-lowest text-on-surface-variant hover:text-primary hover:bg-primary/10 transition-colors";

  return (
    <div className="absolute bottom-24 right-4 z-[1000] flex flex-col gap-2">
      {/* Zoom cluster */}
      <div className="flex flex-col rounded-xl overflow-hidden border border-outline-variant/60 shadow-sm divide-y divide-outline-variant/50">
        <button
          type="button"
          onClick={() => map.zoomIn()}
          aria-label="Zoom in"
          className={btn}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            add
          </span>
        </button>
        <button
          type="button"
          onClick={() => map.zoomOut()}
          aria-label="Zoom out"
          className={btn}
        >
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            remove
          </span>
        </button>
      </div>

      {/* Locate */}
      <button
        type="button"
        onClick={locate}
        aria-label="Locate me"
        className={`${btn} rounded-xl border border-outline-variant/60 shadow-sm`}
      >
        <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
          my_location
        </span>
      </button>

      {/* Fullscreen */}
      <button
        type="button"
        onClick={onToggleFullscreen}
        aria-label={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        className={`${btn} rounded-xl border border-outline-variant/60 shadow-sm`}
      >
        <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
          {isFullscreen ? "fullscreen_exit" : "fullscreen"}
        </span>
      </button>
    </div>
  );
}
