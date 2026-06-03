import { useState, useRef, useEffect, useCallback } from "react";
import { geocode } from "../../lib/routePlanner";

// Debounced location autocomplete (light theme). geocode(query) -> Array<{label,lat,lng}>.
// onSelect receives the chosen { label, lat, lng }.
export default function LocationSearch({
  value,
  placeholder = "Search location…",
  icon = "location_on",
  onSelect,
  onClear,
  autoFocus = false,
}) {
  const [query, setQuery] = useState(value || "");
  const [results, setResults] = useState([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState(-1);
  const wrapRef = useRef(null);
  const timerRef = useRef(null);
  const reqRef = useRef(0);

  useEffect(() => {
    setQuery(value || "");
  }, [value]);

  useEffect(() => {
    function onDoc(e) {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, []);

  const runSearch = useCallback(async (q) => {
    if (!q || q.trim().length < 3) {
      setResults([]);
      setLoading(false);
      return;
    }
    const reqId = ++reqRef.current;
    setLoading(true);
    try {
      const res = await geocode(q.trim());
      if (reqId !== reqRef.current) return;
      setResults(Array.isArray(res) ? res.slice(0, 6) : []);
      setOpen(true);
      setActive(-1);
    } catch {
      if (reqId !== reqRef.current) return;
      setResults([]);
    } finally {
      if (reqId === reqRef.current) setLoading(false);
    }
  }, []);

  function handleChange(e) {
    const q = e.target.value;
    setQuery(q);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => runSearch(q), 350);
  }

  function choose(r) {
    onSelect?.(r);
    setQuery(r.label);
    setOpen(false);
    setResults([]);
  }

  function clear() {
    setQuery("");
    setResults([]);
    setOpen(false);
    onClear?.();
  }

  function onKeyDown(e) {
    if (!open || results.length === 0) return;
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    } else if (e.key === "Enter") {
      if (active >= 0 && results[active]) {
        e.preventDefault();
        choose(results[active]);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  return (
    <div ref={wrapRef} className="group relative">
      <div className="relative flex items-center">
        <span
          className="material-symbols-outlined absolute left-2.5 text-on-surface-variant/80 pointer-events-none transition-colors duration-snappy group-focus-within:text-primary"
          style={{ fontSize: "18px" }}
        >
          {icon}
        </span>
        <input
          type="text"
          value={query}
          onChange={handleChange}
          onFocus={() => results.length > 0 && setOpen(true)}
          onKeyDown={onKeyDown}
          placeholder={placeholder}
          autoFocus={autoFocus}
          className="w-full pl-9 pr-8 py-2.5 rounded-control bg-surface-low border border-outline-variant/50 text-sm text-on-surface placeholder:text-on-surface-variant/60 outline-none hover:border-outline-variant/70 focus:border-primary/50 focus:ring-2 focus:ring-primary/30 transition duration-snappy"
        />
        {loading ? (
          <span
            className="material-symbols-outlined absolute right-2.5 text-primary animate-spin"
            style={{ fontSize: "18px" }}
          >
            progress_activity
          </span>
        ) : query ? (
          <button
            type="button"
            onClick={clear}
            aria-label="Clear"
            className="absolute right-2 flex items-center justify-center rounded-full p-0.5 text-on-surface-variant hover:text-on-surface hover:bg-surface transition-colors duration-snappy"
          >
            <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
              close
            </span>
          </button>
        ) : null}
      </div>

      {open && results.length > 0 && (
        <ul className="absolute z-[1100] mt-1.5 w-full rounded-control bg-surface-lowest border border-outline-variant/50 shadow-nx-lg overflow-hidden max-h-60 overflow-y-auto p-1">
          {results.map((r, i) => (
            <li key={`${r.lat},${r.lng},${i}`}>
              <button
                type="button"
                onMouseEnter={() => setActive(i)}
                onClick={() => choose(r)}
                className={`flex w-full items-start gap-2 rounded-[10px] px-2.5 py-2 text-left text-sm transition-colors duration-snappy ${
                  active === i ? "bg-primary/10 text-on-surface" : "text-on-surface-variant hover:bg-surface-low"
                }`}
              >
                <span
                  className={`material-symbols-outlined mt-0.5 shrink-0 transition-colors duration-snappy ${
                    active === i ? "text-primary" : "text-on-surface-variant/80"
                  }`}
                  style={{ fontSize: "16px" }}
                >
                  location_on
                </span>
                <span className="leading-snug min-w-0">
                  <span className="block font-medium text-on-surface truncate">{r.name || r.label}</span>
                  {r.region ? (
                    <span className="block text-xs text-on-surface-variant truncate">{r.region}</span>
                  ) : null}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
