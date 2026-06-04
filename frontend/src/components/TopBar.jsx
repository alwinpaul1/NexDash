import { useEffect, useState } from "react";
import ThemeToggle from "./ThemeToggle.jsx";

// Live wall-clock for the header: local time (12-hour, with seconds) over the
// weekday/date. Ticks once a second; cleared on unmount. Uses the cockpit mono
// numerals (.ck-num) so the digits don't reflow as they change.
function Clock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  const time = now.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  });
  const date = now.toLocaleDateString("en-GB", {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
  return (
    <div className="hidden md:flex items-center gap-2 pr-1" title={date}>
      <span className="material-symbols-outlined text-on-surface-variant" style={{ fontSize: "18px" }}>
        schedule
      </span>
      <div className="flex flex-col items-end leading-tight">
        <span className="ck-num text-sm font-semibold text-on-surface tabular-nums">{time}</span>
        <span className="text-[10px] text-on-surface-variant -mt-0.5">{date}</span>
      </div>
    </div>
  );
}

export default function TopBar() {
  return (
    <header className="sticky top-0 z-20 flex items-center gap-4 h-16 px-4 sm:px-6 bg-background/80 backdrop-blur-md border-b border-outline-variant/50 supports-[backdrop-filter]:bg-background/70">
      <button
        className="lg:hidden flex items-center justify-center w-10 h-10 rounded-control text-on-surface-variant hover:bg-surface-low hover:text-on-surface transition-colors duration-snappy ease-nx-out nx-focus"
        aria-label="Menu"
      >
        <span className="material-symbols-outlined">menu</span>
      </button>

      <div className="hidden sm:flex items-center gap-3">
        <div>
          <h1 className="font-headline font-bold text-xl text-on-surface tracking-tight">
            Dispatcher Console
          </h1>
          <p className="text-xs text-on-surface-variant -mt-0.5">EV Truck Range Intelligence</p>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-3">
        <Clock />
        <ThemeToggle />
      </div>
    </header>
  );
}
