import ThemeToggle from "./ThemeToggle.jsx";

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
        <span
          aria-hidden="true"
          className="hidden lg:block h-9 w-px bg-outline-variant/40"
        />
        <div>
          <h1 className="font-headline font-bold text-xl text-on-surface tracking-tight">
            Dispatcher Console
          </h1>
          <p className="text-xs text-on-surface-variant -mt-0.5">EV Truck Range Intelligence</p>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <span className="hidden md:inline-flex items-center gap-1.5 rounded-pill border border-outline-variant/50 bg-surface-low/60 px-3 py-1.5 text-[11px] font-medium text-on-surface-variant">
          <span className="h-1.5 w-1.5 rounded-pill bg-accent shadow-[0_0_0_3px] shadow-accent/20" />
          Live
        </span>
        <ThemeToggle />
      </div>
    </header>
  );
}
