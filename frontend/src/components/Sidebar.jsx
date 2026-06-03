export default function Sidebar() {
  return (
    <aside className="hidden lg:flex w-64 shrink-0 flex-col self-stretch sticky top-0 h-screen bg-surface-lowest border-r border-outline-variant/50 shadow-nx-sm">
      {/* Brand */}
      <div className="flex items-center gap-3 px-6 h-16 border-b border-outline-variant/40">
        <img
          src="/nexdash-logo.png"
          alt="NexDash"
          className="w-9 h-9 rounded-control object-cover ring-1 ring-primary/25 shadow-nx-sm"
        />
        <div className="leading-tight">
          <p className="font-headline font-bold text-lg text-on-surface tracking-tight">NexDash</p>
          <p className="text-[11px] font-medium uppercase tracking-wider text-on-surface-variant -mt-0.5">
            Range Intelligence
          </p>
        </div>
      </div>

      {/* Nav — the console is the route planner. */}
      <nav className="px-3 py-5 space-y-1">
        <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant/70">
          Workspace
        </p>
        <span className="group relative flex items-center gap-3 px-3 py-2.5 rounded-control bg-primary text-on-primary font-medium shadow-nx-sm">
          <span
            aria-hidden="true"
            className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-pill bg-on-primary/80"
          />
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            route
          </span>
          Route Planner
        </span>
      </nav>

      {/* Footer marker — fills the rail and anchors the brand identity. */}
      <div className="mt-auto px-6 py-5 border-t border-outline-variant/40">
        <div className="flex items-center gap-2 text-[11px] text-on-surface-variant">
          <span className="h-1.5 w-1.5 rounded-pill bg-accent shadow-[0_0_0_3px] shadow-accent/20" />
          eActros 600 · live model
        </div>
      </div>
    </aside>
  );
}
