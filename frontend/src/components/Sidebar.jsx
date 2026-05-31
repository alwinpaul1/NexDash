export default function Sidebar() {
  return (
    <aside className="hidden lg:flex w-64 shrink-0 flex-col self-start sticky top-0 bg-surface-lowest border-r border-outline-variant/60 rounded-br-2xl">
      {/* Brand */}
      <div className="flex items-center gap-3 px-6 h-16 border-b border-outline-variant/40">
        <img
          src="/nexdash-logo.png"
          alt="NexDash"
          className="w-9 h-9 rounded-xl object-cover ring-1 ring-primary/20"
        />
        <div className="leading-tight">
          <p className="font-headline font-bold text-lg text-on-surface">NexDash</p>
          <p className="text-[11px] text-on-surface-variant -mt-0.5">Range Intelligence</p>
        </div>
      </div>

      {/* Nav — the console is the route planner. */}
      <nav className="px-3 py-5 space-y-1">
        <span className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary text-on-primary font-medium">
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            route
          </span>
          Route Planner
        </span>
      </nav>
    </aside>
  );
}
