export default function Sidebar() {
  return (
    <aside className="hidden lg:flex w-64 shrink-0 flex-col self-stretch sticky top-0 h-screen bg-surface-lowest border-r border-outline-variant/50 shadow-nx-sm">
      {/* Brand — the NexDash logo (same mark as the favicon) + wordmark. */}
      <div className="flex items-center gap-3 px-6 h-16 border-b border-outline-variant/40">
        <img
          src="/nexdash-logo.png"
          alt="NexDash"
          className="w-9 h-9 rounded-control object-cover ring-1 ring-primary/25 shadow-nx-sm"
        />
        <div className="leading-tight">
          <p className="font-headline font-bold text-lg text-on-surface tracking-tight">NexDash</p>
          <p className="ck-label text-[10px] font-medium text-on-surface-variant -mt-0.5">Range Intelligence</p>
        </div>
      </div>

      {/* Nav — the console is the route planner. */}
      <nav className="px-3 py-5 space-y-1">
        <p className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-on-surface-variant/70">
          Workspace
        </p>
        <span className="group flex items-center gap-3 px-3 py-2.5 rounded-control bg-primary text-on-primary font-medium shadow-nx-sm">
          <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
            route
          </span>
          Route Planner
        </span>
      </nav>
    </aside>
  );
}
