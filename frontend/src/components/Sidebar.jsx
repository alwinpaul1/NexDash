export default function Sidebar() {
  return (
    <aside className="hidden lg:flex w-64 shrink-0 flex-col self-stretch sticky top-0 h-screen bg-surface-lowest border-r border-outline-variant/50 shadow-nx-sm">
      {/* Brand — cockpit logo lockup: lime ">>" chevron mark + wordmark. The
          mark box stays dark in both themes so the lime mark reads everywhere. */}
      <div className="flex items-center gap-3 px-6 h-16 border-b border-outline-variant/40">
        <span
          className="relative flex items-center justify-center w-9 h-9 rounded-control bg-[#0b0e0d] ring-1 ring-ck-lime/40 ck-glow-lime"
          aria-hidden="true"
        >
          <svg
            viewBox="0 0 24 24"
            className="w-[18px] h-[18px]"
            fill="none"
            stroke="#c6f24e"
            strokeWidth="3"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M5 5l6 7-6 7" />
            <path d="M13 5l6 7-6 7" />
          </svg>
        </span>
        <div className="leading-tight">
          <p className="font-headline font-bold text-lg text-on-surface tracking-tight">NexDash</p>
          <p className="ck-label text-[10px] font-medium ck-lime -mt-0.5">Range Intelligence</p>
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
