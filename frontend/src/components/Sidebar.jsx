const NAV = [
  { id: "dashboard", label: "Dashboard", icon: "grid_view" },
  { id: "routes", label: "Routes", icon: "route" },
];

export default function Sidebar({ view = "dashboard", onNavigate }) {
  function handleClick(e, item) {
    // Every nav item routes through onNavigate. Dashboard + Routes switch the
    // in-app view; Range Check switches to the dashboard view and scrolls the
    // Range Check panel into view (handled by the App navigate handler).
    e.preventDefault();
    onNavigate?.(item.id);
  }

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

      {/* Nav */}
      <nav className="px-3 py-5 space-y-1">
        {NAV.map((item) => {
          const active = item.id === view;
          const href = item.id === "range-check" ? "#range-check" : "#";
          return (
            <a
              key={item.id}
              href={href}
              onClick={(e) => handleClick(e, item)}
              className={
                active
                  ? "flex items-center gap-3 px-3 py-2.5 rounded-xl bg-primary text-on-primary font-medium"
                  : "flex items-center gap-3 px-3 py-2.5 rounded-xl text-on-surface-variant hover:bg-surface-low transition-colors"
              }
            >
              <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
                {item.icon}
              </span>
              {item.label}
            </a>
          );
        })}
      </nav>
    </aside>
  );
}
