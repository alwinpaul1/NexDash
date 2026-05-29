export default function TopBar() {
  return (
    <header className="sticky top-0 z-20 flex items-center gap-4 h-16 px-4 sm:px-6 bg-background/85 backdrop-blur border-b border-outline-variant/50">
      <button className="lg:hidden flex items-center justify-center w-10 h-10 rounded-xl hover:bg-surface-low" aria-label="Menu">
        <span className="material-symbols-outlined">menu</span>
      </button>

      <div className="hidden sm:block">
        <h1 className="font-headline font-bold text-xl text-on-surface">Dispatcher Console</h1>
        <p className="text-xs text-on-surface-variant -mt-0.5">EV Truck Range Intelligence · Germany Network</p>
      </div>
    </header>
  );
}
