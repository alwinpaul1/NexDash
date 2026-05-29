import { useState, useCallback } from "react";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import MapOverview from "./components/MapOverview.jsx";
import KpiStrip from "./components/console/KpiStrip.jsx";
import ReachabilityWatch from "./components/console/ReachabilityWatch.jsx";
import ChatWidget from "./components/console/ChatWidget.jsx";
import RoutesView from "./components/routes/RoutesView.jsx";

export default function App() {
  const [view, setView] = useState("dashboard");
  const [chatOpen, setChatOpen] = useState(false);

  // Dashboard + Routes switch the view; "Assistant" opens the chat widget.
  const handleNavigate = useCallback((target) => {
    if (target === "assistant") {
      setChatOpen(true);
      return;
    }
    setView(target);
  }, []);

  return (
    <div className="flex min-h-screen bg-background text-on-surface">
      <Sidebar view={view} onNavigate={handleNavigate} />
      <main className="flex-1 min-w-0 flex flex-col">
        <TopBar />
        {view === "routes" ? (
          <RoutesView />
        ) : (
          <div className="p-6 space-y-6">
            <KpiStrip />
            <section className="grid grid-cols-1 xl:grid-cols-3 gap-6">
              <MapOverview />
              <ReachabilityWatch />
            </section>
          </div>
        )}
      </main>

      {/* On-demand floating Dispatcher Assistant — available on every view. */}
      <ChatWidget open={chatOpen} onOpenChange={setChatOpen} />
    </div>
  );
}
