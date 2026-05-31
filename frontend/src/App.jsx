import { useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import ChatWidget from "./components/console/ChatWidget.jsx";
import RoutesView from "./components/routes/RoutesView.jsx";

export default function App() {
  const [chatOpen, setChatOpen] = useState(false);

  return (
    <div className="flex min-h-screen bg-background text-on-surface">
      <Sidebar />
      <main className="flex-1 min-w-0 flex flex-col">
        <TopBar />
        <RoutesView />
      </main>

      {/* On-demand floating Dispatcher Assistant. */}
      <ChatWidget open={chatOpen} onOpenChange={setChatOpen} />
    </div>
  );
}
