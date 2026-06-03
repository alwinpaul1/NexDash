import Sidebar from "./components/Sidebar.jsx";
import TopBar from "./components/TopBar.jsx";
import ChatWidget from "./components/console/ChatWidget.jsx";
import RoutesView from "./components/routes/RoutesView.jsx";
import { ChatProvider } from "./context/ChatContext.jsx";

export default function App() {
  return (
    // ChatProvider lifts the assistant conversation + open state so RoutesView's
    // "Optimize Route" can post a plan summary into the chat.
    <ChatProvider>
      <div className="flex min-h-screen bg-background text-on-surface font-body antialiased">
        <Sidebar />
        <main className="flex-1 min-w-0 flex flex-col">
          <TopBar />
          <div className="flex-1 min-w-0">
            <RoutesView />
          </div>
        </main>

        {/* On-demand floating Dispatcher Assistant. */}
        <ChatWidget />
      </div>
    </ChatProvider>
  );
}
