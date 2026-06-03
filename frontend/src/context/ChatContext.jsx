import { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";

/* ChatContext — lifts the Dispatcher Assistant conversation out of ChatPanel so
 * it can be driven from anywhere in the app (notably RoutesView's "Optimize
 * Route" button, which posts an agent-narrated summary of the just-computed
 * plan into the chat).
 *
 * It owns the message list + the /api/chat send-flow (formerly ChatPanel's
 * local state) and the floating widget's open/close state. ChatPanel consumes
 * `messages`, `loading`, `sendMessage`; ChatWidget consumes `open`/`setOpen`.
 *
 * Contract (unchanged from ChatPanel):
 *   request:  { messages: [{ role, content }, …] }   // full history
 *   response: { reply: string, tools: string[] }
 */

const ChatContext = createContext(null);

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within a <ChatProvider>.");
  return ctx;
}

export function ChatProvider({ children }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  // Mirror of `messages` so sendMessage can read the latest history without a
  // stale closure (it isn't in the callback's dep list).
  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  // Guards against double-posting the same optimize narration (e.g. a second
  // render of the same successful plan). RoutesView passes a stable key.
  const lastPostedRef = useRef(null);
  // RoutesView registers a handler here so a chat-initiated plan (the agent's
  // structured `planRequest`) can fill the planner + run Optimize, populating
  // RouteResults + RouteMap exactly as if the button were clicked.
  const planRequestHandlerRef = useRef(null);
  const registerPlanRequestHandler = useCallback((fn) => {
    planRequestHandlerRef.current = typeof fn === "function" ? fn : null;
    // Return an unsubscribe so the consumer can clean up on unmount.
    return () => {
      if (planRequestHandlerRef.current === fn) planRequestHandlerRef.current = null;
    };
  }, []);

  const sendMessage = useCallback(
    async (text) => {
      const trimmed = String(text ?? "").trim();
      if (!trimmed || loading) return;

      const userMessage = { role: "user", content: trimmed };
      // The request body needs the full history including this new turn. Read
      // it from the ref to avoid a stale-closure race.
      const history = [...messagesRef.current, userMessage];

      setMessages(history);
      setLoading(true);

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          // Strip tool metadata — the contract only wants role + content.
          body: JSON.stringify({
            messages: history.map((m) => ({ role: m.role, content: m.content })),
          }),
        });

        if (!res.ok) throw new Error("Server returned HTTP " + res.status + ".");

        const body = await res.json();
        const reply =
          typeof body.reply === "string" && body.reply.length > 0
            ? body.reply
            : "The assistant returned an empty reply.";
        const tools = Array.isArray(body.tools) ? body.tools : [];

        setMessages((prev) => [...prev, { role: "assistant", content: reply, tools }]);

        // If the agent called plan_route this turn, the server returns a
        // structured `planRequest` (origin/destination WITH coords + planner
        // inputs). Hand it to RoutesView's registered handler, which fills the
        // planner and runs Optimize — RouteResults + RouteMap populate exactly
        // as if the dispatcher had filled the form. The text reply above still
        // renders normally in chat.
        const planRequest = body && typeof body.planRequest === "object" ? body.planRequest : null;
        if (planRequest && planRequestHandlerRef.current) {
          try {
            planRequestHandlerRef.current(planRequest);
          } catch {
            // A handler failure must never break the chat send-flow.
          }
        }
      } catch (err) {
        const detail =
          err instanceof TypeError
            ? "I couldn't reach the assistant. Is the API running on this host?"
            : err && err.message
            ? err.message
            : String(err);
        setMessages((prev) => [
          ...prev,
          {
            role: "assistant",
            content: "Sorry — something went wrong. " + detail,
            tools: [],
            isError: true,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  // Open the widget and post a one-shot message — used by RoutesView after a
  // successful optimize. `key` de-dupes so the same plan is narrated once.
  const postSummary = useCallback(
    (text, key) => {
      if (key != null && lastPostedRef.current === key) return;
      lastPostedRef.current = key ?? null;
      setOpen(true);
      sendMessage(text);
    },
    [sendMessage]
  );

  const value = {
    messages,
    loading,
    open,
    setOpen,
    sendMessage,
    postSummary,
    registerPlanRequestHandler,
  };
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export default ChatContext;
