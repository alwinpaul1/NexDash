import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useChat } from "../../context/ChatContext.jsx";

/* ChatPanel — the dispatcher's plain-language assistant (case-study Part 2).
 *
 * A chat surface over POST /api/chat. The dispatcher types questions about
 * range, energy and reachability for an eActros 600; the backend LLM agent
 * answers, calling tools (check_reachability, estimate_energy, …) as needed.
 *
 * Contract:
 *   request:  { messages: [{ role: "user"|"assistant", content }, …] }  // full history
 *   response: { reply: string, tools: string[] }                        // tools may be []
 *
 * Each assistant turn stores the tool names it used so we can render chips
 * under its bubble. Fail-soft: on fetch/HTTP error we append a readable
 * assistant message instead of crashing. A degraded (no-API-key) reply still
 * comes back 200 and renders as a normal assistant message.
 */

/** Choose an icon for a tool chip based on the tool's name. */
function toolIcon(name) {
  const n = String(name).toLowerCase();
  if (n.includes("plan_route") || n.includes("route") || n.includes("reach")) return "alt_route";
  if (n.includes("energy") || n.includes("charge") || n.includes("soc")) return "bolt";
  return "build";
}

function ToolChips({ tools }) {
  if (!Array.isArray(tools) || tools.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5 mt-1.5">
      {tools.map((tool, i) => (
        <span
          key={tool + i}
          className="inline-flex items-center gap-1 text-[11px] font-medium text-primary bg-primary/10 ring-1 ring-primary/20 rounded-pill px-2 py-0.5"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "13px" }}>
            {toolIcon(tool)}
          </span>
          {tool}
        </span>
      ))}
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 px-3.5 py-2.5 rounded-card rounded-bl-sm bg-surface-low border border-outline-variant/40 w-fit">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="w-1.5 h-1.5 rounded-full bg-on-surface-variant/60 animate-bounce"
          style={{ animationDelay: i * 0.15 + "s" }}
        />
      ))}
    </div>
  );
}

function MessageBubble({ message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="bg-primary text-on-primary rounded-card rounded-br-sm px-3.5 py-2 text-sm max-w-[85%] whitespace-pre-wrap shadow-nx-sm">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant replies are rendered as Markdown (the agent emits tables, bold,
  // lists, and a caveat blockquote); GFM enables the pipe tables it uses.
  return (
    <div className="flex flex-col items-start max-w-[85%]">
      <div className="chat-md bg-surface-low text-on-surface border border-outline-variant/40 rounded-card rounded-bl-sm px-3.5 py-2 text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
      </div>
      <ToolChips tools={message.tools} />
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col gap-4 py-2">
      <div className="nx-card-inset flex items-start gap-2.5 px-3.5 py-3 text-sm text-on-surface-variant">
        <span className="material-symbols-outlined text-primary shrink-0" style={{ fontSize: "20px" }}>
          tips_and_updates
        </span>
        <p>
          Ask about range, energy, or a truck&rsquo;s next stop — or describe a full trip
          (<strong>where, when, load</strong>): e.g. &ldquo;Berlin to Munich, 12 t, depart 9 am,
          deliver by Friday 9 pm&rdquo;. Weather and elevation are added automatically.
        </p>
      </div>
    </div>
  );
}

export default function ChatPanel() {
  // Conversation + send-flow live in ChatContext so the "Optimize Route" button
  // (RoutesView) can post narrated plan summaries into the same conversation.
  const { messages, loading, sendMessage } = useChat();
  const [input, setInput] = useState("");
  const scrollRef = useRef(null);

  // Auto-scroll to the latest message whenever the list grows or thinking shows.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  function handleSend(text) {
    sendMessage(text);
    setInput("");
  }

  function handleSubmit(event) {
    event.preventDefault();
    handleSend(input);
  }

  function handleKeyDown(event) {
    // Enter submits; Shift+Enter inserts a newline.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend(input);
    }
  }

  return (
    <div className="bg-surface-lowest rounded-card border border-outline-variant/50 shadow-nx-lg overflow-hidden flex flex-col h-[620px] max-h-[calc(100vh-7rem)]">
      {/* Header — mirrors the Live Range Check gradient treatment. */}
      <div className="relative px-5 py-4 bg-gradient-to-br from-primary to-accent text-on-primary">
        <div className="flex items-center gap-2.5">
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-control bg-on-primary/15 ring-1 ring-on-primary/20">
            <span className="material-symbols-outlined" style={{ fontSize: "20px" }}>
              forum
            </span>
          </span>
          <div>
            <h2 className="font-headline font-bold text-lg leading-tight">Dispatcher Assistant</h2>
            <p className="text-sm text-on-primary/85 mt-0.5">
              Ask about range, energy &amp; reachability — eActros 600
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.length === 0 ? (
          <EmptyState onPick={sendMessage} disabled={loading} />
        ) : (
          messages.map((message, i) => <MessageBubble key={i} message={message} />)
        )}
        {loading ? <ThinkingDots /> : null}
      </div>

      {/* Input row */}
      <form onSubmit={handleSubmit} className="border-t border-outline-variant/40 bg-surface-low/40 p-3 flex items-end gap-2.5">
        <textarea
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="Ask about a route, payload, departure time, deadline…"
          className="flex-1 resize-none max-h-40 px-4 py-3 rounded-control bg-surface-lowest border border-outline-variant/60 text-base leading-snug text-on-surface placeholder:text-on-surface-variant/60 transition-shadow duration-snappy ease-nx-out focus:outline-none focus:ring-2 focus:ring-primary/40 focus:border-primary/40 disabled:opacity-70"
        />
        <button
          type="submit"
          disabled={loading || input.trim() === ""}
          aria-label="Send message"
          className="shrink-0 w-12 h-12 flex items-center justify-center rounded-pill bg-primary text-on-primary shadow-nx-sm hover:bg-primary/90 active:scale-95 transition duration-snappy ease-nx-out focus:outline-none focus:ring-2 focus:ring-primary/40 focus:ring-offset-2 focus:ring-offset-surface-lowest disabled:opacity-50 disabled:active:scale-100"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "24px" }}>
            send
          </span>
        </button>
      </form>
    </div>
  );
}
