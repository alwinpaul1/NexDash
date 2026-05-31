import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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

const EXAMPLE_PROMPTS = [
  "Truck at 60% SOC with 12 t — does it reach München in this weather?",
  "How much energy for a 240 km leg at 18 tonnes in the cold?",
  "At 45% charge, can I make Berlin→Leipzig fully loaded?",
];

/** Choose an icon for a tool chip based on the tool's name. */
function toolIcon(name) {
  const n = String(name).toLowerCase();
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
          className="inline-flex items-center gap-1 text-[11px] text-primary bg-primary/10 ring-1 ring-primary/20 rounded-full px-2 py-0.5"
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
    <div className="flex items-center gap-1 px-3.5 py-2.5 rounded-2xl rounded-bl-sm bg-surface-low w-fit">
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
        <div className="bg-primary text-on-primary rounded-2xl rounded-br-sm px-3.5 py-2 text-sm max-w-[85%] whitespace-pre-wrap">
          {message.content}
        </div>
      </div>
    );
  }

  // Assistant replies are rendered as Markdown (the agent emits tables, bold,
  // lists, and a caveat blockquote); GFM enables the pipe tables it uses.
  return (
    <div className="flex flex-col items-start max-w-[85%]">
      <div className="chat-md bg-surface-low text-on-surface rounded-2xl rounded-bl-sm px-3.5 py-2 text-sm">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
      </div>
      <ToolChips tools={message.tools} />
    </div>
  );
}

function EmptyState({ onPick, disabled }) {
  return (
    <div className="flex flex-col gap-4 py-2">
      <div className="flex items-start gap-2 text-sm text-on-surface-variant">
        <span className="material-symbols-outlined text-primary shrink-0" style={{ fontSize: "20px" }}>
          tips_and_updates
        </span>
        <p>
          Ask in plain language about range, energy use, or whether a truck will make
          its next stop. Try one of these:
        </p>
      </div>
      <div className="flex flex-col gap-2">
        {EXAMPLE_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            type="button"
            disabled={disabled}
            onClick={() => onPick(prompt)}
            className="text-left text-sm px-3.5 py-2.5 rounded-xl bg-surface-low text-on-surface border border-outline-variant/50 hover:border-primary/50 hover:bg-primary/5 transition focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-60"
          >
            <span className="material-symbols-outlined align-middle mr-1.5 text-primary" style={{ fontSize: "16px" }}>
              auto_awesome
            </span>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function ChatPanel() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef(null);

  // Auto-scroll to the latest message whenever the list grows or thinking shows.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  async function sendMessage(text) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    const userMessage = { role: "user", content: trimmed };
    // The request body needs the full history including this new turn. We build
    // it from the previous state to avoid a stale-closure race.
    const history = [...messages, userMessage];

    setMessages(history);
    setInput("");
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
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage(input);
  }

  function handleKeyDown(event) {
    // Enter submits; Shift+Enter inserts a newline.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage(input);
    }
  }

  return (
    <div className="bg-surface-lowest rounded-2xl border border-outline-variant/40 shadow-sm overflow-hidden flex flex-col h-[620px] max-h-[calc(100vh-7rem)]">
      {/* Header — mirrors the Live Range Check gradient treatment. */}
      <div className="px-5 py-4 bg-gradient-to-r from-primary to-accent text-on-primary">
        <h2 className="font-headline font-bold text-lg">Dispatcher Assistant</h2>
        <p className="text-sm text-on-primary/85 mt-0.5">
          Ask about range, energy &amp; reachability — eActros 600
        </p>
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
      <form onSubmit={handleSubmit} className="border-t border-outline-variant/40 p-3 flex items-end gap-2.5">
        <textarea
          rows={2}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder="Ask about a leg, SOC, payload, weather…"
          className="flex-1 resize-none max-h-40 px-4 py-3 rounded-xl bg-surface-low border border-outline-variant/60 text-base leading-snug text-on-surface placeholder:text-on-surface-variant/60 focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-70"
        />
        <button
          type="submit"
          disabled={loading || input.trim() === ""}
          aria-label="Send message"
          className="shrink-0 w-12 h-12 flex items-center justify-center rounded-full bg-primary text-on-primary hover:bg-primary/90 active:scale-95 transition focus:outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-50 disabled:active:scale-100"
        >
          <span className="material-symbols-outlined" style={{ fontSize: "24px" }}>
            send
          </span>
        </button>
      </form>
    </div>
  );
}
