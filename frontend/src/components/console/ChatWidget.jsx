import ChatPanel from "./ChatPanel.jsx";

/* ChatWidget — the Dispatcher Assistant as an on-demand floating widget.
 *
 * The panel stays mounted and animates in/out (scale + fade + slide from the
 * bottom-right) so BOTH opening and closing are smooth, and the conversation
 * persists while it's tucked away. A circular FAB toggles it.
 */
export default function ChatWidget({ open, onOpenChange }) {
  return (
    <>
      <div
        aria-hidden={!open}
        className={`fixed bottom-24 right-6 z-[3000] w-[460px] max-w-[calc(100vw-3rem)] origin-bottom-right transition-all duration-200 ease-out ${
          open
            ? "opacity-100 translate-y-0 scale-100"
            : "opacity-0 translate-y-3 scale-95 pointer-events-none"
        }`}
      >
        <ChatPanel />
      </div>

      <button
        type="button"
        onClick={() => onOpenChange(!open)}
        aria-label={open ? "Close assistant" : "Open Dispatcher Assistant"}
        className="fixed bottom-6 right-6 z-[3000] w-14 h-14 rounded-full bg-gradient-to-br from-primary to-accent text-on-primary shadow-lg shadow-primary/30 flex items-center justify-center hover:scale-105 active:scale-95 transition-transform"
      >
        <span
          className="material-symbols-outlined transition-transform duration-200"
          style={{ fontSize: "26px" }}
        >
          {open ? "keyboard_arrow_down" : "forum"}
        </span>
      </button>
    </>
  );
}
