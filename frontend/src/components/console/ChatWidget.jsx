import ChatPanel from "./ChatPanel.jsx";
import { useChat } from "../../context/ChatContext.jsx";

/* ChatWidget — the Dispatcher Assistant as an on-demand floating widget.
 *
 * The panel stays mounted and animates in/out (scale + fade + slide from the
 * bottom-right) so BOTH opening and closing are smooth, and the conversation
 * persists while it's tucked away. A circular FAB toggles it.
 *
 * Open/close state lives in ChatContext so RoutesView can open the widget when
 * it posts an optimize summary into the chat.
 */
export default function ChatWidget() {
  const { open, setOpen } = useChat();
  const onOpenChange = setOpen;
  return (
    <>
      <div
        aria-hidden={!open}
        className={`fixed bottom-24 right-6 z-[3000] w-[460px] max-w-[calc(100vw-3rem)] origin-bottom-right transition-all duration-smooth ease-nx-out ${
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
        className="group fixed bottom-6 right-6 z-[3000] w-14 h-14 rounded-pill bg-gradient-to-br from-primary to-accent text-on-primary shadow-nx-lg shadow-primary/30 flex items-center justify-center ring-1 ring-on-primary/10 hover:scale-105 active:scale-95 transition-transform duration-smooth ease-nx-out nx-focus"
      >
        <span
          className="material-symbols-outlined transition-transform duration-smooth ease-nx-out group-hover:rotate-3"
          style={{ fontSize: "26px" }}
        >
          {open ? "keyboard_arrow_down" : "forum"}
        </span>
      </button>
    </>
  );
}
