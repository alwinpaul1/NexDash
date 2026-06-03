import { useEffect, useState } from "react";
import { getCurrentTheme, toggleTheme } from "../lib/theme.js";

// Accessible light/dark switch. Reflects the current theme via the shared
// theme controller (src/lib/theme.js) and flips it on click. Uses semantic
// tokens only, so it reads correctly in both themes.
export default function ThemeToggle() {
  const [theme, setThemeState] = useState(() => getCurrentTheme());

  // Sync local state if the document theme was set elsewhere (e.g. on mount).
  useEffect(() => {
    setThemeState(getCurrentTheme());
  }, []);

  const isDark = theme === "dark";

  function handleToggle() {
    setThemeState(toggleTheme());
  }

  return (
    <button
      type="button"
      onClick={handleToggle}
      role="switch"
      aria-checked={isDark}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Switch to light theme" : "Switch to dark theme"}
      className="nx-focus group inline-flex items-center justify-center w-10 h-10 rounded-control
                 border border-outline-variant/50 bg-surface-lowest text-on-surface-variant
                 transition-colors duration-snappy ease-nx-out
                 hover:bg-surface-low hover:text-on-surface"
    >
      <span
        className="material-symbols-outlined text-[20px] leading-none transition-transform duration-smooth ease-nx-out group-hover:rotate-12"
        aria-hidden="true"
      >
        {isDark ? "dark_mode" : "light_mode"}
      </span>
    </button>
  );
}
