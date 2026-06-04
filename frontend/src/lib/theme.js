// Theme controller for NexDash.
//
// A single source of truth for the active color theme ("light" | "dark").
// The active theme is reflected on <html data-theme="…">, which flips the
// CSS variables defined in src/index.css. Persisted under localStorage so the
// choice sticks across sessions; falls back to the OS preference on first run.

const STORAGE_KEY = "nexdash-theme";

/** Resolve the theme to use on load: an explicit saved choice, else light.
 * Light is the default for everyone — we intentionally do NOT follow the OS
 * dark preference, so a first-time visitor always lands on the light theme. */
export function getInitialTheme() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // localStorage unavailable (private mode / SSR) — fall through.
  }
  return "light";
}

/** Apply a theme to the document (no persistence). */
export function applyTheme(theme) {
  if (typeof document !== "undefined") {
    document.documentElement.dataset.theme = theme;
  }
}

/** Persist + apply a theme. Returns the theme set. */
export function setTheme(theme) {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // ignore write failures (private mode)
  }
  applyTheme(theme);
  return theme;
}

/** Read the currently-applied theme from the document. */
export function getCurrentTheme() {
  if (typeof document !== "undefined") {
    return document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  }
  return "light";
}

/** Flip light ⇄ dark, persist, and return the new theme. */
export function toggleTheme() {
  const next = getCurrentTheme() === "dark" ? "light" : "dark";
  return setTheme(next);
}

export { STORAGE_KEY };
