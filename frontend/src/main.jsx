import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import ErrorBoundary from "./components/ErrorBoundary.jsx";
import { applyTheme, getInitialTheme } from "./lib/theme.js";
import "./index.css";

// Apply the resolved theme BEFORE first paint to avoid a flash of the wrong
// theme (FOWT). Variables in index.css switch on <html data-theme="…">.
applyTheme(getInitialTheme());

createRoot(document.getElementById("root")).render(
  <ErrorBoundary>
    <App />
  </ErrorBoundary>
);
