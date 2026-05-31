import { Component } from "react";

/* Top-level error boundary. Without one, any uncaught render error (e.g. a
 * malformed plan/summary shape) blanks the entire app — which would undercut
 * the deliberate fail-soft story the routePlanner.js fallbacks otherwise tell.
 * This catches it and shows a recoverable card instead. */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Log for debugging; the UI below handles user-facing recovery.
    console.error("NexDash UI error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div
          role="alert"
          className="min-h-screen flex items-center justify-center bg-surface p-6"
        >
          <div className="max-w-md w-full rounded-2xl border border-outline-variant/50 bg-surface-lowest shadow-sm p-6 text-center">
            <span
              className="material-symbols-outlined text-error"
              style={{ fontSize: "40px" }}
            >
              error
            </span>
            <h1 className="mt-2 font-headline font-bold text-xl text-on-surface">
              Something went wrong
            </h1>
            <p className="mt-1 text-sm text-on-surface-variant">
              The console hit an unexpected error. Your data is safe — reloading
              usually clears it.
            </p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 inline-flex items-center gap-1.5 px-4 py-2 rounded-xl bg-primary text-on-primary text-sm font-medium hover:opacity-90 transition"
            >
              <span className="material-symbols-outlined" style={{ fontSize: "18px" }}>
                refresh
              </span>
              Reload
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
