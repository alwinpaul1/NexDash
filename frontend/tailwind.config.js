/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        primary: "#006d32",
        accent: "#00d166",
        "on-primary": "#ffffff",
        background: "#f8f9ff",
        "surface-lowest": "#ffffff",
        "surface-low": "#eff4ff",
        surface: "#e5eeff",
        "on-surface": "#0b1c30",
        "on-surface-variant": "#3c4a3d",
        secondary: "#0059bb",
        error: "#ba1a1a",
        "outline-variant": "#bbcbb9",
        // NexOS dark theme tokens
        "nex-bg": "#0a0d0e",
        "nex-panel": "#0e1413",
        "nex-panel-alt": "#101817",
        "nex-border": "rgba(255,255,255,0.08)",
        "nex-accent": "#10b981",
        "nex-accent-bright": "#34d399",
        "nex-muted": "#9ca3af",
      },
      fontFamily: {
        headline: ['"Space Grotesk"', "sans-serif"],
        body: ['"Inter"', "sans-serif"],
      },
    },
  },
  plugins: [],
};
