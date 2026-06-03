/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Semantic tokens — driven by CSS variables (see src/index.css) so the
        // SAME class names (bg-surface-low, text-on-surface-variant, …) resolve
        // to the active theme. The "/ <alpha-value>" form keeps Tailwind alpha
        // suffixes (e.g. border-outline-variant/40) working in both themes.
        primary: "rgb(var(--c-primary) / <alpha-value>)",
        accent: "rgb(var(--c-accent) / <alpha-value>)",
        "on-primary": "rgb(var(--c-on-primary) / <alpha-value>)",
        background: "rgb(var(--c-background) / <alpha-value>)",
        "surface-lowest": "rgb(var(--c-surface-lowest) / <alpha-value>)",
        "surface-low": "rgb(var(--c-surface-low) / <alpha-value>)",
        surface: "rgb(var(--c-surface) / <alpha-value>)",
        "on-surface": "rgb(var(--c-on-surface) / <alpha-value>)",
        "on-surface-variant": "rgb(var(--c-on-surface-variant) / <alpha-value>)",
        secondary: "rgb(var(--c-secondary) / <alpha-value>)",
        error: "rgb(var(--c-error) / <alpha-value>)",
        "outline-variant": "rgb(var(--c-outline-variant) / <alpha-value>)",

        // Legacy NexOS dark tokens — kept as static hex for any direct refs.
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
      borderRadius: {
        // Named radii scale for consistent rounding rhythm.
        card: "1rem", // 16px — panels / cards
        control: "0.75rem", // 12px — buttons / inputs
        pill: "9999px",
      },
      boxShadow: {
        // Theme-aware elevation (vars swap per theme; see index.css).
        "nx-sm": "var(--shadow-sm)",
        "nx-md": "var(--shadow-md)",
        "nx-lg": "var(--shadow-lg)",
        "nx-ring": "var(--shadow-ring)",
      },
      transitionTimingFunction: {
        // Named easing tokens for purposeful, consistent motion.
        "nx-out": "cubic-bezier(0.16, 1, 0.3, 1)",
        "nx-in-out": "cubic-bezier(0.4, 0, 0.2, 1)",
      },
      transitionDuration: {
        snappy: "150ms",
        smooth: "240ms",
      },
    },
  },
  plugins: [],
};
