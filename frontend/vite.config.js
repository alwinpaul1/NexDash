import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Backend port: the dev launcher (scripts/dev.mjs) sets NEXDASH_API_PORT to the
// free port it bound the backend to, so the /api proxy always follows the backend
// even when 8000 was busy. Falls back to 8000 for a standalone `vite`.
const apiPort = process.env.NEXDASH_API_PORT || "8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // strictPort is off by default → Vite auto-bumps if 5173 is busy
    proxy: {
      "/api": {
        target: `http://localhost:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
});
