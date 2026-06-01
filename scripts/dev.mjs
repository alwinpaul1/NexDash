// Single dev entrypoint: starts the FastAPI backend and the Vite frontend
// together, and handles busy ports GRACEFULLY.
//
//   * Backend: picks the first free port starting at NEXDASH_API_PORT (default
//     8000). If 8000 is taken (e.g. a stale dev server), it transparently uses
//     8001, 8002, … and passes that SAME port to Vite's /api proxy, so the two
//     always stay wired together.
//   * Frontend: Vite auto-increments its own port (strictPort is off), so a busy
//     5173 is handled by Vite itself.
//
// Either process exiting tears the other down, and Ctrl-C stops both cleanly.
// Uses only Node built-ins — no extra dependencies.

import { spawn } from "node:child_process";
import net from "node:net";

const START_PORT = Number(process.env.NEXDASH_API_PORT) || 8000;

// Resolve the first free TCP port >= `from` (bumps on EADDRINUSE).
function findFreePort(from) {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once("error", (err) => {
      if (err.code === "EADDRINUSE") resolve(findFreePort(from + 1));
      else reject(err);
    });
    srv.listen(from, "0.0.0.0", () => {
      const { port } = srv.address();
      srv.close(() => resolve(port));
    });
  });
}

const apiPort = await findFreePort(START_PORT);
if (apiPort !== START_PORT) {
  console.log(`\x1b[33m[dev]\x1b[0m port ${START_PORT} is busy — backend will use :${apiPort} (frontend /api proxy follows it).`);
} else {
  console.log(`\x1b[33m[dev]\x1b[0m backend → :${apiPort}, frontend (Vite) → :5173 (auto-bumps if busy). Ctrl-C stops both.`);
}

const baseEnv = { ...process.env, NEXDASH_API_PORT: String(apiPort), FORCE_COLOR: "1" };

const services = [
  {
    name: "backend",
    color: "\x1b[32m", // green
    cmd: ".venv/bin/python",
    args: ["dashboard/server.py"],
    env: { ...baseEnv, PYTHONPATH: "src" },
  },
  {
    name: "frontend",
    color: "\x1b[36m", // cyan
    cmd: "npm",
    args: ["--prefix", "frontend", "run", "dev"],
    env: baseEnv,
  },
];

let shuttingDown = false;
const children = [];

function shutdown(reason) {
  if (shuttingDown) return;
  shuttingDown = true;
  if (reason) console.log(`\x1b[33m[dev]\x1b[0m ${reason} — shutting down both processes.`);
  for (const c of children) {
    try {
      c.kill("SIGTERM");
    } catch {
      /* already gone */
    }
  }
  setTimeout(() => process.exit(0), 400);
}

for (const svc of services) {
  const tag = `${svc.color}[${svc.name}]\x1b[0m`;
  const child = spawn(svc.cmd, svc.args, { env: svc.env, stdio: ["inherit", "pipe", "pipe"] });
  children.push(child);

  const prefix = (chunk) => {
    const text = chunk.toString();
    // Prefix each non-empty line so interleaved backend/frontend logs are legible.
    const out = text
      .split("\n")
      .map((line, i, arr) => (line || i < arr.length - 1 ? `${tag} ${line}` : line))
      .join("\n");
    process.stdout.write(out);
  };
  child.stdout.on("data", prefix);
  child.stderr.on("data", prefix);
  child.on("exit", (code) => shutdown(`${svc.name} exited (code ${code})`));
  child.on("error", (err) => shutdown(`${svc.name} failed to start (${err.message})`));
}

process.on("SIGINT", () => shutdown("received SIGINT"));
process.on("SIGTERM", () => shutdown("received SIGTERM"));
