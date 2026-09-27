import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies the API to `modelbuilder serve` (default port 7878).
const api = process.env.MODELBUILDER_API ?? "http://localhost:7878";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": { target: api, changeOrigin: true } } },
  build: { outDir: "dist", sourcemap: true },
});
