import { useEffect, useState } from "react";
import { api } from "./api/client";
import type { Health } from "./api/types";
import { PathPicker } from "./components/PathPicker";
import { useStored } from "./components/ui";
import { InspectPage } from "./pages/Inspect";
import { JobsPage } from "./pages/Jobs";
import { PlanPage } from "./pages/Plan";
import { StatsPage } from "./pages/Stats";

const TABS = [
  { id: "inspect", label: "Inspect" },
  { id: "stats", label: "Weights" },
  { id: "plan", label: "Plan" },
  { id: "jobs", label: "Jobs" },
] as const;
type TabId = (typeof TABS)[number]["id"];

function tabFromHash(): TabId {
  const h = window.location.hash.slice(1);
  return TABS.find((t) => t.id === h)?.id ?? "inspect";
}

type Theme = "auto" | "light" | "dark";

export function App() {
  const [tab, setTab] = useState<TabId>(tabFromHash);
  const [model, setModel] = useStored("mb.model", "");
  const [draft, setDraft] = useState(model);
  const [recent, setRecent] = useStored<string[]>("mb.model.recent", []);
  const [theme, setTheme] = useStored<Theme>("mb.theme", "auto");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    const on = () => setTab(tabFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  useEffect(() => {
    api.health().then(setHealth, () => setOffline(true));
  }, []);

  const open = (p: string) => {
    const path = p.trim();
    if (!path) return;
    setDraft(path);
    setModel(path);
    setRecent([path, ...recent.filter((r) => r !== path)].slice(0, 8));
  };

  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">modelbuilder</span>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              className="tab"
              role="tab"
              aria-selected={tab === t.id}
              onClick={() => {
                window.location.hash = t.id;
                setTab(t.id);
              }}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <span className="spacer" />
        <select value={theme} onChange={(e) => setTheme(e.target.value as Theme)} aria-label="theme">
          <option value="auto">Auto theme</option>
          <option value="light">Light</option>
          <option value="dark">Dark</option>
        </select>
        {health && <span className="muted">v{health.version}</span>}
      </header>
      {offline && (
        <div className="error" style={{ marginBottom: 16 }}>
          Can't reach the modelbuilder API. Start it with <code>modelbuilder serve</code> (or <code>cargo run -- serve</code>).
        </div>
      )}
      {tab !== "jobs" && (
        <div className="row" style={{ marginBottom: 16 }}>
          <PathPicker
            value={draft}
            onChange={setDraft}
            onSubmit={open}
            accept={["gguf", "hf_model"]}
            placeholder="model: a .gguf file or a Hugging Face directory (config.json + safetensors)"
            recent={recent}
          />
          <button className="btn primary" onClick={() => open(draft)} disabled={!draft.trim()}>
            Open
          </button>
        </div>
      )}
      <main>
        {tab === "inspect" && <InspectPage model={model} />}
        {tab === "stats" && <StatsPage model={model} />}
        {tab === "plan" && <PlanPage model={model} />}
        {tab === "jobs" && <JobsPage />}
      </main>
    </div>
  );
}
