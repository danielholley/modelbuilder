import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { DirListing, EntryKind } from "../api/types";
import { bytes } from "../lib/format";

const LABEL: Record<EntryKind, string> = {
  dir: "folder",
  gguf: "GGUF",
  hf_model: "HF model",
  recipe: "recipe",
  job_spec: "job spec",
  events: "events",
  other: "",
};

/** A path field with a server-side file browser. `accept` lists the entry kinds that can be picked. */
export function PathPicker({
  value,
  onChange,
  onSubmit,
  accept,
  placeholder,
  recent = [],
}: {
  value: string;
  onChange: (p: string) => void;
  onSubmit?: (p: string) => void;
  accept: EntryKind[];
  placeholder?: string;
  recent?: string[];
}) {
  const [open, setOpen] = useState(false);
  const [listing, setListing] = useState<DirListing | null>(null);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  const browse = async (path: string) => {
    setError(null);
    try {
      setListing(await api.list({ path }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => box.current && !box.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const start = () => {
    setOpen(true);
    // Start in the value's directory when it has one, else the server's working directory.
    const dir = value.includes("/") ? value.replace(/\/[^/]*$/, "") || "/" : "";
    void browse(dir);
  };

  const pick = (p: string) => {
    onChange(p);
    setOpen(false);
    onSubmit?.(p);
  };

  return (
    <div className="picker browser" ref={box}>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        list={recent.length ? "recent-paths" : undefined}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onSubmit?.(value)}
        aria-label={placeholder ?? "path"}
      />
      {recent.length > 0 && (
        <datalist id="recent-paths">
          {recent.map((r) => (
            <option key={r} value={r} />
          ))}
        </datalist>
      )}
      <button className="btn" onClick={() => (open ? setOpen(false) : start())} type="button">
        Browse…
      </button>
      {open && (
        <div className="browser-panel" role="dialog" aria-label="Choose a file">
          <div className="path">{listing?.path ?? "…"}</div>
          {error && <div className="error">{error}</div>}
          {listing?.parent && (
            <button className="entry" onClick={() => void browse(listing.parent!)}>
              <span className="name">..</span>
            </button>
          )}
          {listing?.entries.map((e) => {
            const pickable = accept.includes(e.kind);
            const enter = e.kind === "dir" || (e.kind === "hf_model" && !pickable);
            return (
              <button
                key={e.path}
                className="entry"
                onClick={() => (pickable ? pick(e.path) : enter ? void browse(e.path) : undefined)}
                onDoubleClick={() => (e.kind === "dir" || e.kind === "hf_model") && void browse(e.path)}
                disabled={!pickable && !enter}
                title={pickable && e.kind === "hf_model" ? "Click to choose; double-click to open" : undefined}
              >
                <span className="name">
                  {e.name}
                  {e.kind === "dir" || e.kind === "hf_model" ? "/" : ""}
                </span>
                {LABEL[e.kind] && <span className={pickable ? "badge" : "muted"}>{LABEL[e.kind]}</span>}
                <span className="muted mono">{e.size !== null ? bytes(e.size) : ""}</span>
              </button>
            );
          })}
          {listing && listing.entries.length === 0 && <div className="empty">Empty folder</div>}
        </div>
      )}
    </div>
  );
}
