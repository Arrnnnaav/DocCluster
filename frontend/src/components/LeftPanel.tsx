import { useRef, useState } from "react";
import { pickFileColor, useAppStore } from "../store/useAppStore";
import { ProgressIndicator } from "./ProgressIndicator";

const ACCEPTED = ".pdf,.docx,.txt,.md,.markdown";

export function LeftPanel() {
  const files = useAppStore((s) => s.files);
  const addFiles = useAppStore((s) => s.addFiles);
  const removeFile = useAppStore((s) => s.removeFile);
  const minClusterSize = useAppStore((s) => s.minClusterSize);
  const setMinClusterSize = useAppStore((s) => s.setMinClusterSize);
  const topics = useAppStore((s) => s.topics);
  const llmConfig = useAppStore((s) => s.llmConfig);
  const setLLMConfig = useAppStore((s) => s.setLLMConfig);

  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);

  const uploadFiles = async (list: FileList | File[]) => {
    const arr = Array.from(list);
    if (!arr.length) return;
    const form = new FormData();
    arr.forEach((f) => form.append("files", f));
    setUploading(true);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: form });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const json = (await res.json()) as {
        files: { file_id: string; filename: string; size: number }[];
      };
      const base = files.length;
      addFiles(
        json.files.map((f, i) => ({
          file_id: f.file_id,
          filename: f.filename,
          size: f.size,
          color: pickFileColor(base + i),
        }))
      );
    } catch (err) {
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) void uploadFiles(e.dataTransfer.files);
  };

  const dispatchRun = (n: number) => {
    window.dispatchEvent(
      new CustomEvent("pipeline:run", {
        detail: {
          min_cluster_size: n,
          llm_config: {
            provider: llmConfig.provider,
            model_name: llmConfig.modelName,
            base_url: llmConfig.baseUrl,
          },
        },
      })
    );
  };

  const triggerRecluster = (n: number) => {
    setMinClusterSize(n);
    dispatchRun(n);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="p-4 flex items-center gap-2 border-b border-border">
        <ClusterIcon />
        <span className="font-mono text-accent font-semibold tracking-wide">
          DocuCluster
        </span>
      </div>

      <div className="p-4 space-y-3">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          className={
            "cursor-pointer rounded-md border border-dashed p-4 text-center text-xs font-mono transition-colors " +
            (dragOver
              ? "border-accent bg-accent/10 text-accent"
              : "border-border text-text-secondary hover:border-accent/60 hover:text-text-primary")
          }
        >
          <div className="mb-1">
            {uploading ? "Uploading…" : "Drop files here"}
          </div>
          <div className="text-[10px] opacity-70">PDF · DOCX · TXT · MD</div>
          <input
            ref={inputRef}
            type="file"
            multiple
            accept={ACCEPTED}
            hidden
            onChange={(e) => {
              if (e.target.files) void uploadFiles(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {files.length > 0 && (
          <button
            onClick={() => dispatchRun(minClusterSize)}
            className="w-full font-mono text-xs px-3 py-1.5 rounded bg-accent/10 border border-accent/30 text-accent hover:bg-accent/20 transition-colors"
          >
            Run Pipeline
          </button>
        )}

        {files.length > 0 && (
          <ul className="space-y-1">
            {files.map((f) => (
              <li
                key={f.file_id}
                className="flex items-center gap-2 px-2 py-1 rounded bg-bg-base border border-border text-xs"
              >
                <span
                  className="w-2 h-2 rounded-full shrink-0"
                  style={{ background: f.color }}
                />
                <span
                  className="flex-1 truncate font-mono text-text-primary"
                  title={f.filename}
                >
                  {f.filename}
                </span>
                <button
                  onClick={() => removeFile(f.file_id)}
                  className="text-text-secondary hover:text-accent"
                  aria-label="Remove"
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <SectionLabel>Clustering</SectionLabel>
      <div className="px-4 pb-3 space-y-2">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-text-secondary">Cluster Granularity</span>
          <span className="font-mono text-xs text-accent">{minClusterSize}</span>
        </div>
        <input
          type="range"
          min={2}
          max={20}
          step={1}
          value={minClusterSize}
          onChange={(e) => triggerRecluster(Number(e.target.value))}
          className="w-full accent-[#7c3aed]"
        />
        <div className="font-mono text-[11px] text-text-secondary">
          {topics.length} cluster{topics.length === 1 ? "" : "s"} found
        </div>
      </div>

      <SectionLabel>LLM Settings</SectionLabel>
      <div className="px-4 pb-3 space-y-2">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="radio"
            name="provider"
            checked={llmConfig.provider === "local"}
            onChange={() =>
              setLLMConfig({ provider: "local", modelName: "google/flan-t5-base" })
            }
            className="accent-[#7c3aed]"
          />
          <span>Local (Flan-T5)</span>
        </label>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="radio"
            name="provider"
            checked={llmConfig.provider === "ollama"}
            onChange={() =>
              setLLMConfig({ provider: "ollama", modelName: "llama3.2" })
            }
            className="accent-[#7c3aed]"
          />
          <span>Ollama</span>
        </label>

        {llmConfig.provider === "ollama" && (
          <div className="space-y-1 pt-1">
            <LabeledInput
              label="Base URL"
              value={llmConfig.baseUrl}
              onChange={(v) => setLLMConfig({ baseUrl: v })}
            />
            <LabeledInput
              label="Model"
              value={llmConfig.modelName}
              onChange={(v) => setLLMConfig({ modelName: v })}
            />
          </div>
        )}
      </div>

      <div className="px-4 pb-4 mt-auto">
        <ProgressIndicator />
      </div>
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-4 pt-3 pb-1 border-t border-border font-mono text-[10px] uppercase tracking-wider text-text-secondary">
      {children}
    </div>
  );
}

function LabeledInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="block">
      <span className="block text-[10px] font-mono text-text-secondary mb-0.5 uppercase tracking-wider">
        {label}
      </span>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-bg-base border border-border rounded px-2 py-1 text-xs font-mono text-text-primary focus:outline-none focus:border-accent"
      />
    </label>
  );
}

function ClusterIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="#7c3aed"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="5" cy="6" r="2" />
      <circle cx="19" cy="6" r="2" />
      <circle cx="12" cy="12" r="2.5" fill="#7c3aed" />
      <circle cx="5" cy="18" r="2" />
      <circle cx="19" cy="18" r="2" />
      <line x1="6.6" y1="7" x2="10.3" y2="10.7" />
      <line x1="17.4" y1="7" x2="13.7" y2="10.7" />
      <line x1="6.6" y1="17" x2="10.3" y2="13.3" />
      <line x1="17.4" y1="17" x2="13.7" y2="13.3" />
    </svg>
  );
}

export default LeftPanel;
