import { useAppStore, PipelineStage } from "../store/useAppStore";

const STEPS: { id: PipelineStage; label: string }[] = [
  { id: "parsing", label: "Parsing files" },
  { id: "embedding", label: "Embedding chunks" },
  { id: "umap", label: "UMAP reduction" },
  { id: "clustering", label: "HDBSCAN clustering" },
  { id: "outliers", label: "Reducing outliers" },
  { id: "ctfidf", label: "c-TF-IDF" },
  { id: "labeling", label: "LLM labels" },
];

const STAGE_ORDER: Record<PipelineStage, number> = {
  idle: -1,
  parsing: 0,
  embedding: 1,
  umap: 2,
  clustering: 3,
  outliers: 4,
  ctfidf: 5,
  labeling: 6,
  done: 7,
  error: -2,
};

export function ProgressIndicator() {
  const status = useAppStore((s) => s.pipelineStatus);
  const isProcessing = useAppStore((s) => s.isProcessing);

  if (!isProcessing && status.stage !== "done") return null;

  const currentIdx = STAGE_ORDER[status.stage] ?? -1;

  return (
    <div className="border-t border-border pt-3 mt-3 px-1">
      <div className="flex items-center justify-between mb-2 font-mono text-[10px] uppercase tracking-wider text-text-secondary">
        <span>Pipeline</span>
        <span className="text-accent">{status.pct}%</span>
      </div>

      <div className="h-1 w-full bg-bg-base rounded-full overflow-hidden mb-3">
        <div
          className="h-full bg-accent transition-all duration-300"
          style={{ width: `${status.pct}%` }}
        />
      </div>

      <ul className="space-y-1 text-xs font-mono">
        {STEPS.map((step, idx) => {
          const done = currentIdx > idx || status.stage === "done";
          const active = currentIdx === idx;
          return (
            <li
              key={step.id}
              className={
                "flex items-center gap-2 " +
                (active
                  ? "text-accent"
                  : done
                  ? "text-text-primary"
                  : "text-text-secondary")
              }
            >
              <span className="w-3 inline-block">
                {done ? "✓" : active ? "›" : "·"}
              </span>
              <span>{step.label}</span>
            </li>
          );
        })}
      </ul>

      {status.message && (
        <div className="mt-2 text-[11px] text-text-secondary truncate">
          {status.message}
        </div>
      )}
    </div>
  );
}

export default ProgressIndicator;
