import { useAppStore } from "./store/useAppStore";
import { LeftPanel } from "./components/LeftPanel";
import { UMAPPlot } from "./components/UMAPPlot";
import { RightPanel } from "./components/RightPanel";
import { SearchBar } from "./components/SearchBar";
import { usePipeline } from "./hooks/usePipeline";

const LEFT_W = 280;
const RIGHT_W = 320;
const TOP_H = 48;

export default function App() {
  usePipeline();
  const status = useAppStore((s) => s.pipelineStatus);
  const isProcessing = useAppStore((s) => s.isProcessing);

  return (
    <div className="flex flex-col h-screen w-screen bg-bg-base text-text-primary font-sans">
      <header
        style={{ height: TOP_H }}
        className="flex items-center gap-4 px-4 bg-bg-panel border-b border-border shrink-0"
      >
        <span className="font-mono text-accent font-semibold tracking-wide shrink-0">
          DocuCluster
        </span>
        <SearchBar />
        <div className="ml-auto flex items-center gap-3 text-xs text-text-secondary font-mono shrink-0">
          {isProcessing ? (
            <span>
              <span className="text-accent">{status.stage}</span> · {status.pct}%
            </span>
          ) : (
            <span className="text-accent-green">idle</span>
          )}
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        <aside
          style={{ width: LEFT_W }}
          className="shrink-0 bg-bg-panel border-r border-border overflow-y-auto"
        >
          <LeftPanel />
        </aside>

        <main className="flex-1 min-w-0 bg-bg-base overflow-hidden">
          <UMAPPlot />
        </main>

        <aside
          style={{ width: RIGHT_W }}
          className="shrink-0 bg-bg-panel border-l border-border overflow-y-auto"
        >
          <RightPanel />
        </aside>
      </div>
    </div>
  );
}
