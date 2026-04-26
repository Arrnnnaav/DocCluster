import { useAppStore } from "../store/useAppStore";
import { ChunkList, ChunkRow } from "./ChunkList";
import { TopicCard } from "./TopicCard";
import { ClusterDictionary } from "./ClusterDictionary";

export function RightPanel() {
  const selectedTopicId = useAppStore((s) => s.selectedTopicId);
  const setSelectedTopicId = useAppStore((s) => s.setSelectedTopicId);
  const topics = useAppStore((s) => s.topics);
  const isProcessing = useAppStore((s) => s.isProcessing);
  const searchResults = useAppStore((s) => s.searchResults);
  const searchQuery = useAppStore((s) => s.searchQuery);
  const clearSearch = useAppStore((s) => s.clearSearch);

  const inSearchMode = searchResults.length > 0;
  const showDictionary = topics.length > 0 && !isProcessing;

  return (
    <div className="flex flex-col h-full transition-opacity duration-200">
      {inSearchMode ? (
        <SearchResultsView
          query={searchQuery}
          onClear={clearSearch}
          results={searchResults}
        />
      ) : selectedTopicId !== null ? (
        <TopicDetailView
          topicId={selectedTopicId}
          topics={topics}
          onClose={() => setSelectedTopicId(null)}
        />
      ) : showDictionary ? (
        <ClusterDictionary />
      ) : (
        <EmptyState />
      )}
    </div>
  );
}

function TopicDetailView({
  topicId,
  topics,
  onClose,
}: {
  topicId: number;
  topics: { id: number; label: string; keywords: string[]; doc_count: number; chunk_ids: string[] }[];
  onClose: () => void;
}) {
  const topic = topics.find((t) => t.id === topicId);
  if (!topic) {
    return <EmptyState message="Topic not available." />;
  }
  return (
    <>
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <span className="font-mono text-[10px] uppercase tracking-wider text-text-secondary">
          Topic
        </span>
        <button
          onClick={onClose}
          className="text-text-secondary hover:text-accent text-sm"
          aria-label="Close topic"
        >
          ×
        </button>
      </div>
      <TopicCard topic={topic} />
      <div className="flex-1 overflow-y-auto">
        <ChunkList topicId={topicId} />
      </div>
    </>
  );
}

function SearchResultsView({
  query,
  results,
  onClear,
}: {
  query: string;
  results: {
    chunk_id: string;
    chunk_text: string;
    heading: string;
    source: string;
    topic_id: number;
    topic_label: string;
    bm25_score: number;
    semantic_score: number;
    final_score: number;
  }[];
  onClear: () => void;
}) {
  // Group by topic while preserving order of first appearance.
  const groups = new Map<
    number,
    { label: string; items: typeof results }
  >();
  for (const r of results) {
    let g = groups.get(r.topic_id);
    if (!g) {
      g = { label: r.topic_label || `Topic ${r.topic_id}`, items: [] };
      groups.set(r.topic_id, g);
    }
    g.items.push(r);
  }

  return (
    <>
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="min-w-0">
          <div className="font-mono text-[10px] uppercase tracking-wider text-text-secondary">
            Search
          </div>
          <div className="font-mono text-xs text-text-primary truncate">
            {results.length} result{results.length === 1 ? "" : "s"} for{" "}
            <span className="text-accent">“{query}”</span>
          </div>
        </div>
        <button
          onClick={onClear}
          className="text-text-secondary hover:text-accent text-sm"
          aria-label="Clear search"
        >
          ×
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {Array.from(groups.entries()).map(([tid, g]) => (
          <div key={tid}>
            <div className="px-4 py-1.5 bg-bg-base/60 border-b border-border font-mono text-[10px] uppercase tracking-wider text-accent">
              {g.label}
            </div>
            <ul className="divide-y divide-border">
              {g.items.map((r) => (
                <ChunkRow
                  key={r.chunk_id}
                  chunk={{
                    id: r.chunk_id,
                    text: r.chunk_text,
                    heading: r.heading,
                    source: r.source,
                  }}
                  score={{
                    bm25: r.bm25_score,
                    semantic: r.semantic_score,
                    final: r.final_score,
                  }}
                />
              ))}
            </ul>
          </div>
        ))}
      </div>
    </>
  );
}

function EmptyState({
  message = "Click a cluster or search to explore",
}: {
  message?: string;
}) {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="text-center text-text-secondary">
        <svg
          width="36"
          height="36"
          viewBox="0 0 24 24"
          fill="none"
          stroke="#30363d"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mx-auto mb-2"
        >
          <circle cx="12" cy="12" r="3" />
          <circle cx="5" cy="6" r="2" />
          <circle cx="19" cy="6" r="2" />
          <circle cx="5" cy="18" r="2" />
          <circle cx="19" cy="18" r="2" />
          <line x1="7" y1="7" x2="10" y2="10" />
          <line x1="17" y1="7" x2="14" y2="10" />
          <line x1="7" y1="17" x2="10" y2="14" />
          <line x1="17" y1="17" x2="14" y2="14" />
        </svg>
        <div className="font-mono text-xs">{message}</div>
      </div>
    </div>
  );
}

export default RightPanel;
