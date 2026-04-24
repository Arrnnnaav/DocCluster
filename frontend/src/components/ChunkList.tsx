import { useEffect, useState } from "react";
import { useAppStore } from "../store/useAppStore";

type Chunk = {
  id: string;
  text: string;
  heading: string;
  source: string;
  page: number;
  word_count: number;
  topic_id: number;
};

export function ChunkList({ topicId }: { topicId: number }) {
  const [chunks, setChunks] = useState<Chunk[] | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetch(`/api/chunks/${topicId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((data: { chunks: Chunk[] }) => {
        if (!cancelled) setChunks(data.chunks);
      })
      .catch(() => {
        if (!cancelled) setChunks([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [topicId]);

  if (loading) return <div className="p-4 text-xs text-text-secondary">Loading…</div>;
  if (!chunks || chunks.length === 0)
    return <div className="p-4 text-xs text-text-secondary">No chunks.</div>;

  return (
    <ul className="divide-y divide-border">
      {chunks.map((c) => (
        <ChunkRow key={c.id} chunk={c} />
      ))}
    </ul>
  );
}

export function ChunkRow({
  chunk,
  score,
}: {
  chunk: { id: string; text: string; heading: string; source: string };
  score?: { bm25: number; semantic: number; final: number };
}) {
  const files = useAppStore((s) => s.files);
  const setHoveredChunkId = useAppStore((s) => s.setHoveredChunkId);
  const color = files.find((f) => f.filename === chunk.source)?.color ?? "#8b949e";

  return (
    <li
      onMouseEnter={() => setHoveredChunkId(chunk.id)}
      onMouseLeave={() => setHoveredChunkId(null)}
      className="px-4 py-3 hover:bg-bg-hover transition-colors cursor-default"
    >
      <div className="flex items-center gap-2 mb-1">
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ background: color }}
        />
        <span className="font-mono text-[11px] text-accent-green truncate">
          {chunk.source}
        </span>
      </div>
      {chunk.heading && (
        <div className="font-mono text-xs text-text-primary mb-1 truncate">
          {chunk.heading}
        </div>
      )}
      <div className="text-xs text-text-secondary leading-snug">
        {chunk.text.slice(0, 200)}
        {chunk.text.length > 200 ? "…" : ""}
      </div>
      {score && (
        <div className="mt-2">
          <div className="flex items-center justify-between font-mono text-[10px] text-text-secondary mb-0.5">
            <span>relevance</span>
            <span className="text-accent">{score.final.toFixed(3)}</span>
          </div>
          <div className="h-1 w-full bg-bg-base rounded-full overflow-hidden">
            <div
              className="h-full bg-accent"
              style={{ width: `${Math.min(100, Math.max(4, score.final * 100))}%` }}
            />
          </div>
        </div>
      )}
    </li>
  );
}

export default ChunkList;
