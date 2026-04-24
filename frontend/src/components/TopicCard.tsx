import { Topic, useAppStore } from "../store/useAppStore";

export function TopicCard({ topic }: { topic: Topic }) {
  const files = useAppStore((s) => s.files);
  const umapPoints = useAppStore((s) => s.umapPoints);

  const sourcesInTopic = new Set<string>();
  umapPoints.forEach((p) => {
    if (p.topic_id === topic.id) sourcesInTopic.add(p.source);
  });
  const fileCount = sourcesInTopic.size || files.length;

  return (
    <div className="px-4 py-3 border-b border-border">
      <div className="text-text-primary font-mono text-base font-semibold leading-tight">
        {topic.label}
      </div>
      <div className="mt-1 text-[11px] font-mono text-text-secondary">
        {topic.doc_count} chunk{topic.doc_count === 1 ? "" : "s"} from{" "}
        {fileCount} file{fileCount === 1 ? "" : "s"}
      </div>
      {topic.keywords.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {topic.keywords.slice(0, 8).map((kw) => (
            <span
              key={kw}
              className="text-[10px] font-mono px-1.5 py-0.5 rounded-full bg-accent/10 border border-accent/30 text-accent"
            >
              {kw}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default TopicCard;
