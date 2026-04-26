import { useMemo, useState } from "react";
import { useAppStore } from "../store/useAppStore";
import { colorFor } from "../utils/colors";

export function ClusterDictionary() {
  const topics = useAppStore((s) => s.topics);
  const setSelectedTopicId = useAppStore((s) => s.setSelectedTopicId);
  const [filter, setFilter] = useState("");

  // Strip outlier cluster and any garbage-labeled topics before showing in dictionary
  const validTopics = topics.filter(
    (t) =>
      t.id >= 0 &&
      t.label &&
      t.label.trim() !== "" &&
      t.label !== "—" &&
      !t.label.toLowerCase().startsWith("topic ")
  );

  const colorMap = useMemo(() => {
    const m: Record<string, string> = {};
    validTopics.forEach((t) => {
      const numId = typeof t.id === "number" ? t.id : parseInt(String(t.id), 10);
      m[String(t.id)] = isNaN(numId) ? "#6b7280" : colorFor(numId);
    });
    return m;
  }, [validTopics]);

  const dedupedTopics = useMemo(() => {
    const seen = new Map<string, number>();
    return validTopics.map((t) => {
      const base = t.label;
      if (seen.has(base)) {
        seen.set(base, seen.get(base)! + 1);
        return { ...t, label: `${base} (${seen.get(base)})` };
      }
      seen.set(base, 1);
      return t;
    });
  }, [validTopics]);

  const q = filter.toLowerCase();
  const filtered = q
    ? dedupedTopics.filter(
        (t) =>
          t.label.toLowerCase().includes(q) ||
          t.keywords.some((k) => k.toLowerCase().includes(q))
      )
    : dedupedTopics;

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border shrink-0">
        <div className="font-mono text-[10px] uppercase tracking-wider text-text-secondary mb-2">
          Cluster Dictionary
        </div>
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by label or keyword…"
          className="w-full bg-bg-base border border-border rounded px-2 py-1 text-xs text-text-primary placeholder-text-secondary font-mono outline-none focus:border-accent"
        />
      </div>

      {filtered.length === 0 ? (
        <div className="flex-1 flex items-center justify-center p-4">
          <span className="font-mono text-xs text-text-secondary">No matches</span>
        </div>
      ) : (
        <ul className="flex-1 overflow-y-auto divide-y divide-border">
          {filtered.map((topic) => {
            const color = colorMap[String(topic.id)] ?? colorMap[Number(topic.id)] ?? "#6b7280";
            const bg = color + "1a";
            const border = color + "40";
            return (
              <li
                key={topic.id}
                onClick={() => setSelectedTopicId(topic.id)}
                className="flex items-start gap-2.5 px-4 py-2.5 cursor-pointer hover:bg-white/5 transition-colors"
              >
                <span
                  className="shrink-0 rounded-full mt-1"
                  style={{ width: 10, height: 10, backgroundColor: color, display: "inline-block" }}
                />
                <div className="min-w-0 flex-1">
                  <div className="text-xs text-text-primary font-mono truncate mb-1.5">
                    {topic.label}
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {topic.keywords.slice(0, 3).map((kw) => (
                      <span
                        key={kw}
                        className="px-1.5 py-0.5 rounded text-[10px] font-mono"
                        style={{ backgroundColor: bg, color, border: `1px solid ${border}` }}
                      >
                        {kw}
                      </span>
                    ))}
                  </div>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

export default ClusterDictionary;
