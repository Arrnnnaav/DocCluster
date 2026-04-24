import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout, PlotMouseEvent } from "plotly.js-dist";
import { useAppStore } from "../store/useAppStore";

const BG = "#0d1117";
const ACCENT = "#7c3aed";
const TEXT_SECONDARY = "#8b949e";

const CLUSTER_PALETTE = [
  "#7c3aed",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#3b82f6",
  "#ec4899",
  "#14b8a6",
  "#eab308",
  "#a78bfa",
  "#22d3ee",
  "#f97316",
  "#84cc16",
];

const SYMBOLS: string[] = [
  "circle",
  "square",
  "diamond",
  "cross",
  "x",
  "triangle-up",
  "triangle-down",
  "star",
];

const colorFor = (topicId: number) => {
  if (topicId < 0) return "#4b5563";
  return CLUSTER_PALETTE[topicId % CLUSTER_PALETTE.length];
};

const symbolFor = (sourceIndex: number) =>
  SYMBOLS[sourceIndex % SYMBOLS.length];

export function UMAPPlot() {
  const umapPoints = useAppStore((s) => s.umapPoints);
  const topics = useAppStore((s) => s.topics);
  const selectedTopicId = useAppStore((s) => s.selectedTopicId);
  const setSelectedTopicId = useAppStore((s) => s.setSelectedTopicId);
  const searchResults = useAppStore((s) => s.searchResults);
  const files = useAppStore((s) => s.files);

  const topicLabelById = useMemo(() => {
    const m = new Map<number, string>();
    topics.forEach((t) => m.set(t.id, t.label));
    return m;
  }, [topics]);

  const chunkById = useMemo(() => {
    const m = new Map<string, { text: string; heading: string }>();
    searchResults.forEach((r) =>
      m.set(r.chunk_id, { text: r.chunk_text, heading: r.heading })
    );
    return m;
  }, [searchResults]);

  const sourceIndex = useMemo(() => {
    const m = new Map<string, number>();
    files.forEach((f, i) => m.set(f.filename, i));
    // fall back: assign indices to any sources not in files
    umapPoints.forEach((p) => {
      if (!m.has(p.source)) m.set(p.source, m.size);
    });
    return m;
  }, [files, umapPoints]);

  const matchedChunkIds = useMemo(
    () => new Set(searchResults.map((r) => r.chunk_id)),
    [searchResults]
  );
  const hasSearch = matchedChunkIds.size > 0;

  const traces: Data[] = useMemo(() => {
    if (!umapPoints.length) return [];

    // Group points by (topic, source) so each group gets one trace with
    // consistent color + symbol + per-point size/opacity arrays.
    const groups = new Map<
      string,
      {
        topicId: number;
        source: string;
        xs: number[];
        ys: number[];
        ids: string[];
        sizes: number[];
        opacities: number[];
        labels: string[];
      }
    >();

    for (const p of umapPoints) {
      const key = `${p.topic_id}|${p.source}`;
      let g = groups.get(key);
      if (!g) {
        g = {
          topicId: p.topic_id,
          source: p.source,
          xs: [],
          ys: [],
          ids: [],
          sizes: [],
          opacities: [],
          labels: [],
        };
        groups.set(key, g);
      }

      let size = 6;
      let opacity = 0.7;
      const isMatch = hasSearch && matchedChunkIds.has(p.chunk_id);
      const isSelected =
        selectedTopicId !== null && p.topic_id === selectedTopicId;

      if (hasSearch) {
        if (isMatch) {
          size = 12;
          opacity = 1.0;
        } else {
          opacity = 0.15;
        }
      } else if (selectedTopicId !== null) {
        if (isSelected) {
          size = 9;
          opacity = 1.0;
        } else {
          opacity = 0.2;
        }
      }

      g.xs.push(p.x);
      g.ys.push(p.y);
      g.ids.push(p.chunk_id);
      g.sizes.push(size);
      g.opacities.push(opacity);

      const topicLabel = topicLabelById.get(p.topic_id) ?? `Topic ${p.topic_id}`;
      const hit = chunkById.get(p.chunk_id);
      const snippet = hit ? hit.text.slice(0, 100) : "";
      const heading = hit?.heading ?? "";
      g.labels.push(
        `<b>${escapeHtml(topicLabel)}</b>` +
          (heading ? `<br>${escapeHtml(heading)}` : "") +
          `<br><span style="color:#10b981">${escapeHtml(p.source)}</span>` +
          (snippet ? `<br><i>${escapeHtml(snippet)}…</i>` : "")
      );
    }

    return Array.from(groups.values()).map((g): Data => ({
      type: "scattergl",
      mode: "markers",
      x: g.xs,
      y: g.ys,
      ids: g.ids,
      customdata: g.ids.map(() => g.topicId) as any,
      text: g.labels,
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: g.sizes,
        opacity: g.opacities,
        color: colorFor(g.topicId),
        symbol: symbolFor(sourceIndex.get(g.source) ?? 0),
        line: { width: 0 },
      } as any,
      showlegend: false,
      name: `${g.topicId}|${g.source}`,
    }));
  }, [
    umapPoints,
    topicLabelById,
    chunkById,
    sourceIndex,
    matchedChunkIds,
    hasSearch,
    selectedTopicId,
  ]);

  const annotations = useMemo(() => {
    if (!umapPoints.length) return [];
    const centroids = new Map<number, { x: number; y: number; n: number }>();
    for (const p of umapPoints) {
      if (p.topic_id < 0) continue;
      const c = centroids.get(p.topic_id) ?? { x: 0, y: 0, n: 0 };
      c.x += p.x;
      c.y += p.y;
      c.n += 1;
      centroids.set(p.topic_id, c);
    }
    return Array.from(centroids.entries()).map(([tid, c]) => ({
      x: c.x / c.n,
      y: c.y / c.n,
      text: topicLabelById.get(tid) ?? `Topic ${tid}`,
      showarrow: false,
      font: { size: 11, color: ACCENT, family: "JetBrains Mono" },
      bgcolor: "rgba(13,17,23,0.6)",
      borderpad: 2,
    }));
  }, [umapPoints, topicLabelById]);

  const layout: Partial<Layout> = useMemo(
    () => ({
      autosize: true,
      paper_bgcolor: BG,
      plot_bgcolor: BG,
      margin: { l: 0, r: 0, t: 0, b: 0 },
      xaxis: {
        visible: false,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
      },
      yaxis: {
        visible: false,
        showgrid: false,
        zeroline: false,
        showticklabels: false,
        scaleanchor: "x",
      },
      hoverlabel: {
        bgcolor: "#161b22",
        bordercolor: "#30363d",
        font: { color: "#e6edf3", family: "Inter", size: 12 },
      },
      annotations,
      transition: { duration: 400, easing: "cubic-in-out" },
      dragmode: "pan",
      showlegend: false,
    }),
    [annotations]
  );

  const handleClick = (evt: Readonly<PlotMouseEvent>) => {
    const pt: any = evt.points?.[0];
    if (!pt) return;
    const tid = pt.customdata;
    if (typeof tid === "number") setSelectedTopicId(tid);
  };

  if (!umapPoints.length) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-text-secondary font-mono text-sm text-center">
          <div className="text-text-primary mb-1">No points yet</div>
          <div style={{ color: TEXT_SECONDARY }}>
            Upload files to begin clustering
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full">
      <Plot
        data={traces}
        layout={layout}
        onClick={handleClick}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{
          displayModeBar: false,
          responsive: true,
          doubleClick: "reset",
        }}
      />
    </div>
  );
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export default UMAPPlot;
