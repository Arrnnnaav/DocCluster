import { useMemo } from "react";
import Plot from "react-plotly.js";
import type { Layout, PlotMouseEvent } from "plotly.js";
import { useAppStore } from "../store/useAppStore";
import { colorFor } from "../utils/colors";

export function UMAPPlot() {
  const umapPoints = useAppStore((s) => s.umapPoints);
  const topics = useAppStore((s) => s.topics);
  const selectedTopicId = useAppStore((s) => s.selectedTopicId);
  const setSelectedTopicId = useAppStore((s) => s.setSelectedTopicId);
  const searchResults = useAppStore((s) => s.searchResults);
  const chunkHeadings = useAppStore((s) => s.chunkHeadings);

  const topicLabelById = useMemo(() => {
    const m = new Map<number, string>();
    topics.forEach((t) => m.set(t.id, t.label));
    return m;
  }, [topics]);

  const matchedChunkIds = useMemo(
    () => new Set(searchResults.map((r) => r.chunk_id)),
    [searchResults]
  );
  const hasSearch = matchedChunkIds.size > 0;

  const traces = useMemo(() => {
    if (!umapPoints.length) return [];

    const validPoints = umapPoints.filter(
      (p) => p.x != null && p.y != null && isFinite(p.x) && isFinite(p.y)
    );

    const groups = new Map<
      string,
      {
        topicId: number;
        xs: number[];
        ys: number[];
        ids: string[];
        sizes: number[];
        opacities: number[];
        texts: string[];
      }
    >();

    for (const p of validPoints) {
      const key = `${p.topic_id}`;
      let g = groups.get(key);
      if (!g) {
        g = { topicId: p.topic_id, xs: [], ys: [], ids: [], sizes: [], opacities: [], texts: [] };
        groups.set(key, g);
      }

      const isOutlier = Number(p.topic_id) < 0;
      let size = isOutlier ? 5 : 10;
      let opacity = isOutlier ? 0.35 : 0.9;
      const isMatch = hasSearch && matchedChunkIds.has(p.chunk_id);
      const isSelected = selectedTopicId !== null && p.topic_id === selectedTopicId;

      if (hasSearch) {
        if (isMatch) { size = isOutlier ? 8 : 14; opacity = 1.0; }
        else         { opacity = 0.08; }
      } else if (selectedTopicId !== null) {
        if (isSelected) { size = isOutlier ? 7 : 13; opacity = 1.0; }
        else            { opacity = isOutlier ? 0.1 : 0.18; }
      }

      const label = topicLabelById.get(p.topic_id) ?? `Topic ${p.topic_id}`;
      const heading = (chunkHeadings[p.chunk_id] ?? "").slice(0, 80);
      const hoverText = heading
        ? `<b>${escHtml(label)}</b><br>${escHtml(heading)}…`
        : `<b>${escHtml(label)}</b><br><span style="color:#8b949e">${escHtml(p.source)}</span>`;

      g.xs.push(p.x);
      g.ys.push(p.y);
      g.ids.push(p.chunk_id);
      g.sizes.push(size);
      g.opacities.push(opacity);
      g.texts.push(hoverText);
    }

    const traceList = Array.from(groups.values()).map((g) => ({
      // WHY: scatter (not scattergl) — scattergl ignores scaleanchor, causing oval dots
      type: "scatter" as const,
      mode: "markers" as const,
      x: g.xs,
      y: g.ys,
      ids: g.ids,
      customdata: g.ids.map(() => g.topicId),
      text: g.texts,
      hovertemplate: "%{text}<extra></extra>",
      marker: {
        size: g.sizes,
        opacity: g.opacities,
        color: colorFor(g.topicId),
        symbol: "circle",
        line: { width: 1, color: "rgba(255,255,255,0.25)" },
      },
      showlegend: false,
      name: `${g.topicId}`,
    }));
    return traceList;
  }, [umapPoints, topicLabelById, chunkHeadings, matchedChunkIds, hasSearch, selectedTopicId]);

  const layout: Partial<Layout> = useMemo(
    () => ({
      autosize: true,
      uirevision: "constant",
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 20, r: 20, t: 20, b: 20 },
      xaxis: {
        autorange: true,
        showgrid: true,
        gridcolor: "rgba(255,255,255,0.05)",
        zeroline: false,
        showticklabels: false,
        showline: false,
      },
      yaxis: {
        autorange: true,
        showgrid: true,
        gridcolor: "rgba(255,255,255,0.04)",
        zeroline: false,
        showticklabels: false,
        showline: false,
        scaleanchor: "x",
        scaleratio: 1,
      },
      hoverlabel: {
        bgcolor: "#161b22",
        bordercolor: "#30363d",
        font: { color: "#e6edf3", family: "Inter", size: 12 },
      },
      dragmode: "pan",
      showlegend: false,
    }),
    []
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
          <div style={{ color: "#8b949e" }}>Upload files to begin clustering</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full h-full" style={{ aspectRatio: "unset" }}>
      <Plot
        data={traces}
        layout={layout}
        onClick={handleClick}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        config={{ displayModeBar: false, responsive: true, doubleClick: "reset" }}
      />
    </div>
  );
}

function escHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export default UMAPPlot;
