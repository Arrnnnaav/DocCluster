// Shared cluster color mapping — used by both UMAPPlot and ClusterDictionary
// so colors stay in sync between the scatter plot and the dictionary panel.
// Plotly "Dark24" palette — 24 visually distinct colors, cycles for 40+ clusters.

export const CLUSTER_PALETTE = [
  "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
  "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
  "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
  "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038",
];

export function colorFor(topicId: number): string {
  if (topicId < 0) return "#8c8c8c";
  return CLUSTER_PALETTE[topicId % CLUSTER_PALETTE.length];
}
