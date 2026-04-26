import { useCallback, useEffect, useRef } from "react";
import { PipelineStage, useAppStore } from "../store/useAppStore";

type WsIncoming = {
  stage: string;
  pct: number;
  message: string;
  data?: {
    topics: {
      id: number;
      label: string;
      keywords: string[];
      doc_count: number;
      chunk_ids: string[];
    }[];
    umap_points: {
      chunk_id: string;
      x: number;
      y: number;
      topic_id: number;
      source: string;
    }[];
    chunk_map?: Record<string, { heading: string; source: string; topic_id: number }>;
  };
};

type RunEventDetail = {
  min_cluster_size: number;
  llm_config: { provider: string; model_name: string; base_url: string };
};

export function usePipeline() {
  const wsRef = useRef<WebSocket | null>(null);
  const setPipelineStatus = useAppStore((s) => s.setPipelineStatus);
  const setIsProcessing = useAppStore((s) => s.setIsProcessing);
  const setTopics = useAppStore((s) => s.setTopics);
  const setUmapPoints = useAppStore((s) => s.setUmapPoints);
  const setChunkHeadings = useAppStore((s) => s.setChunkHeadings);

  const handleMessage = useCallback(
    (ev: MessageEvent<string>) => {
      let msg: WsIncoming;
      try {
        msg = JSON.parse(ev.data) as WsIncoming;
      } catch {
        return;
      }
      const stage = msg.stage as PipelineStage;
      setPipelineStatus({ stage, pct: msg.pct, message: msg.message });
      if (stage === "done") {
        setIsProcessing(false);
        if (msg.data) {
          setTopics(msg.data.topics);
          setUmapPoints(msg.data.umap_points);
          if (msg.data.chunk_map) {
            const headings: Record<string, string> = {};
            for (const [cid, meta] of Object.entries(msg.data.chunk_map)) {
              headings[cid] = meta.heading ?? "";
            }
            setChunkHeadings(headings);
          }
        }
      } else if (stage === "error") {
        setIsProcessing(false);
      } else {
        setIsProcessing(true);
      }
    },
    [setPipelineStatus, setIsProcessing, setTopics, setUmapPoints, setChunkHeadings]
  );

  const getWs = useCallback((): WebSocket => {
    const existing = wsRef.current;
    if (existing && existing.readyState < WebSocket.CLOSING) return existing;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/pipeline`);
    ws.onmessage = handleMessage;
    ws.onerror = () => setIsProcessing(false);
    wsRef.current = ws;
    return ws;
  }, [handleMessage, setIsProcessing]);

  useEffect(() => {
    getWs();

    const handleRun = (e: Event) => {
      const { detail } = e as CustomEvent<RunEventDetail>;
      const ws = getWs();
      const send = () =>
        ws.send(
          JSON.stringify({
            action: "run",
            min_cluster_size: detail.min_cluster_size,
            llm_config: detail.llm_config,
          })
        );
      if (ws.readyState === WebSocket.OPEN) {
        send();
      } else {
        ws.addEventListener("open", send, { once: true });
      }
    };

    window.addEventListener("pipeline:run", handleRun);
    return () => {
      window.removeEventListener("pipeline:run", handleRun);
      wsRef.current?.close();
    };
  }, [getWs]);
}
