import { create } from "zustand";

export type FileEntry = {
  file_id: string;
  filename: string;
  color: string;
  size?: number;
};

export type Topic = {
  id: number;
  label: string;
  keywords: string[];
  doc_count: number;
  chunk_ids: string[];
};

export type UMAPPoint = {
  chunk_id: string;
  x: number;
  y: number;
  topic_id: number;
  source: string;
};

export type SearchResultItem = {
  chunk_id: string;
  chunk_text: string;
  heading: string;
  source: string;
  topic_id: number;
  topic_label: string;
  bm25_score: number;
  semantic_score: number;
  final_score: number;
};

export type PipelineStage =
  | "idle"
  | "parsing"
  | "embedding"
  | "umap"
  | "clustering"
  | "outliers"
  | "ctfidf"
  | "labeling"
  | "done"
  | "error";

export type PipelineStatus = {
  stage: PipelineStage;
  pct: number;
  message: string;
};

export type LLMConfig = {
  provider: "local" | "ollama";
  modelName: string;
  baseUrl: string;
};

type AppState = {
  files: FileEntry[];
  topics: Topic[];
  umapPoints: UMAPPoint[];
  selectedTopicId: number | null;
  searchResults: SearchResultItem[];
  matchedTopicIds: number[];
  pipelineStatus: PipelineStatus;
  isProcessing: boolean;
  llmConfig: LLMConfig;
  minClusterSize: number;
  hoveredChunkId: string | null;
  searchQuery: string;
  isSearching: boolean;
  chunkHeadings: Record<string, string>;

  addFiles: (files: FileEntry[]) => void;
  removeFile: (fileId: string) => void;
  clearFiles: () => void;
  setMinClusterSize: (n: number) => void;
  setTopics: (topics: Topic[]) => void;
  setUmapPoints: (points: UMAPPoint[]) => void;
  setSelectedTopicId: (id: number | null) => void;
  setSearchResults: (results: SearchResultItem[], matchedTopicIds?: number[]) => void;
  setPipelineStatus: (status: PipelineStatus) => void;
  setIsProcessing: (processing: boolean) => void;
  setLLMConfig: (config: Partial<LLMConfig>) => void;
  setHoveredChunkId: (id: string | null) => void;
  setSearchQuery: (q: string) => void;
  setIsSearching: (s: boolean) => void;
  clearSearch: () => void;
  setChunkHeadings: (headings: Record<string, string>) => void;
  reset: () => void;
};

const FILE_PALETTE = [
  "#7c3aed",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#3b82f6",
  "#ec4899",
  "#14b8a6",
  "#eab308",
];

export const pickFileColor = (index: number) =>
  FILE_PALETTE[index % FILE_PALETTE.length];

const INITIAL_STATUS: PipelineStatus = { stage: "idle", pct: 0, message: "" };

const INITIAL_LLM: LLMConfig = {
  provider: "local",
  modelName: "google/flan-t5-base",
  baseUrl: "http://localhost:11434",
};

export const useAppStore = create<AppState>((set) => ({
  files: [],
  topics: [],
  umapPoints: [],
  selectedTopicId: null,
  searchResults: [],
  matchedTopicIds: [],
  pipelineStatus: INITIAL_STATUS,
  isProcessing: false,
  llmConfig: INITIAL_LLM,
  minClusterSize: 5,
  hoveredChunkId: null,
  searchQuery: "",
  isSearching: false,
  chunkHeadings: {},

  addFiles: (newFiles) =>
    set((state) => {
      const base = state.files.length;
      const colored = newFiles.map((f, i) => ({
        ...f,
        color: f.color || pickFileColor(base + i),
      }));
      return { files: [...state.files, ...colored] };
    }),
  removeFile: (fileId) =>
    set((state) => ({ files: state.files.filter((f) => f.file_id !== fileId) })),
  clearFiles: () => set({ files: [] }),
  setMinClusterSize: (minClusterSize) => set({ minClusterSize }),
  setTopics: (topics) => set({ topics }),
  setUmapPoints: (umapPoints) => set({ umapPoints }),
  setSelectedTopicId: (selectedTopicId) => set({ selectedTopicId }),
  setSearchResults: (searchResults, matchedTopicIds = []) =>
    set({ searchResults, matchedTopicIds }),
  setPipelineStatus: (pipelineStatus) => set({ pipelineStatus }),
  setIsProcessing: (isProcessing) => set({ isProcessing }),
  setLLMConfig: (config) =>
    set((state) => ({ llmConfig: { ...state.llmConfig, ...config } })),
  setHoveredChunkId: (hoveredChunkId) => set({ hoveredChunkId }),
  setSearchQuery: (searchQuery) => set({ searchQuery }),
  setIsSearching: (isSearching) => set({ isSearching }),
  setChunkHeadings: (chunkHeadings) => set({ chunkHeadings }),
  clearSearch: () =>
    set({
      searchResults: [],
      matchedTopicIds: [],
      searchQuery: "",
      isSearching: false,
    }),
  reset: () =>
    set({
      files: [],
      topics: [],
      umapPoints: [],
      selectedTopicId: null,
      searchResults: [],
      matchedTopicIds: [],
      pipelineStatus: INITIAL_STATUS,
      isProcessing: false,
      chunkHeadings: {},
    }),
}));
