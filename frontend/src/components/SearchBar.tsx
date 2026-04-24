import { useAppStore, SearchResultItem } from "../store/useAppStore";

export function SearchBar() {
  const searchQuery = useAppStore((s) => s.searchQuery);
  const setSearchQuery = useAppStore((s) => s.setSearchQuery);
  const setSearchResults = useAppStore((s) => s.setSearchResults);
  const clearSearch = useAppStore((s) => s.clearSearch);
  const isSearching = useAppStore((s) => s.isSearching);
  const setIsSearching = useAppStore((s) => s.setIsSearching);

  const doSearch = async (q: string) => {
    if (!q.trim()) {
      clearSearch();
      return;
    }
    setIsSearching(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, top_k: 20 }),
      });
      if (!res.ok) throw new Error(`Search failed: ${res.status}`);
      const data = (await res.json()) as {
        results: SearchResultItem[];
        matched_topic_ids: number[];
      };
      setSearchResults(data.results, data.matched_topic_ids);
    } catch {
      // keep previous results on error
    } finally {
      setIsSearching(false);
    }
  };

  return (
    <form
      className="flex-1 flex items-center gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        void doSearch(searchQuery);
      }}
    >
      <input
        type="text"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        placeholder="Search chunks…"
        className="flex-1 bg-bg-base border border-border rounded px-3 py-1 text-xs font-mono text-text-primary placeholder-text-secondary focus:outline-none focus:border-accent"
      />
      {searchQuery && (
        <button
          type="button"
          onClick={clearSearch}
          className="text-text-secondary hover:text-accent text-sm"
          aria-label="Clear search"
        >
          ×
        </button>
      )}
      <button
        type="submit"
        disabled={isSearching}
        className="font-mono text-xs px-3 py-1 rounded bg-accent/10 border border-accent/30 text-accent hover:bg-accent/20 disabled:opacity-50"
      >
        {isSearching ? "…" : "Search"}
      </button>
    </form>
  );
}

export default SearchBar;
