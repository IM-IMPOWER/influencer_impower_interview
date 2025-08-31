"use client";
import { useEffect, useState } from "react";

type KOL = {
  id: number;
  username: string;
  platform: string;
  followers?: number | null;
  category?: string[] | null;   // note: 'category', not 'categories'
};

export default function KolsPage() {
  const [kols, setKols] = useState<KOL[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        // Fetch via Next.js rewrite proxy → forwards to FastAPI /api/kols
        const res = await fetch("/api/kols?limit=50", { cache: "no-store" });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data: KOL[] = await res.json();
        setKols(data);
      } catch (e: any) {
        setError(e.message || "Failed to load");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1>KOL Directory</h1>

      {loading && <p>Loading…</p>}
      {error && <p style={{ color: "crimson" }}>Error: {error}</p>}

      {(!loading && !error && kols.length === 0) && (
        <p>No results. Try importing seed data first.</p>
      )}

      {kols.map((k) => (
        <div key={k.id} style={{ border: "1px solid #ccc", margin: "1rem 0", padding: "1rem" }}>
          <h2>@{k.username}</h2>
          <p>Platform: {k.platform}</p>
          <p>Followers: {k.followers ?? "—"}</p>
          <p>Categories: {k.category?.join(", ") || "—"}</p>
        </div>
      ))}
    </div>
  );
}
