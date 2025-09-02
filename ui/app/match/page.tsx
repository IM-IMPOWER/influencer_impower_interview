"use client";
import { useState } from "react";

type Hit = {
  id: number;
  username: string;
  display_name: string;
  platform: string;
  followers?: number;
  category?: string[];
  reason: string;
  sample_images: string[];
  profile_url: string;
  score?: number;
};

export default function MatchPage() {
  const [brief, setBrief] = useState("Condo Cooking");
  const [hits, setHits] = useState<Hit[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run() {
    setLoading(true); setErr(null);
    try {
      const res = await fetch(`/api/match?brief=` + encodeURIComponent(brief), { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Hit[] = await res.json();
      setHits(data);
    } catch (e: any) {
      setErr(e.message || "Failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="max-w-5xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-semibold">Match KOLs to Brief</h1>

      <div className="flex gap-3">
        <input
          className="border rounded px-3 py-2 w-full"
          value={brief}
          onChange={(e) => setBrief(e.target.value)}
          placeholder="e.g., Condo Cooking, Beauty and Personal Care, Office Worker"
        />
        <button className="border rounded px-4 py-2" onClick={run} disabled={loading}>
          {loading ? "Searching…" : "Search"}
        </button>
      </div>

      {err && <p className="text-red-600">Error: {err}</p>}

      <div className="grid grid-cols-1 gap-4">
        {hits.map((h) => (
          <div key={h.id} className="border rounded-xl p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">{h.display_name} <span className="text-gray-500">@{h.username}</span></div>
                <div className="text-sm text-gray-600">
                  {h.platform.toUpperCase()} • {h.followers?.toLocaleString() ?? "—"} followers
                </div>
                {h.category?.length ? (
                  <div className="text-sm text-gray-600">{h.category.join(", ")}</div>
                ) : null}
                <div className="text-sm mt-1">{h.reason}</div>
              </div>
              <a className="text-blue-600 underline text-sm" href={h.profile_url} target="_blank" rel="noreferrer">
                View profile →
              </a>
            </div>
            {h.sample_images?.length ? (
              <div className="mt-3 flex gap-3">
                {h.sample_images.map((src, i) => (
                  <img key={i} src={src.startsWith("/") ? src : `/api/media?path=${encodeURIComponent(src)}`} alt="" className="h-24 w-24 object-cover rounded" />
                ))}
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </main>
  );
}
