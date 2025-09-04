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
      {hits.map((h, i) => {
        const srcFor = (m: any) => m.media_url || `/api/media/${m.id}`;
        const kolId = h?.kol_id ?? h?.id;
        const followers =
          typeof h?.followers === "number"
            ? h.followers.toLocaleString()
            : h?.followers
            ? Number(h.followers).toLocaleString()
            : "—";

        return (
          <div key={kolId ?? `hit-${i}`} className="border rounded-xl p-4">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0">
                <div className="font-medium truncate">
                  {h?.display_name || h?.username || `KOL #${kolId ?? i}`}{" "}
                  {h?.username && <span className="text-gray-500">@{h.username}</span>}
                </div>
                <div className="text-sm text-gray-600">
                  {(h?.platform || "tiktok").toUpperCase()} • {followers} followers
                </div>
                {Array.isArray(h?.category) && h.category.length > 0 && (
                  <div className="text-sm text-gray-600">{h.category.join(", ")}</div>
                )}
                {h?.reason && <div className="text-sm mt-1">{h.reason}</div>}
              </div>

              <div className="shrink-0 flex flex-col items-end gap-1">
                <a
                  href={kolId ? `/kol/${kolId}` : "#"}
                  className={`text-sm underline inline-block ${kolId ? "" : "pointer-events-none opacity-50"}`}
                  onClick={(e) => {
                    if (!kolId) e.preventDefault();
                  }}
                >
                  View profile →
                </a>
                <a
                  href={kolId ? `/crm?kol_id=${kolId}` : "#"}
                  className={`text-sm underline inline-block ${kolId ? "" : "pointer-events-none opacity-50"}`}
                  onClick={(e) => {
                    if (!kolId) e.preventDefault();
                  }}
                >
                  Start conversation →
                </a>
              </div>
            </div>

            {Array.isArray(h?.sample_images) && h.sample_images.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-3">
                {h.sample_images.map((src: string, j: number) => {
                  // If you returned raw file paths, your API streamer is /api/media/{id}.
                  // If you returned URLs, use them directly.
                  const resolved = src.startsWith("http") || src.startsWith("/")
                    ? src
                    : `/api/media?path=${encodeURIComponent(src)}`;
                  return (
                    <img
                      key={`${kolId ?? i}-img-${j}`}
                      src={resolved}
                      alt=""
                      className="h-24 w-24 object-cover rounded border"
                    />
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  </main>
);

}
