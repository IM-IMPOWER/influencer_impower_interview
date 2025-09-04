"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

type MediaRow = {
  id: number;
  kol_id: number;
  kind: "profile" | "thumb" | string;
  media_url?: string | null;
  path?: string | null; // not shown, but returned
  width?: number | null;
  height?: number | null;
  mime?: string | null;
  created_at?: string | null;
};

export default function KolProfilePage() {
  const params = useParams<{ id: string }>();
  const kolId = Number(params.id);
  const [media, setMedia] = useState<MediaRow[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setErr(null);
      setLoading(true);
      try {
        const res = await fetch(`/api/kols/${kolId}/media`, { cache: "no-store" });
        if (!res.ok) throw new Error(await res.text());
        const rows: MediaRow[] = await res.json();
        setMedia(rows);
      } catch (e: any) {
        setErr(e.message || String(e));
      } finally {
        setLoading(false);
      }
    }
    if (!Number.isNaN(kolId)) load();
  }, [kolId]);

  const profile = media.find((m) => m.kind === "profile");
  const thumbs = media.filter((m) => m.kind === "thumb").slice(0, 8);

  const srcFor = (m: MediaRow) => m.media_url || `/api/media/${m.id}`;

  return (
    <main className="max-w-5xl mx-auto p-6">
      <a href="/match" className="text-sm underline">&larr; Back to matches</a>
      <h1 className="text-2xl font-semibold mb-4">KOL Profile</h1>

      {loading && <div>Loading…</div>}
      {err && <div className="text-red-600 text-sm mb-3">{err}</div>}

      {!loading && !err && (
        <>
          {/* Profile image */}
          <div className="mb-6">
            <div className="text-sm text-gray-600 mb-2">Profile</div>
            {profile ? (
              <img
                src={srcFor(profile)}
                alt="profile"
                className="w-48 h-48 object-cover rounded-lg border"
              />
            ) : (
              <div className="text-gray-500 text-sm">No profile image.</div>
            )}
          </div>

          {/* Thumbnails grid */}
          <div>
            <div className="text-sm text-gray-600 mb-2">Recent thumbnails</div>
            {thumbs.length ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {thumbs.map((t) => (
                  <img
                    key={t.id}
                    src={srcFor(t)}
                    alt="thumb"
                    className="w-full aspect-[9/16] object-cover rounded-lg border"
                  />
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-sm">No thumbnails found.</div>
            )}
          </div>
        </>
      )}
    </main>
  );
}
