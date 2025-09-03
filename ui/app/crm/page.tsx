"use client";
import { useEffect, useState } from "react";

type Conversation = {
  id: number;
  kol_id: number;
  status: "contacted" | "negotiating" | "confirmed" | "closed";
  channel: "dm" | "email" | "line";
  last_message_at?: string | null;
  created_at: string;
  kol_username?: string | null;
  kol_display_name?: string | null;
  kol_followers?: number | null;
  category?: string[] | null;
  proposed_deliverables?: string | null;
  proposed_timeline?: string | null;
  proposed_price_integer?: number | null;
  agreed_deliverables?: string | null;
  agreed_timeline?: string | null;
  agreed_price_integer?: number | null;
};

type Message = {
  id: number;
  conversation_id: number;
  direction: "out" | "in";
  body: string;
  created_at: string;
};

/* ---------------- NEW: templates ---------------- */
const TEMPLATES = [
  {
    id: "opening",
    label: "Opening",
    text:
      "Hi {display_name}, we’ve been enjoying your content around {category}. " +
      "We think you’d be a great fit for an upcoming campaign and wanted to see if you’re open to a collaboration?",
  },
  {
    id: "offer",
    label: "Offer",
    text:
      "Thanks for connecting, {display_name}!\n" +
      "We’d like to propose:\n• Deliverables: 1 TikTok video + 1 story\n• Timing: within the next 2 weeks\n• Compensation: THB {budget}\n\nWould this work for you?",
  },
  {
    id: "followup",
    label: "Polite follow-up",
    text:
      "Hi {display_name}, just checking in on my earlier message.\n" +
      "If you’re interested, we’d be happy to adjust timing or scope to fit your schedule.",
  },
  {
    id: "final",
    label: "Final confirmation",
    text:
      "Perfect, thanks for agreeing, {display_name}.\nTo confirm:\n" +
      "• Deliverables: {deliverables}\n• Timeline: {timeline}\n• Compensation: THB {budget}\n\nLooking forward to collaborating with you!",
  },
];

function fillTemplate(
  raw: string,
  conv: Conversation | null,
  extras: Record<string, string> = {}
) {
  const display_name =
    (conv?.kol_display_name || conv?.kol_username || "").trim() || "there";
  const category =
    (conv?.category && conv.category.length > 0
      ? conv.category.join(", ")
      : "your niche");
  const followers =
    conv?.kol_followers != null ? conv.kol_followers.toLocaleString() : "—";

  const map: Record<string, string> = {
    display_name,
    category,
    followers,
    budget: extras.budget ?? "30,000",
    deliverables: extras.deliverables ?? "1 TikTok video + 1 story",
    timeline: extras.timeline ?? "next 2 weeks",
  };

  return raw.replace(/\{(\w+)\}/g, (_, k) => map[k] ?? `{${k}}`);
}

export default function CRMPage({
  searchParams,
}: {
  searchParams?: { kol_id?: string };
}) {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<Conversation | null>(null);
  const [msgs, setMsgs] = useState<Message[]>([]);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingThread, setLoadingThread] = useState(false);
  const [newMsg, setNewMsg] = useState("");
  const [err, setErr] = useState<string | null>(null);

  /* ---------------- load conversations ---------------- */
  async function loadConvs() {
    setLoadingList(true);
    setErr(null);
    try {
      const res = await fetch("/api/conversations?limit=100", { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load conversations");
      const data: Conversation[] = await res.json();
      setConvs(data);
      if (!selected && data.length) setSelected(data[0]);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoadingList(false);
    }
  }

  async function loadThread(convId: number) {
    setLoadingThread(true);
    setErr(null);
    try {
      const res = await fetch(`/api/conversations/${convId}`, { cache: "no-store" });
      if (!res.ok) throw new Error("Failed to load conversation");
      const data = await res.json();
      setMsgs(data.messages as Message[]);
      setSelected(data.conversation as Conversation);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoadingThread(false);
    }
  }

  async function ensureConversationForKol(kolId: number) {
    const existing = convs.find((c) => c.kol_id === kolId);
    if (existing) {
      await loadThread(existing.id);
      return;
    }
    const res = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kol_id: kolId, channel: "dm" }),
    });
    if (!res.ok) throw new Error("Failed to create conversation");
    await loadConvs();
  }

  useEffect(() => {
    loadConvs();
  }, []);

  useEffect(() => {
    if (searchParams?.kol_id) {
      const idNum = Number(searchParams.kol_id);
      if (!Number.isNaN(idNum)) {
        ensureConversationForKol(idNum).catch((e) => setErr(String(e)));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams?.kol_id]);

  /* ---------------- actions ---------------- */
  async function sendMessage(direction: "out" | "in") {
    if (!selected || !newMsg.trim()) return;
    const res = await fetch(`/api/conversations/${selected.id}/messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ direction, body: newMsg.trim() }),
    });
    if (!res.ok) {
      setErr("Failed to send message");
      return;
    }
    setNewMsg("");
    await loadThread(selected.id);
    await loadConvs();
  }

  async function updateStatus(status: Conversation["status"]) {
    if (!selected) return;
    const res = await fetch(`/api/conversations/${selected.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status }),
    });
    if (!res.ok) {
      setErr("Failed to update status");
      return;
    }
    await loadThread(selected.id);
    await loadConvs();
  }

  /* ---------------- NEW: template insert ---------------- */
  const [tplId, setTplId] = useState<string>("opening");
  function insertTemplate() {
    const tpl = TEMPLATES.find((t) => t.id === tplId);
    if (!tpl) return;
    const filled = fillTemplate(tpl.text, selected);
    setNewMsg((prev) => (prev ? prev + "\n\n" + filled : filled));
  }

  const statusColor: Record<Conversation["status"], string> = {
    contacted: "#d1fae5",
    negotiating: "#fef3c7",
    confirmed: "#bfdbfe",
    closed: "#e5e7eb",
  };

return (
  <main className="max-w-6xl mx-auto p-6">
    <h1 className="text-2xl font-semibold mb-4">Outreach CRM</h1>

    {err && <div style={{ color: "crimson", marginBottom: 12 }}>Error: {err}</div>}

    <div className="grid grid-cols-12 gap-4">
      {/* Sidebar */}
      <aside className="col-span-4 border rounded-xl p-3 max-h-[75vh] overflow-auto">
        <div className="flex items-center justify-between mb-2">
          <div className="font-medium">Conversations</div>
          <button className="text-sm border rounded px-2 py-1" onClick={loadConvs} disabled={loadingList}>
            {loadingList ? "Loading…" : "Refresh"}
          </button>
        </div>
        {convs.length === 0 && <div className="text-gray-500 text-sm">No conversations yet.</div>}
        <div className="flex flex-col gap-2">
          {convs.map((c) => (
            <button
              key={c.id}
              className={`text-left border rounded-lg p-3 ${selected?.id === c.id ? "bg-gray-50" : ""}`}
              onClick={() => loadThread(c.id)}
            >
              <div className="flex items-center justify-between">
                <div className="font-medium">
                  {c.kol_display_name || c.kol_username || `KOL #${c.kol_id}`}
                </div>
                <span className="text-xs px-2 py-1 rounded-full" style={{ background: statusColor[c.status] }}>
                  {c.status}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                @{c.kol_username} · {c.channel.toUpperCase()} · {c.kol_followers?.toLocaleString() ?? "—"} followers
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {c.last_message_at
                  ? new Date(c.last_message_at).toLocaleString()
                  : new Date(c.created_at).toLocaleString()}
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* Thread */}
      <section className="col-span-8 border rounded-xl p-4 flex flex-col max-h-[75vh]">
        {!selected ? (
          <div className="text-gray-500">Select a conversation</div>
        ) : (
          <>
            <div className="flex items-center justify-between mb-3">
              <div>
                <div className="font-medium text-lg">
                  {selected.kol_display_name || selected.kol_username}{" "}
                  <span className="text-gray-500">@{selected.kol_username}</span>
                </div>
                <div className="text-sm text-gray-600">
                  {selected.channel.toUpperCase()} · {selected.kol_followers?.toLocaleString() ?? "—"} followers
                </div>
              </div>
              <div className="flex gap-2 items-center">
                <label className="text-sm text-gray-600">Status:</label>
                <select
                  value={selected.status}
                  onChange={(e) => updateStatus(e.target.value as Conversation["status"])}
                  className="border rounded px-2 py-1 text-sm"
                >
                  <option value="contacted">contacted</option>
                  <option value="negotiating">negotiating</option>
                  <option value="confirmed">confirmed</option>
                  <option value="closed">closed</option>
                </select>
              </div>
            </div>

            {/* ✅ Flow Stages: Negotiation panel */}
            <div className="w-full mb-3 border rounded-lg p-3 bg-white">
              <div className="font-medium mb-2">Negotiation</div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <input
                  className="border rounded px-2 py-1 text-sm"
                  placeholder="Deliverables (e.g., 1 TikTok + 1 Story)"
                  defaultValue={selected.proposed_deliverables ?? ""}
                  onChange={(e) => (selected.proposed_deliverables = e.target.value)}
                />
                <input
                  className="border rounded px-2 py-1 text-sm"
                  placeholder="Timeline (e.g., within 2 weeks)"
                  defaultValue={selected.proposed_timeline ?? ""}
                  onChange={(e) => (selected.proposed_timeline = e.target.value)}
                />
                <input
                  className="border rounded px-2 py-1 text-sm"
                  placeholder="Price THB (e.g., 30000)"
                  defaultValue={selected.proposed_price_integer ?? ""}
                  onChange={(e) => (selected.proposed_price_integer = Number(e.target.value) || undefined)}
                />
              </div>
              <div className="mt-2 flex gap-2">
                <button
                  className="border rounded px-3 py-1 text-sm"
                  onClick={async () => {
                    if (!selected) return;
                    const body = {
                      status: "negotiating",
                      proposed_deliverables: selected.proposed_deliverables ?? "",
                      proposed_timeline: selected.proposed_timeline ?? "",
                      proposed_price_integer: selected.proposed_price_integer ?? null,
                    };
                    const res = await fetch(`/api/conversations/${selected.id}`, {
                      method: "PATCH",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(body),
                    });
                    if (res.ok) { await loadThread(selected.id); await loadConvs(); }
                  }}
                >
                  Save proposal
                </button>

                <button
                  className="border rounded px-3 py-1 text-sm"
                  onClick={async () => {
                    if (!selected) return;
                    const body = {
                      status: "confirmed",
                      agreed_deliverables: selected.proposed_deliverables ?? "",
                      agreed_timeline: selected.proposed_timeline ?? "",
                      agreed_price_integer: selected.proposed_price_integer ?? null,
                    };
                    const res = await fetch(`/api/conversations/${selected.id}`, {
                      method: "PATCH",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify(body),
                    });
                    if (res.ok) { await loadThread(selected.id); await loadConvs(); }
                  }}
                >
                  Confirm
                </button>
              </div>

              {selected.status === "confirmed" && (
                <div className="mt-2 text-sm text-gray-700">
                  <div><b>Agreed deliverables:</b> {selected["agreed_deliverables"] ?? selected["proposed_deliverables"] ?? "—"}</div>
                  <div><b>Agreed timeline:</b> {selected["agreed_timeline"] ?? selected["proposed_timeline"] ?? "—"}</div>
                  <div>
                    <b>Agreed price:</b>{" "}
                    {selected["agreed_price_integer"]
                      ? `THB ${Number(selected["agreed_price_integer"]).toLocaleString()}`
                      : "—"}
                  </div>
                </div>
              )}
            </div>

            <div className="flex-1 overflow-auto border rounded-lg p-3 bg-gray-50">
              {loadingThread && <div className="text-sm">Loading thread…</div>}
              {msgs.length === 0 && !loadingThread && (
                <div className="text-gray-500 text-sm">No messages yet.</div>
              )}
              <div className="flex flex-col gap-2">
                {msgs.map((m) => (
                  <div
                    key={m.id}
                    className={`max-w-[75%] p-2 rounded-md ${
                      m.direction === "out" ? "self-end bg-white border" : "self-start bg-blue-50 border-blue-100 border"
                    }`}
                  >
                    <div className="text-sm whitespace-pre-wrap">{m.body}</div>
                    <div className="text-[11px] text-gray-500 mt-1">
                      {new Date(m.created_at).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Template controls */}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <label className="text-sm text-gray-600">Templates:</label>
              <select
                value={tplId}
                onChange={(e) => setTplId(e.target.value)}
                className="border rounded px-2 py-1 text-sm"
              >
                {TEMPLATES.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
              <button className="border rounded px-3 py-1 text-sm" onClick={insertTemplate}>
                Insert
              </button>
            </div>

            <div className="mt-2 flex gap-2">
              <textarea
                className="border rounded-lg p-2 w-full text-sm"
                rows={3}
                placeholder="Type a message…"
                value={newMsg}
                onChange={(e) => setNewMsg(e.target.value)}
              />
              <div className="flex flex-col gap-2">
                <button
                  className="border rounded px-3 py-2 text-sm"
                  onClick={() => sendMessage("out")}
                  disabled={!newMsg.trim()}
                >
                  Send (out)
                </button>
                <button
                  className="border rounded px-3 py-2 text-sm"
                  onClick={() => sendMessage("in")}
                  disabled={!newMsg.trim()}
                  title="Log an incoming reply (for demo)"
                >
                  Log reply (in)
                </button>
              </div>
            </div>
          </>
        )}
      </section>
    </div>
  </main>
);
}