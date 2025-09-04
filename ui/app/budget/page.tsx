"use client";
import { useState } from "react";

type PlanItem = {
  kol_id: number;
  username: string;
  display_name?: string | null;
  item: string;
  price: number;
  est_reach: number;
  match_score: number;
  roi: number;
  rationale: string;
};

type PlanResponse = {
  brief?: string | null;
  total_budget: number;
  total_spend: number;
  est_total_reach: number;
  items: PlanItem[];
};

export default function BudgetPage() {
  const [budget, setBudget] = useState<number>(150000);
  const [minReach, setMinReach] = useState<number | "">("");
  const [categories, setCategories] = useState<string>(""); // comma-separated
  const [minTh, setMinTh] = useState<number | "">("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [plan, setPlan] = useState<PlanResponse | null>(null);
  const [brief, setBrief] = useState<string>("");
  const [topK, setTopK] = useState<number>(100);

  async function runPlan() {
    setErr(null);
    setPlan(null);
    setLoading(true);
    try {
      const payload: any = { total_budget: Number(budget), top_k:topK };
      if (brief.trim()) payload.brief = brief.trim();
      if (minReach !== "" && !Number.isNaN(Number(minReach))) payload.min_reach = Number(minReach);
      const cats = categories
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      if (cats.length) payload.categories = cats;
      if (minTh !== "" && !Number.isNaN(Number(minTh))) payload.min_th_audience = Number(minTh);

      const res = await fetch("/api/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || "Plan failed");
      }
      const data: PlanResponse = await res.json();
      setPlan(data);
    } catch (e: any) {
      setErr(e.message?.replace(/^"+|"+$/g, "") || String(e));
    } finally {
      setLoading(false);
    }
  }

  function exportCSV() {
    if (!plan) return;
    const header = ["kol_id", "username", "display_name", "item", "price", "est_reach", "roi", "rationale"];
    const rows = plan.items.map((it) => [
      it.kol_id,
      it.username,
      it.display_name ?? "",
      it.item,
      it.price,
      it.est_reach,
      it.roi,
      it.rationale.replace(/\n/g, " ").replace(/,/g, ";"),
    ]);
    const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "plan.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="max-w-5xl mx-auto p-6">
      <h1 className="text-2xl font-semibold mb-4">Budget Optimizer</h1>
    
        <div className="col-span-1 md:col-span-4">
        <label className="block text-sm text-gray-600 mb-1">Brief (used for matching)</label>
        <textarea
            className="border rounded px-2 py-1 w-full"
            rows={3}
            placeholder="e.g., Thai beauty creators who post skincare routines"
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
        />
        </div>

        <div>
        <label className="block text-sm text-gray-600 mb-1">Shortlist size (Top K)</label>
        <input
            type="number"
            className="border rounded px-2 py-1 w-full"
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Number(e.target.value) || 50))}
            min={1}
        />
        </div>
      <div className="border rounded-xl p-4 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <label className="block text-sm text-gray-600 mb-1">Total budget (THB)</label>
            <input
              type="number"
              className="border rounded px-2 py-1 w-full"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value) || 0)}
              min={0}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">Min total reach (optional)</label>
            <input
              type="number"
              className="border rounded px-2 py-1 w-full"
              value={minReach}
              onChange={(e) => setMinReach(e.target.value === "" ? "" : Number(e.target.value))}
              min={0}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">Categories (comma separated)</label>
            <input
              type="text"
              className="border rounded px-2 py-1 w-full"
              placeholder="beauty, skincare"
              value={categories}
              onChange={(e) => setCategories(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 mb-1">Min TH audience % (optional)</label>
            <input
              type="number"
              className="border rounded px-2 py-1 w-full"
              value={minTh}
              onChange={(e) => setMinTh(e.target.value === "" ? "" : Number(e.target.value))}
              min={0}
              max={100}
            />
          </div>
        </div>

        <div className="mt-3 flex gap-2">
          <button
            onClick={runPlan}
            className="border rounded px-3 py-2 text-sm"
            disabled={loading || budget <= 0}
          >
            {loading ? "Running…" : "Run plan"}
          </button>
          {plan && (
            <button onClick={exportCSV} className="border rounded px-3 py-2 text-sm">
              Export CSV
            </button>
          )}
        </div>
      </div>

      {err && (
        <div className="text-red-600 text-sm mb-3">
          {err}
        </div>
      )}

      {plan && (
        <div className="border rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="font-medium">Suggested Allocation</div>
            <div className="text-sm text-gray-700">
              <b>Total spend:</b> THB {plan.total_spend.toLocaleString()} ·{" "}
              <b>Est. total reach:</b> {plan.est_total_reach.toLocaleString()}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left border-b">
                  <th className="py-2 pr-4">KOL</th>
                  <th className="py-2 pr-4">Item</th>
                  <th className="py-2 pr-4">Price (THB)</th>
                  <th className="py-2 pr-4">Est. Reach</th>
                  <th className="py-2 pr-4">ROI (reach/THB)</th>
                  <th className="py-2 pr-4">Rationale</th>
                </tr>
              </thead>
              <tbody>
                {plan.items.map((it) => (
                  <tr key={`${it.kol_id}-${it.item}`} className="border-b">
                    <td className="py-2 pr-4">
                      <div className="font-medium">
                        {it.display_name || it.username} <span className="text-gray-500">@{it.username}</span>
                      </div>
                      <div className="text-xs text-gray-500">ID {it.kol_id}</div>
                    </td>
                    <td className="py-2 pr-4">{it.item}</td>
                    <td className="py-2 pr-4">THB {it.price.toLocaleString()}</td>
                    <td className="py-2 pr-4">{it.est_reach.toLocaleString()}</td>
                    <td className="py-2 pr-4">{it.roi.toFixed(2)}</td>
                    <td className="py-2 pr-4 whitespace-pre-wrap">{it.rationale}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </main>
  );
}
