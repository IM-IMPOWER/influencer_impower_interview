import { useEffect, useState } from "react";

export default function KolsPage() {
  const [kols, setKols] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/kols")
      .then((res) => res.json())
      .then((data) => setKols(data));
  }, []);

  return (
    <div style={{ padding: "2rem" }}>
      <h1>KOL Directory</h1>
      {kols.map((k, i) => (
        <div key={i} style={{ border: "1px solid #ccc", margin: "1rem 0", padding: "1rem" }}>
          <h2>{k.handle}</h2>
          <p>Platform: {k.platform}</p>
          <p>Followers: {k.followers}</p>
          <p>Categories: {k.categories.join(", ")}</p>
        </div>
      ))}
    </div>
  );
}