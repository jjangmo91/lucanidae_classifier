"use client";

import { useEffect, useState } from "react";

export function StatsBar() {
  const [today, setToday] = useState<number | null>(null);
  const [total, setTotal] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/v1/stats")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (d) { setToday(d.today_count); setTotal(d.total_count); }
      })
      .catch(() => {});
  }, []);

  if (today === null) return null;

  return (
    <div className="flex items-center gap-4 text-center">
      <div className="space-y-0.5">
        <p className="text-lg font-black tabular-nums text-amber-400">{today.toLocaleString()}</p>
        <p className="text-[10px] text-muted-foreground font-mono tracking-wider uppercase">오늘</p>
      </div>
      <div className="w-px h-8 bg-border" />
      <div className="space-y-0.5">
        <p className="text-lg font-black tabular-nums">{total?.toLocaleString()}</p>
        <p className="text-[10px] text-muted-foreground font-mono tracking-wider uppercase">누적</p>
      </div>
    </div>
  );
}
