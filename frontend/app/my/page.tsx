"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface HistoryEntry {
  prediction_id: string;
  result_type: "identified" | "low_confidence" | "uncertain" | "foreign" | "no_beetle";
  species: string;
  species_ko: string | null;
  confidence: number;
  is_ood: boolean;
  analyzed_at: string;
}

const RESULT_TYPE_LABEL: Record<string, { label: string; cls: string }> = {
  identified:     { label: "동정 완료",  cls: "text-amber-400 border-amber-500/40 bg-amber-500/10" },
  low_confidence: { label: "신뢰도 낮음", cls: "text-yellow-400 border-yellow-500/40 bg-yellow-500/10" },
  uncertain:      { label: "인식 불가",  cls: "text-slate-400 border-slate-500/30 bg-slate-500/8" },
  foreign:        { label: "한국 외 종", cls: "text-sky-400 border-sky-500/30 bg-sky-500/8" },
  no_beetle:      { label: "미감지",     cls: "text-slate-400 border-slate-500/30 bg-slate-500/8" },
};

function formatDate(iso: string) {
  const d = new Date(iso);
  return `${d.getMonth() + 1}.${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

export default function MyPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loaded, setLoaded]   = useState(false);

  useEffect(() => {
    const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    const results = ids
      .map((id) => {
        const raw = localStorage.getItem(id);
        return raw ? (JSON.parse(raw) as HistoryEntry) : null;
      })
      .filter(Boolean) as HistoryEntry[];
    setEntries(results);
    setLoaded(true);
  }, []);

  function clearHistory() {
    const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    ids.forEach((id) => localStorage.removeItem(id));
    localStorage.removeItem("prediction_history");
    setEntries([]);
  }

  return (
    <div className="py-8 space-y-6">

      <div className="flex items-end justify-between">
        <div className="space-y-1">
          <p className="text-[10px] font-mono tracking-[0.3em] text-amber-500/45 uppercase">My Collection</p>
          <h1 className="text-2xl font-black">내 도감</h1>
          <p className="text-sm text-muted-foreground">이 기기에서 분류한 사슴벌레 기록</p>
        </div>
        {entries.length > 0 && (
          <button
            onClick={clearHistory}
            className="text-xs text-muted-foreground/50 hover:text-destructive transition-colors"
          >
            기록 초기화
          </button>
        )}
      </div>

      {!loaded ? null : entries.length === 0 ? (
        <div className="py-24 flex flex-col items-center gap-4 text-center">
          <div className="w-20 h-20 rounded-full border border-border flex items-center justify-center specimen-pod">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/lucanus.png" alt="" className="w-14 h-14 object-contain opacity-50" />
          </div>
          <div>
            <p className="font-semibold text-foreground/70">아직 분류한 기록이 없어요</p>
            <p className="text-sm text-muted-foreground mt-1">사진을 올리면 여기에 저장됩니다</p>
          </div>
          <Link
            href="/"
            className="text-sm text-amber-400 border border-amber-500/30 rounded-full px-4 py-2 hover:bg-amber-500/10 transition-colors"
          >
            동정하러 가기
          </Link>
        </div>
      ) : (
        <>
          <p className="text-xs text-muted-foreground font-mono">총 {entries.length}건</p>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {entries.map((e) => {
              const badge   = RESULT_TYPE_LABEL[e.result_type] ?? RESULT_TYPE_LABEL.uncertain;
              const isResult = e.result_type === "identified" || e.result_type === "low_confidence";

              return (
                <Link
                  key={e.prediction_id}
                  href={`/result/${e.prediction_id}`}
                  className="group rounded-xl border border-border bg-card p-4 hover:border-amber-500/30 hover:bg-amber-500/5 transition-all duration-200 space-y-3"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${badge.cls}`}>
                      {badge.label}
                    </span>
                    <span className="text-[10px] font-mono text-muted-foreground/50 shrink-0">
                      {e.analyzed_at ? formatDate(e.analyzed_at) : ""}
                    </span>
                  </div>

                  {isResult ? (
                    <div className="space-y-1.5">
                      <p className="font-black text-lg leading-tight group-hover:text-amber-400 transition-colors">
                        {e.species_ko ?? e.species}
                      </p>
                      <p className="text-[11px] italic text-muted-foreground">
                        {e.species.replace(/_/g, " ")}
                      </p>
                      <div className="flex items-center gap-2 pt-1">
                        <p className="text-sm font-black tabular-nums text-amber-400">
                          {(e.confidence * 100).toFixed(1)}%
                        </p>
                        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full bg-amber-400 rounded-full"
                            style={{ width: `${e.confidence * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">결과 없음</p>
                  )}
                </Link>
              );
            })}
          </div>
        </>
      )}

      <style>{`
        .specimen-pod {
          background: radial-gradient(circle at 40% 38%, #f9f6f0, #e9dfd2);
        }
      `}</style>
    </div>
  );
}
