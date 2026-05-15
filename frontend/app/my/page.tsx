"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth, authHeader } from "@/lib/auth";
import { signIn } from "next-auth/react";
import { cn } from "@/lib/utils";
import { GradeCard, PokedexGrid } from "@/components/predict/GradeCard";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface HistoryEntry {
  prediction_id?: string;
  id?: string;
  result_type: string;
  species: string | null;
  species_ko?: string | null;
  confidence: number;
  is_ood?: boolean;
  analyzed_at?: string;
  upload_time?: string | null;
  image_url?: string | null;
}

const RESULT_TYPE_LABEL: Record<string, { label: string; cls: string }> = {
  identified:     { label: "동정 완료",  cls: "text-amber-400 border-amber-500/40 bg-amber-500/10" },
  low_confidence: { label: "신뢰도 낮음", cls: "text-yellow-400 border-yellow-500/40 bg-yellow-500/10" },
  uncertain:      { label: "인식 불가",  cls: "text-slate-400 border-slate-500/30 bg-slate-500/8" },
  foreign:        { label: "한국 외 종", cls: "text-sky-400 border-sky-500/30 bg-sky-500/8" },
  no_beetle:      { label: "미감지",     cls: "text-slate-400 border-slate-500/30 bg-slate-500/8" },
};

const GRADE_COLOR: Record<string, string> = {
  "딱린이":    "text-zinc-400  border-zinc-500/30  bg-zinc-500/10",
  "표본수집가": "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
  "주간채집러": "text-sky-400   border-sky-500/30   bg-sky-500/10",
  "야간채집러": "text-indigo-400 border-indigo-500/30 bg-indigo-500/10",
  "루카나이더": "text-violet-400 border-violet-500/30 bg-violet-500/10",
  "6티어":     "text-amber-400  border-amber-500/30  bg-amber-500/10",
};

const SPECIALTY_ICON: Record<string, string> = {
  "레어헌터":   "💎",
  "분류학자":   "🔬",
  "도감탐험가": "🗺️",
  "채집왕":     "👑",
};

function formatDate(iso: string) {
  const d = new Date(iso);
  return `${d.getMonth() + 1}.${d.getDate()} ${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`;
}

export default function MyPage() {
  const { isLoggedIn, isLoading, username, avatar, grade, specialty, score, backendToken } = useAuth();
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loaded, setLoaded]   = useState(false);
  const [meStats, setMeStats] = useState<any>(null);

  useEffect(() => {
    if (isLoading) return;

    if (isLoggedIn && backendToken) {
      // 유저 stats (등급, 도감 등)
      fetch(`${API}/api/v1/auth/me`, { headers: authHeader(backendToken) })
        .then(r => r.json())
        .then(setMeStats);

      // DB에서 fetch
      fetch(`${API}/api/v1/my/specimens`, { headers: authHeader(backendToken) })
        .then(r => r.json())
        .then((data: any[]) => {
          setEntries(data.map(d => ({
            id:          d.id,
            result_type: d.result_type ?? "uncertain",
            species:     d.species,
            confidence:  d.confidence,
            upload_time: d.upload_time,
            image_url:   d.image_url,
          })));
          setLoaded(true);
        });
    } else {
      // localStorage fallback (비로그인)
      const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") || "[]");
      const results = ids
        .map(id => { const r = localStorage.getItem(id); return r ? JSON.parse(r) : null; })
        .filter(Boolean);
      setEntries(results);
      setLoaded(true);
    }
  }, [isLoggedIn, isLoading, backendToken]);

  function clearHistory() {
    const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") || "[]");
    ids.forEach(id => localStorage.removeItem(id));
    localStorage.removeItem("prediction_history");
    setEntries([]);
  }

  const displayGrade  = specialty ?? grade ?? "";
  const gradeColorCls = GRADE_COLOR[grade ?? ""] ?? GRADE_COLOR["6티어"];
  const specialtyIcon = specialty ? SPECIALTY_ICON[specialty] : null;

  return (
    <div className="max-w-4xl mx-auto py-8 space-y-6">

      {/* 헤더 */}
      <div className="flex flex-wrap items-end justify-between gap-2">
        <div className="space-y-1">
          <p className="text-[10px] font-mono tracking-[0.3em] text-amber-500/45 uppercase">My Collection</p>
          <h1 className="text-2xl font-black">내 도감</h1>
          <p className="text-sm text-muted-foreground">
            {isLoggedIn ? "내가 업로드한 사슴벌레 기록" : "이 기기에서 분류한 사슴벌레 기록"}
          </p>
        </div>
        {!isLoggedIn && entries.length > 0 && (
          <button onClick={clearHistory} className="text-xs text-muted-foreground/50 hover:text-destructive transition-colors">
            기록 초기화
          </button>
        )}
      </div>

      {/* 로그인 유저 프로필 카드 */}
      {isLoggedIn && (
        <div className="flex items-center gap-4 p-4 rounded-xl border border-white/10 bg-card/50">
          {avatar && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={avatar} alt="" className="w-14 h-14 rounded-full" />
          )}
          <div className="flex-1 min-w-0">
            <p className="font-black text-lg">{username}</p>
            <div className="flex items-center gap-2 mt-1">
              <span className={cn("text-xs px-2 py-0.5 rounded border font-semibold", gradeColorCls)}>
                {specialtyIcon && <span className="mr-1">{specialtyIcon}</span>}
                {displayGrade}
              </span>
              <span className="text-xs text-muted-foreground">{score.toLocaleString()} pts</span>
            </div>
          </div>
          <Link href="/rank" className="text-xs text-amber-400 hover:underline shrink-0">
            랭킹 보기 →
          </Link>
        </div>
      )}

      {/* 레벨 진행 카드 + 도감 격자 */}
      {isLoggedIn && meStats && (
        <>
          <GradeCard stats={{
            score:            meStats.score,
            grade:            meStats.grade,
            specialty:        meStats.specialty,
            total_uploads:    meStats.total_uploads,
            found_species:    meStats.found_species ?? [],
            rare_ratio:       meStats.rare_ratio ?? 0,
            correction_count: meStats.correction_count ?? 0,
            species_count:    meStats.species_count ?? 0,
          }} />
          <PokedexGrid foundKeys={meStats.found_species ?? []} />
        </>
      )}

      {/* 비로그인 유도 */}
      {!isLoggedIn && !isLoading && (
        <div className="p-3 rounded-lg border border-amber-500/20 bg-amber-500/5 flex items-center justify-between gap-3">
          <p className="text-sm text-amber-400/80">로그인하면 기록이 영구 저장되고 랭킹에 참여할 수 있어요</p>
          <button
            onClick={() => signIn("google")}
            className="text-xs px-3 py-1.5 rounded border border-amber-500/30 text-amber-400 hover:bg-amber-500/10 transition-all shrink-0"
          >
            로그인
          </button>
        </div>
      )}

      {/* 기록 목록 */}
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
          <Link href="/" className="text-sm text-amber-400 border border-amber-500/30 rounded-full px-4 py-2 hover:bg-amber-500/10 transition-colors">
            동정하러 가기
          </Link>
        </div>
      ) : (
        <>
          <p className="text-xs text-muted-foreground font-mono">총 {entries.length}건</p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {entries.map((e) => {
              const id    = e.prediction_id ?? e.id ?? "";
              const badge = RESULT_TYPE_LABEL[e.result_type] ?? RESULT_TYPE_LABEL.uncertain;
              const isResult = e.result_type === "identified" || e.result_type === "low_confidence";
              const time  = e.analyzed_at ?? e.upload_time;

              return (
                <Link
                  key={id}
                  href={`/result/${id}`}
                  className="group rounded-xl border border-border bg-card hover:border-amber-500/30 hover:bg-amber-500/5 transition-all duration-200 overflow-hidden"
                >
                  {e.image_url && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={e.image_url} alt="" className="w-full h-36 object-cover group-hover:scale-105 transition-transform duration-300" />
                  )}
                  <div className="p-4 space-y-3">
                    <div className="flex items-start justify-between gap-2">
                      <span className={`text-[10px] font-medium px-2 py-0.5 rounded border ${badge.cls}`}>
                        {badge.label}
                      </span>
                      {time && (
                        <span className="text-[10px] font-mono text-muted-foreground/50 shrink-0">
                          {formatDate(time)}
                        </span>
                      )}
                    </div>
                    {isResult ? (
                      <div className="space-y-1.5">
                        <p className="font-black text-lg leading-tight group-hover:text-amber-400 transition-colors">
                          {e.species_ko ?? e.species?.replace(/_/g, " ") ?? "—"}
                        </p>
                        <div className="flex items-center gap-2 pt-1">
                          <p className="text-sm font-black tabular-nums text-amber-400">
                            {(e.confidence * 100).toFixed(1)}%
                          </p>
                          <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                            <div className="h-full bg-amber-400 rounded-full" style={{ width: `${e.confidence * 100}%` }} />
                          </div>
                        </div>
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">결과 없음</p>
                    )}
                  </div>
                </Link>
              );
            })}
          </div>
        </>
      )}

      <style>{`.specimen-pod { background: radial-gradient(circle at 40% 38%, #f9f6f0, #e9dfd2); }`}</style>
    </div>
  );
}
