"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useAuth, authHeader } from "@/lib/auth";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface RankEntry {
  rank:      number;
  user_id:   string;
  username:  string;
  avatar:    string | null;
  grade:     string;
  specialty: string | null;
  score:     number;
}

const GRADE_COLOR: Record<string, string> = {
  "딱린이":    "text-zinc-400   border-zinc-500/30   bg-zinc-500/10",
  "표본수집가": "text-emerald-400 border-emerald-500/30 bg-emerald-500/10",
  "주간채집러": "text-sky-400    border-sky-500/30    bg-sky-500/10",
  "야간채집러": "text-indigo-400  border-indigo-500/30  bg-indigo-500/10",
  "루카나이더": "text-violet-400  border-violet-500/30  bg-violet-500/10",
  "6티어":     "text-amber-400   border-amber-500/30   bg-amber-500/10",
};

const SPECIALTY_ICON: Record<string, string> = {
  "레어헌터":   "💎",
  "분류학자":   "🔬",
  "도감탐험가": "🗺️",
  "채집왕":     "👑",
};

const RANK_MEDAL = ["🥇", "🥈", "🥉"];

export default function RankPage() {
  const { isLoggedIn, userId, backendToken } = useAuth();
  const [entries, setEntries] = useState<RankEntry[]>([]);
  const [myRank,  setMyRank]  = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/v1/ranking?limit=50`)
      .then(r => r.json())
      .then((data: RankEntry[]) => {
        setEntries(data);
        if (userId) {
          const mine = data.find(e => e.user_id === userId);
          setMyRank(mine?.rank ?? null);
        }
        setLoading(false);
      });
  }, [userId]);

  const myEntry = entries.find(e => e.user_id === userId);

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-black text-amber-400">채집가 랭킹</h1>
        <p className="text-sm text-muted-foreground mt-1">희귀종 보너스 포함 누적 점수 순위</p>
      </div>

      {/* 내 순위 요약 (로그인 시) */}
      {isLoggedIn && myEntry && (
        <div className="p-3 rounded-xl border border-amber-500/30 bg-amber-500/5 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-amber-400 font-black text-lg">{myEntry.rank}위</span>
            <span className="text-sm text-muted-foreground">내 현재 순위</span>
          </div>
          <div className="text-right">
            <span className="text-amber-400 font-black">{myEntry.score.toLocaleString()} pts</span>
            {myEntry.rank > 1 && (() => {
              const above = entries.find(e => e.rank === myEntry.rank - 1);
              const gap   = above ? above.score - myEntry.score : 0;
              return gap > 0 ? (
                <p className="text-xs text-muted-foreground">
                  {above?.username}까지 <span className="text-amber-400 font-bold">{gap}pts</span>
                </p>
              ) : null;
            })()}
          </div>
        </div>
      )}

      {loading && <div className="text-center text-muted-foreground py-20">로딩 중…</div>}

      <div className="space-y-2">
        {entries.map((e, i) => {
          const isMe        = e.user_id === userId;
          const displayGrade = e.specialty ?? e.grade;
          const colorClass  = GRADE_COLOR[e.grade] ?? GRADE_COLOR["6티어"];
          const icon        = e.specialty ? SPECIALTY_ICON[e.specialty] : null;
          const aboveEntry  = i > 0 ? entries[i - 1] : null;
          const gap         = aboveEntry ? aboveEntry.score - e.score : null;

          return (
            <a key={e.user_id} href={`/users/${e.user_id}`} className="block group">
              <div className={cn(
                "flex items-center gap-2 sm:gap-4 p-3 rounded-xl border transition-all",
                isMe
                  ? "border-amber-500/40 bg-amber-500/5 shadow-[0_0_12px_rgba(245,158,11,0.08)]"
                  : "border-white/5 bg-card/50 hover:border-amber-500/20 hover:bg-card"
              )}>
                {/* 순위 */}
                <div className={cn(
                  "w-7 text-center font-black text-base shrink-0",
                  e.rank === 1 ? "text-amber-400" :
                  e.rank === 2 ? "text-zinc-300"  :
                  e.rank === 3 ? "text-amber-700"  : "text-muted-foreground"
                )}>
                  {e.rank <= 3 ? RANK_MEDAL[e.rank - 1] : e.rank}
                </div>

                {/* 아바타 */}
                {e.avatar ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={e.avatar} alt="" className="w-8 h-8 rounded-full shrink-0" />
                ) : (
                  <div className="w-8 h-8 rounded-full bg-white/10 shrink-0" />
                )}

                {/* 이름 + 등급 */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1">
                    <p className="font-semibold text-sm truncate">{e.username}</p>
                    {isMe && <span className="text-[10px] text-amber-400 font-bold shrink-0">← 나</span>}
                  </div>
                  <span className={cn("text-[10px] px-1.5 py-0.5 rounded border font-medium", colorClass)}>
                    {icon && <span className="mr-0.5">{icon}</span>}
                    {displayGrade}
                  </span>
                </div>

                {/* 점수 + 갭 */}
                <div className="text-right shrink-0">
                  <p className="font-black text-sm text-amber-400">{e.score.toLocaleString()}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {gap !== null && gap > 0 ? `-${gap}pts` : "pts"}
                  </p>
                </div>
              </div>
            </a>
          );
        })}
      </div>

      {/* 내가 50위 밖이면 별도 표시 */}
      {isLoggedIn && myRank && myRank > 50 && (
        <div className="p-3 rounded-xl border border-white/10 text-center text-sm text-muted-foreground">
          현재 <span className="text-amber-400 font-bold">{myRank}위</span> — 더 많이 올려서 Top 50에 도전하세요!
        </div>
      )}
    </div>
  );
}
