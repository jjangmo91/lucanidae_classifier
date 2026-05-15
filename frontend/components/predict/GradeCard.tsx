"use client";

import { cn } from "@/lib/utils";
import { ALL_SPECIES, SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

const GRADE_THRESHOLDS = [
  { name: "딱린이",    min: 0,   max: 9   },
  { name: "표본수집가", min: 10,  max: 29  },
  { name: "주간채집러", min: 30,  max: 59  },
  { name: "야간채집러", min: 60,  max: 119 },
  { name: "루카나이더", min: 120, max: 199 },
  { name: "6티어",     min: 200, max: Infinity },
];

const GRADE_COLOR: Record<string, string> = {
  "딱린이":    "from-zinc-600   to-zinc-400",
  "표본수집가": "from-emerald-600 to-emerald-400",
  "주간채집러": "from-sky-600    to-sky-400",
  "야간채집러": "from-indigo-600  to-indigo-400",
  "루카나이더": "from-violet-600  to-violet-400",
  "6티어":     "from-amber-600   to-amber-400",
};

const SPECIALTY_INFO: Record<string, { label: string; icon: string; hint: (s: Stats) => string }> = {
  "레어헌터":   { label: "레어헌터",   icon: "💎", hint: s => `S·A종 비율 ${Math.round(s.rare_ratio * 100)}% (목표 60%)` },
  "분류학자":   { label: "분류학자",   icon: "🔬", hint: s => `교정 ${s.correction_count}회 (목표 20회)` },
  "도감탐험가": { label: "도감탐험가", icon: "🗺️", hint: s => `${s.species_count}종 발견 (목표 12종)` },
  "채집왕":     { label: "채집왕",     icon: "👑", hint: s => `${s.total_uploads}건 업로드` },
};

interface Stats {
  score:            number;
  grade:            string;
  specialty:        string | null;
  total_uploads:    number;
  found_species:    string[];
  rare_ratio:       number;
  correction_count: number;
  species_count:    number;
}

export function GradeCard({ stats }: { stats: Stats }) {
  const tier    = GRADE_THRESHOLDS.find(t => stats.grade === t.name) ?? GRADE_THRESHOLDS[0];
  const isMax   = stats.grade === "6티어";
  const progress = isMax ? 100
    : Math.round(((stats.score - tier.min) / (tier.max - tier.min + 1)) * 100);
  const nextGrade = GRADE_THRESHOLDS[GRADE_THRESHOLDS.findIndex(t => t.name === stats.grade) + 1];
  const ptsLeft   = isMax ? 0 : (nextGrade?.min ?? 200) - stats.score;

  const displayGrade = stats.specialty ?? stats.grade;
  const barColor     = GRADE_COLOR[stats.grade] ?? GRADE_COLOR["6티어"];
  const specInfo     = stats.specialty ? SPECIALTY_INFO[stats.specialty] : null;

  return (
    <div className="p-4 rounded-xl border border-white/10 bg-card/50 space-y-4">
      {/* 등급 + 점수 */}
      <div className="flex items-center justify-between">
        <div>
          <p className="text-xs text-muted-foreground font-mono uppercase tracking-widest">현재 등급</p>
          <p className="text-2xl font-black mt-0.5">
            {specInfo && <span className="mr-1">{specInfo.icon}</span>}
            {displayGrade}
          </p>
        </div>
        <div className="text-right">
          <p className="text-3xl font-black text-amber-400">{stats.score.toLocaleString()}</p>
          <p className="text-xs text-muted-foreground">pts</p>
        </div>
      </div>

      {/* 진행 바 */}
      <div className="space-y-1.5">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>{stats.grade}</span>
          <span>{isMax ? "최고 등급" : nextGrade?.name}</span>
        </div>
        <div className="h-2.5 rounded-full bg-white/10 overflow-hidden">
          <div
            className={cn("h-full rounded-full bg-gradient-to-r transition-all duration-700", barColor)}
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>
        {!isMax && (
          <p className="text-xs text-muted-foreground text-right">
            다음 등급까지 <span className="text-amber-400 font-bold">{ptsLeft}pts</span>
          </p>
        )}
      </div>

      {/* 직업 달성 조건 (200pts 이상 or 근접) */}
      {(stats.score >= 150 || isMax) && !stats.specialty && (
        <div className="pt-2 border-t border-white/5 space-y-2">
          <p className="text-xs text-muted-foreground font-mono">직업 달성 조건</p>
          <div className="grid grid-cols-2 gap-2">
            {Object.entries(SPECIALTY_INFO).map(([key, info]) => (
              <div key={key} className="text-xs p-2 rounded-lg bg-white/5 border border-white/5">
                <span className="mr-1">{info.icon}</span>
                <span className="font-semibold">{info.label}</span>
                <p className="text-muted-foreground mt-0.5">{info.hint(stats)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 달성한 직업 힌트 */}
      {specInfo && (
        <div className="pt-2 border-t border-white/5">
          <p className="text-xs text-muted-foreground">{specInfo.hint(stats)}</p>
        </div>
      )}
    </div>
  );
}

/* ── 16종 도감 격자 ─────────────────────────────────────────────── */

const RARITY_TIER: Record<number, { label: string; color: string }> = {
  1: { label: "C",  color: "text-slate-400  border-slate-500/40  bg-slate-500/10"  },
  2: { label: "B",  color: "text-green-400  border-green-500/40  bg-green-500/10"  },
  3: { label: "A",  color: "text-amber-400  border-amber-500/40  bg-amber-500/10"  },
  4: { label: "S",  color: "text-red-400    border-red-500/40    bg-red-500/10"    },
};

export function PokedexGrid({ foundKeys }: { foundKeys: string[] }) {
  const foundSet = new Set(foundKeys);
  const found    = ALL_SPECIES.filter(s => foundSet.has(s.modelKey)).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground font-mono uppercase tracking-widest">종 도감</p>
        <p className="text-xs font-bold">
          <span className="text-amber-400">{found}</span>
          <span className="text-muted-foreground"> / {ALL_SPECIES.length}종</span>
        </p>
      </div>

      {/* 완성도 바 */}
      <div className="h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-amber-600 to-amber-400 transition-all duration-700"
          style={{ width: `${(found / ALL_SPECIES.length) * 100}%` }}
        />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {ALL_SPECIES.map(sp => {
          const isFound = foundSet.has(sp.modelKey);
          const tier    = RARITY_TIER[sp.rarity];
          return (
            <a
              key={sp.slug}
              href={`/species/${sp.slug}`}
              className={cn(
                "relative rounded-lg border p-2 text-center transition-all duration-200",
                isFound
                  ? "bg-card hover:border-amber-500/40 hover:bg-amber-500/5 border-white/10"
                  : "border-white/5 bg-white/3 opacity-40 grayscale hover:opacity-60"
              )}
            >
              {/* 희귀도 배지 */}
              <span className={cn(
                "absolute top-1 right-1 text-[9px] font-black px-1 rounded border",
                tier.color
              )}>
                {tier.label}
              </span>
              <p className={cn(
                "text-[11px] sm:text-[10px] font-semibold leading-tight mt-1",
                isFound ? "text-foreground" : "text-muted-foreground"
              )}>
                {isFound ? sp.ko : "???"}
              </p>
            </a>
          );
        })}
      </div>
    </div>
  );
}
