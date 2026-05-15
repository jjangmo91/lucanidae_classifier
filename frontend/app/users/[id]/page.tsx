"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface SpecimenCard {
  id: string;
  image_url: string | null;
  species: string | null;
  confidence: number;
  result_type: string | null;
  upload_time: string | null;
}

interface Profile {
  user_id:          string;
  username:         string;
  avatar:           string | null;
  grade:            string;
  specialty:        string | null;
  score:            number;
  recent_specimens: SpecimenCard[];
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

const RARITY_BADGE: Record<number, { label: string; cls: string }> = {
  1: { label: "C", cls: "text-slate-300 bg-slate-500/20 border-slate-500/40" },
  2: { label: "B", cls: "text-green-300 bg-green-500/20 border-green-500/40" },
  3: { label: "A", cls: "text-amber-300 bg-amber-500/20 border-amber-500/40" },
  4: { label: "S", cls: "text-red-300   bg-red-500/20   border-red-500/40"   },
};

export default function UserProfilePage() {
  const { id }          = useParams<{ id: string }>();
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) return;
    fetch(`${API}/api/v1/users/${id}`)
      .then(r => {
        if (r.status === 404) { setNotFound(true); setLoading(false); return null; }
        return r.json();
      })
      .then(d => { if (d) { setProfile(d); setLoading(false); } });
  }, [id]);

  if (loading) return <div className="text-center py-32 text-muted-foreground">로딩 중…</div>;
  if (notFound || !profile) return (
    <div className="text-center py-32 space-y-3">
      <p className="text-muted-foreground">존재하지 않는 유저입니다</p>
      <Link href="/rank" className="text-amber-400 text-sm hover:underline">랭킹으로 돌아가기</Link>
    </div>
  );

  const displayGrade = profile.specialty ?? profile.grade;
  const gradeColor   = GRADE_COLOR[profile.grade] ?? GRADE_COLOR["6티어"];
  const icon         = profile.specialty ? SPECIALTY_ICON[profile.specialty] : null;

  return (
    <div className="max-w-3xl mx-auto space-y-8 py-6">

      {/* 프로필 헤더 */}
      <div className="flex items-center gap-4 flex-wrap">
        {profile.avatar ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={profile.avatar} alt="" className="w-20 h-20 rounded-full ring-2 ring-amber-500/20" />
        ) : (
          <div className="w-20 h-20 rounded-full bg-white/10" />
        )}
        <div className="space-y-2">
          <h1 className="text-2xl font-black">{profile.username}</h1>
          <div className="flex items-center gap-3">
            <span className={cn("text-sm px-2.5 py-1 rounded-lg border font-semibold", gradeColor)}>
              {icon && <span className="mr-1.5">{icon}</span>}
              {displayGrade}
            </span>
            <span className="text-amber-400 font-black">{profile.score.toLocaleString()} pts</span>
          </div>
          <Link href="/rank" className="text-xs text-muted-foreground hover:text-amber-400 transition-colors">
            랭킹 보기 →
          </Link>
        </div>
      </div>

      {/* 최근 표본 */}
      {profile.recent_specimens.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-widest">최근 채집</h2>
          <div className="grid grid-cols-3 sm:grid-cols-4 gap-3">
            {profile.recent_specimens.map(s => {
              const spInfo = s.species ? SPECIES_BY_MODEL_KEY[s.species] : null;
              const rarity = spInfo ? RARITY_BADGE[spInfo.rarity] : null;
              return (
                <a key={s.id} href={`/result/${s.id}`} className="group block">
                  <div className="relative aspect-square rounded-xl overflow-hidden bg-white/5 border border-white/10 group-hover:border-amber-500/30 transition-all">
                    {s.image_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={s.image_url} alt="" className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground">이미지 없음</div>
                    )}
                    {rarity && (
                      <span className={`absolute top-1.5 right-1.5 text-[10px] font-black px-1.5 py-0.5 rounded border backdrop-blur-sm ${rarity.cls}`}>
                        {rarity.label}
                      </span>
                    )}
                  </div>
                  <p className="mt-1 text-xs text-foreground/80 truncate">
                    {spInfo?.ko ?? s.species?.replace(/_/g, " ") ?? "—"}
                  </p>
                </a>
              );
            })}
          </div>
        </div>
      )}

      {profile.recent_specimens.length === 0 && (
        <p className="text-center text-muted-foreground py-12">아직 업로드한 사진이 없습니다</p>
      )}
    </div>
  );
}
