"use client";

import { useEffect, useState, useCallback } from "react";
import { cn } from "@/lib/utils";
import { SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Card {
  id: string;
  image_url: string | null;
  species: string | null;
  confidence: number;
  result_type: string | null;
  upload_time: string | null;
  user_id: string | null;
}

const RESULT_LABEL: Record<string, string> = {
  identified:      "동정 완료",
  low_confidence:  "저신뢰도",
  uncertain:       "불확실",
  foreign:         "외래종",
  no_beetle:       "사슴벌레 아님",
};

const RARITY_BADGE: Record<number, { label: string; cls: string }> = {
  1: { label: "C", cls: "text-slate-300  bg-slate-500/20  border-slate-500/40" },
  2: { label: "B", cls: "text-green-300  bg-green-500/20  border-green-500/40" },
  3: { label: "A", cls: "text-amber-300  bg-amber-500/20  border-amber-500/40" },
  4: { label: "S", cls: "text-red-300    bg-red-500/20    border-red-500/40 animate-pulse" },
};

type SortKey    = "newest" | "oldest" | "confidence";
type RarityKey  = 0 | 1 | 2 | 3 | 4;  // 0 = 전체

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: "newest",     label: "최신순" },
  { value: "oldest",     label: "오래된순" },
  { value: "confidence", label: "신뢰도순" },
];

const RARITY_OPTIONS: { value: RarityKey; label: string }[] = [
  { value: 0, label: "전체" },
  { value: 4, label: "S" },
  { value: 3, label: "A" },
  { value: 2, label: "B" },
  { value: 1, label: "C" },
];

const RARITY_PILL: Record<number, string> = {
  0: "border-white/15 text-muted-foreground hover:border-amber-500/30 hover:text-foreground",
  4: "border-red-500/40    text-red-300",
  3: "border-amber-500/40  text-amber-300",
  2: "border-green-500/40  text-green-300",
  1: "border-slate-500/40  text-slate-300",
};

export default function GalleryPage() {
  const [cards, setCards]     = useState<Card[]>([]);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset]   = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [sort, setSort]       = useState<SortKey>("newest");
  const [rarity, setRarity]   = useState<RarityKey>(0);

  const load = useCallback(async (o: number, s: SortKey, r: RarityKey) => {
    setLoading(true);
    const params = new URLSearchParams({ limit: "30", offset: String(o), sort: s });
    if (r !== 0) params.set("rarity", String(r));
    const res  = await fetch(`${API}/api/v1/gallery?${params}`);
    const data: Card[] = await res.json();
    if (o === 0) setCards(data);
    else setCards(prev => [...prev, ...data]);
    setHasMore(data.length === 30);
    setLoading(false);
  }, []);

  // 필터/정렬 변경 시 처음부터 재로드
  useEffect(() => {
    setOffset(0);
    load(0, sort, rarity);
  }, [sort, rarity, load]);

  function changeSort(s: SortKey) {
    if (s === sort) return;
    setSort(s);
  }

  function changeRarity(r: RarityKey) {
    if (r === rarity) return;
    setRarity(r);
  }

  return (
    <div className="max-w-5xl mx-auto space-y-5">

      {/* 헤더 */}
      <div>
        <h1 className="text-2xl font-black text-amber-400">커뮤니티 갤러리</h1>
        <p className="text-sm text-muted-foreground mt-1">모든 유저가 올린 사슴벌레 사진</p>
      </div>

      {/* 필터 + 정렬 바 */}
      <div className="flex flex-wrap items-center gap-3">

        {/* 희귀도 필터 */}
        <div className="flex items-center gap-1.5">
          {RARITY_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => changeRarity(opt.value)}
              className={cn(
                "px-3 py-1 rounded-full text-xs font-bold border transition-all",
                rarity === opt.value
                  ? opt.value === 0
                    ? "bg-amber-500/20 border-amber-500/50 text-amber-400"
                    : `${RARITY_PILL[opt.value]} bg-white/8`
                  : `${RARITY_PILL[opt.value]} bg-transparent opacity-60 hover:opacity-100`
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* 구분선 */}
        <div className="hidden sm:block w-px h-4 bg-white/10" />

        {/* 정렬 */}
        <div className="flex items-center gap-1">
          {SORT_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => changeSort(opt.value)}
              className={cn(
                "px-3 py-1 rounded-full text-xs font-medium border transition-all",
                sort === opt.value
                  ? "bg-white/8 border-white/20 text-foreground"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-white/10"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* 그리드 */}
      {cards.length === 0 && !loading ? (
        <p className="text-center text-muted-foreground py-20">
          {rarity !== 0 ? "해당 희귀도 사진이 없습니다" : "아직 업로드된 사진이 없습니다"}
        </p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {cards.map(card => {
            const spInfo = card.species ? SPECIES_BY_MODEL_KEY[card.species] : null;
            const badge  = spInfo ? RARITY_BADGE[spInfo.rarity] : null;
            return (
              <a key={card.id} href={`/result/${card.id}`} className="group block">
                <div className="relative aspect-square rounded-xl overflow-hidden bg-white/5 border border-white/10 group-hover:border-amber-500/40 transition-all">
                  {card.image_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={card.image_url}
                      alt={card.species ?? ""}
                      className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-muted-foreground text-xs">
                      이미지 없음
                    </div>
                  )}
                  {badge && (
                    <span className={`absolute top-1.5 right-1.5 text-[10px] font-black px-1.5 py-0.5 rounded border backdrop-blur-sm ${badge.cls}`}>
                      {badge.label}
                    </span>
                  )}
                </div>
                <p className="mt-1 text-xs text-foreground/80 truncate">
                  {spInfo?.ko ?? card.species?.replace(/_/g, " ") ?? (RESULT_LABEL[card.result_type ?? ""] ?? "—")}
                </p>
                <p className="text-xs text-muted-foreground">
                  {card.confidence ? `${(card.confidence * 100).toFixed(1)}%` : ""}
                </p>
              </a>
            );
          })}
        </div>
      )}

      {loading && (
        <div className="flex justify-center py-8">
          <div className="w-6 h-6 rounded-full border-2 border-amber-500/30 border-t-amber-400 animate-spin" />
        </div>
      )}

      {!loading && hasMore && (
        <div className="flex justify-center pt-2">
          <button
            onClick={() => { const o = offset + 30; setOffset(o); load(o, sort, rarity); }}
            className="px-6 py-2 rounded-lg border border-amber-500/30 text-amber-400 hover:bg-amber-500/10 transition-all text-sm"
          >
            더 보기
          </button>
        </div>
      )}
    </div>
  );
}
