"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { SPECIES_BY_MODEL_KEY, ALL_SPECIES } from "@/lib/species-data";
import { useAuth } from "@/lib/auth";

interface Prediction {
  prediction_id: string;
  result_type: "identified" | "low_confidence" | "uncertain" | "foreign" | "no_beetle";
  species: string;
  species_ko: string | null;
  confidence: number;
  is_ood: boolean;
  top3: { species: string; confidence: number }[];
  is_fallback: boolean;
}

type FeedbackState = "idle" | "correct_sent" | "correcting" | "corrected" | "error";

const KAKAO_KEY = process.env.NEXT_PUBLIC_KAKAO_APP_KEY ?? "";

const TIER_LABEL: Record<number, string> = { 4: "S", 3: "A", 2: "B", 1: "C" };

// 희귀도 내림차순(S→A→B→C)으로 정렬된 종 목록
const SPECIES_OPTIONS = [...ALL_SPECIES]
  .sort((a, b) => b.rarity - a.rarity)
  .map(sp => ({
    value: sp.modelKey,
    label: `[${TIER_LABEL[sp.rarity]}] ${sp.ko}`,
    rarity: sp.rarity,
  }));

const NON_RESULT_MESSAGE: Record<string, { icon: string; title: string; text: string }> = {
  no_beetle:      { icon: "🔍", title: "감지 실패",     text: "사진에서 사슴벌레를 찾을 수 없어요" },
  uncertain:      { icon: "😕", title: "인식 불가",     text: "더 선명한 사진을 올려주세요" },
  foreign:        { icon: "🌏", title: "한국 외 종",    text: "한국에서 서식하지 않는 종으로 추정됩니다" },
  low_confidence: { icon: "🤔", title: "신뢰도 낮음",   text: "아래 후보 종을 확인해보세요" },
};

export function ResultCard({ predictionId }: { predictionId: string }) {
  const { isLoggedIn } = useAuth();
  const [data, setData]             = useState<Prediction | null>(null);
  const [feedback, setFeedback]     = useState<FeedbackState>("idle");
  const [correction, setCorrection] = useState("");
  const [sex, setSex]               = useState("");
  const [maleForm, setMaleForm]     = useState("");
  const [copied, setCopied]         = useState(false);

  useEffect(() => {
    const raw = sessionStorage.getItem(predictionId) ?? localStorage.getItem(predictionId);
    if (raw) setData(JSON.parse(raw));
  }, [predictionId]);

  useEffect(() => {
    if (!KAKAO_KEY) return;
    const script = document.createElement("script");
    script.src = "https://developers.kakao.com/sdk/js/kakao.min.js";
    script.onload = () => {
      const K = (window as any).Kakao;
      if (K && !K.isInitialized()) K.init(KAKAO_KEY);
    };
    document.head.appendChild(script);
  }, []);

  async function handleShare() {
    if (!data) return;
    const url  = `${window.location.origin}/result/${predictionId}`;
    const text = `AI가 ${data.species_ko ?? data.species.replace(/_/g, " ")}로 동정했어요! 신뢰도 ${(data.confidence * 100).toFixed(1)}%`;
    if (navigator.share) {
      try { await navigator.share({ title: "비틀덱스 동정 결과", text, url }); } catch { /* 취소 */ }
    } else {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  function handleKakaoShare() {
    if (!data) return;
    const K = (window as any).Kakao;
    if (!K?.isInitialized()) return;
    const url = `${window.location.origin}/result/${predictionId}`;
    K.Share.sendDefault({
      objectType: "feed",
      content: {
        title:       `${data.species_ko ?? data.species.replace(/_/g, " ")} 동정 완료`,
        description: `AI 신뢰도 ${(data.confidence * 100).toFixed(1)}% · 비틀덱스 BeetleDex`,
        imageUrl:    `${window.location.origin}/lucanus.png`,
        link:        { mobileWebUrl: url, webUrl: url },
      },
    });
  }

  async function sendFeedback(isCorrect: boolean, correctSpecies?: string) {
    try {
      const res = await fetch(`/api/v1/predictions/${predictionId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          is_correct:      isCorrect,
          correct_species: correctSpecies ?? null,
          sex:             sex || null,
          male_form:       maleForm || null,
        }),
      });
      if (!res.ok) throw new Error();
      setFeedback(isCorrect ? "correct_sent" : "corrected");
    } catch {
      setFeedback("error");
    }
  }

  /* Loading */
  if (!data) {
    return (
      <div className="rounded-2xl border border-border bg-card p-10 text-center space-y-3">
        <div className="w-16 h-16 mx-auto rounded-full border-2 border-amber-500/40 flex items-center justify-center specimen-pod">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/lucanus.png" alt="" className="w-12 h-12 object-contain animate-float" />
        </div>
        <p className="text-sm text-muted-foreground">결과를 불러오는 중...</p>
      </div>
    );
  }

  /* Non-standard results */
  if (data.result_type !== "identified" && data.result_type !== "low_confidence") {
    const msg = NON_RESULT_MESSAGE[data.result_type];
    return (
      <div className="rounded-2xl border border-border bg-card p-8 text-center space-y-3">
        <p className="text-5xl">{msg?.icon ?? "❓"}</p>
        <div>
          <p className="font-bold text-base">{msg?.title}</p>
          <p className="text-sm text-muted-foreground mt-1">{msg?.text}</p>
        </div>
        <Link href="/" className="inline-block mt-2 text-xs text-amber-400 border border-amber-500/30 rounded-full px-4 py-1.5 hover:bg-amber-500/10 transition-colors">
          다시 시도하기
        </Link>
      </div>
    );
  }

  const spInfo       = SPECIES_BY_MODEL_KEY[data.species];
  const rarityPts    = spInfo?.rarity ?? 0;
  const confidencePct = (data.confidence * 100).toFixed(1);
  const isLow        = data.result_type === "low_confidence";

  return (
    <div className={`rounded-2xl border bg-card overflow-hidden ${isLow ? "border-amber-500/20" : "border-amber-500/35 glow-amber"}`}>

      {/* Header bar */}
      <div className="flex items-center justify-between px-5 py-2.5 border-b border-border/60 bg-amber-500/5">
        <span className="text-[10px] font-mono tracking-[0.25em] text-amber-500/60 uppercase">
          동정 완료
        </span>
        <span className="text-[10px] font-mono text-muted-foreground/50">
          {predictionId.slice(0, 8).toUpperCase()}
        </span>
      </div>

      <div className="p-5 space-y-5">

        {/* Species identity */}
        <div className="text-center space-y-1.5">
          {isLow && (
            <p className="text-[11px] text-amber-400/70 font-medium">신뢰도 낮음 — 후보 종을 확인해보세요</p>
          )}
          <p className="text-3xl font-black tracking-tight leading-tight">
            {data.species_ko ?? data.species}
          </p>
          <p className="text-xs italic text-muted-foreground tracking-wide">
            {data.species.replace(/_/g, " ")}
          </p>
        </div>

        {/* Confidence display */}
        <div className="flex flex-col items-center gap-2">
          <p className="text-5xl font-black tabular-nums text-amber-400 text-glow leading-none">
            {confidencePct}
            <span className="text-2xl font-bold text-amber-400/60">%</span>
          </p>
          <div className="w-full max-w-xs h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-amber-400 rounded-full"
              style={{
                width: `${confidencePct}%`,
                animation: "fill-bar 0.8s ease-out",
              }}
            />
          </div>
          <p className="text-[11px] text-muted-foreground font-mono tracking-wider">AI 신뢰도</p>
        </div>

        {/* Species page link */}
        {spInfo && (
          <div className="text-center">
            <Link
              href={`/species/${spInfo.slug}`}
              className="inline-flex items-center gap-1.5 text-xs text-amber-400 border border-amber-500/30 rounded-full px-4 py-1.5 hover:bg-amber-500/10 transition-colors"
            >
              종 상세 정보 →
            </Link>
          </div>
        )}

        {/* Top 3 candidates */}
        <div className="space-y-2 border-t border-border/50 pt-4">
          <p className="text-[11px] text-muted-foreground font-mono tracking-widest uppercase">후보 종</p>
          {data.top3.map((p, i) => (
            <div key={p.species} className="flex items-center gap-2.5">
              <span className={`text-xs font-black w-4 text-center tabular-nums ${i === 0 ? "text-amber-400" : "text-muted-foreground/50"}`}>
                {i + 1}
              </span>
              <span className="text-sm flex-1 truncate text-foreground/80">
                {p.species.replace(/_/g, " ")}
              </span>
              <span className={`text-xs font-bold tabular-nums ${i === 0 ? "text-amber-400" : "text-muted-foreground"}`}>
                {(p.confidence * 100).toFixed(1)}%
              </span>
              <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full ${i === 0 ? "bg-amber-400" : "bg-muted-foreground/40"}`}
                  style={{ width: `${p.confidence * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Feedback */}
        <div className="border-t border-border/50 pt-4 space-y-3">
          {feedback === "idle" && (
            <>
              <p className="text-[11px] text-muted-foreground font-mono uppercase tracking-widest">
                AI 동정이 맞나요?
              </p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="flex-1 text-sm border-emerald-700/40 text-emerald-400 hover:bg-emerald-900/20 hover:border-emerald-600/50"
                  onClick={() => sendFeedback(true)}
                >
                  ✓ 맞아요{isLoggedIn && rarityPts > 0 ? ` +${rarityPts}pts` : ""}
                </Button>
                <Button
                  variant="outline"
                  className="flex-1 text-sm border-border hover:border-red-500/30 hover:bg-red-500/5 hover:text-red-400"
                  onClick={() => setFeedback("correcting")}
                >
                  ✗ 아니에요
                </Button>
              </div>
              {!isLoggedIn && (
                <p className="text-[10px] text-muted-foreground/60 text-center">
                  로그인하면 피드백 포인트가 적립됩니다
                </p>
              )}
            </>
          )}

          {feedback === "correcting" && (
            <div className="space-y-2.5">
              <p className="text-[11px] text-muted-foreground font-mono uppercase tracking-widest">
                실제 종을 알려주세요
              </p>

              <select
                className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
                value={correction}
                onChange={(e) => setCorrection(e.target.value)}
              >
                <option value="">종 선택 (S→A→B→C 순)</option>
                {SPECIES_OPTIONS.map((s) => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>

              <div className="flex gap-2">
                <select
                  className="flex-1 border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
                  value={sex}
                  onChange={(e) => setSex(e.target.value)}
                >
                  <option value="">성별 (선택)</option>
                  <option value="male">수컷</option>
                  <option value="female">암컷</option>
                  <option value="unknown">모름</option>
                </select>

                {sex === "male" && (
                  <select
                    className="flex-1 border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
                    value={maleForm}
                    onChange={(e) => setMaleForm(e.target.value)}
                  >
                    <option value="">형태 (선택)</option>
                    <option value="major">대형 (뿔 큼)</option>
                    <option value="minor">소형 (뿔 작음)</option>
                    <option value="intermediate">중간형</option>
                  </select>
                )}
              </div>

              <p className="text-[10px] text-muted-foreground/50">
                교정 정보는 AI 학습 데이터로 활용됩니다
              </p>

              <div className="flex gap-2">
                <Button
                  className="flex-1 bg-amber-500 hover:bg-amber-400 text-amber-950 font-bold"
                  onClick={() => sendFeedback(false, correction)}
                  disabled={!correction.trim()}
                >
                  제출
                </Button>
                <Button variant="outline" className="flex-1 border-border" onClick={() => setFeedback("idle")}>
                  취소
                </Button>
              </div>
            </div>
          )}

          {feedback === "correct_sent" && (
            <div className="text-center space-y-1.5 py-1">
              <p className="text-emerald-400 font-bold">✓ 피드백 감사합니다!</p>
              {isLoggedIn && rarityPts > 0 ? (
                <p className="text-sm text-amber-400 font-black">+{rarityPts} pts 적립</p>
              ) : !isLoggedIn ? (
                <p className="text-xs text-muted-foreground">
                  <Link href="/my" className="text-amber-400 hover:underline">로그인</Link>하면 포인트가 쌓여요
                </p>
              ) : null}
            </div>
          )}

          {feedback === "corrected" && (
            <div className="text-center space-y-1 py-1">
              <p className="text-amber-400 font-bold">교정 정보가 저장됐어요</p>
              <p className="text-xs text-muted-foreground">AI 학습에 반영됩니다. 감사합니다!</p>
            </div>
          )}

          {feedback === "error" && (
            <div className="text-center space-y-2 py-1">
              <p className="text-sm text-destructive">저장에 실패했어요</p>
              <button
                className="text-xs text-amber-400 hover:underline"
                onClick={() => setFeedback("idle")}
              >
                다시 시도하기
              </button>
            </div>
          )}
        </div>

        {/* Share */}
        <div className="flex gap-2 border-t border-border/50 pt-4">
          <Button
            variant="outline"
            className="flex-1 text-sm border-border hover:border-amber-500/30 hover:bg-amber-500/5"
            onClick={handleShare}
          >
            {copied ? "✓ 복사됨" : "📤 공유하기"}
          </Button>
          {KAKAO_KEY && (
            <Button
              variant="outline"
              className="flex-1 text-sm border-yellow-600/40 text-yellow-400 hover:bg-yellow-900/20 hover:border-yellow-500/50"
              onClick={handleKakaoShare}
            >
              💬 카카오
            </Button>
          )}
        </div>
      </div>

      <style>{`
        .specimen-pod {
          background: radial-gradient(circle at 40% 38%, #f9f6f0, #e9dfd2);
        }
        @keyframes fill-bar {
          from { width: 0; }
        }
      `}</style>
    </div>
  );
}
