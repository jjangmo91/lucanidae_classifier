"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

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

const SPECIES_OPTIONS = [
  { value: "Prosopocoilus_astacoides_blanchardi",  label: "두점박이사슴벌레" },
  { value: "Dorcus_titanus_castanicolor",           label: "넓적사슴벌레" },
  { value: "Lucanus_maculifemoratus_dybowskyi",     label: "사슴벌레" },
  { value: "Dorcus_hopei_binodulosus",              label: "왕사슴벌레" },
  { value: "Prosopocoilus_inclinatus_inclinatus",   label: "톱사슴벌레" },
  { value: "Dorcus_rectus_rectus",                  label: "애사슴벌레" },
  { value: "Dorcus_rubrofemoratus_rubrofemoratus",  label: "홍다리사슴벌레" },
  { value: "Prismognathus_dauricus",                label: "다우리아사슴벌레" },
  { value: "Dorcus_carinulatus_koreanus",           label: "털보왕사슴벌레" },
  { value: "Platycerus_hongwonpyoi_hongwonpyoi",    label: "원표애보라사슴벌레" },
  { value: "Dorcus_consentaneus_consentaneus",      label: "참넓적사슴벌레" },
  { value: "Aegus_laevicollis_subnitidus",          label: "꼬마넓적사슴벌레" },
  { value: "Nigidius_miwai",                        label: "뿔꼬마사슴벌레" },
  { value: "Dorcus_tenuihirsutus",                  label: "엷은털왕사슴벌레" },
  { value: "Figulus_punctatus",                     label: "길쭉꼬마사슴벌레" },
  { value: "Figulus_binodulus",                     label: "큰꼬마사슴벌레" },
];

const NON_RESULT_MESSAGE: Record<string, { icon: string; title: string; text: string }> = {
  no_beetle:      { icon: "🔍", title: "감지 실패",     text: "사진에서 사슴벌레를 찾을 수 없어요" },
  uncertain:      { icon: "😕", title: "인식 불가",     text: "더 선명한 사진을 올려주세요" },
  foreign:        { icon: "🌏", title: "한국 외 종",    text: "한국에서 서식하지 않는 종으로 추정됩니다" },
  low_confidence: { icon: "🤔", title: "신뢰도 낮음",   text: "아래 후보 종을 확인해보세요" },
};

export function ResultCard({ predictionId }: { predictionId: string }) {
  const [data, setData]             = useState<Prediction | null>(null);
  const [feedback, setFeedback]     = useState<FeedbackState>("idle");
  const [correction, setCorrection] = useState("");
  const [sex, setSex]               = useState("");
  const [maleForm, setMaleForm]     = useState("");

  useEffect(() => {
    const raw = sessionStorage.getItem(predictionId) ?? localStorage.getItem(predictionId);
    if (raw) setData(JSON.parse(raw));
  }, [predictionId]);

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
        {feedback === "idle" && (
          <div className="flex gap-2 border-t border-border/50 pt-4">
            <Button
              variant="outline"
              className="flex-1 text-sm border-emerald-700/40 text-emerald-400 hover:bg-emerald-900/20 hover:border-emerald-600/50"
              onClick={() => sendFeedback(true)}
            >
              맞아요
            </Button>
            <Button
              variant="outline"
              className="flex-1 text-sm border-border hover:border-amber-500/30 hover:bg-amber-500/5"
              onClick={() => setFeedback("correcting")}
            >
              아니에요
            </Button>
          </div>
        )}

        {feedback === "correcting" && (
          <div className="space-y-2 border-t border-border/50 pt-4">
            <select
              className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
            >
              <option value="">종을 선택하세요</option>
              {SPECIES_OPTIONS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label} — {s.value.replace(/_/g, " ")}
                </option>
              ))}
            </select>

            <select
              className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
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
                className="w-full border border-border rounded-lg px-3 py-2 text-sm bg-background text-foreground focus:outline-none focus:border-amber-500/50"
                value={maleForm}
                onChange={(e) => setMaleForm(e.target.value)}
              >
                <option value="">수컷 형태 (선택)</option>
                <option value="major">대형 (뿔 큰 수컷)</option>
                <option value="minor">소형 (뿔 작은 수컷)</option>
                <option value="intermediate">중간형</option>
              </select>
            )}

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

        {(feedback === "correct_sent" || feedback === "corrected") && (
          <p className="text-center text-sm text-amber-400/80 border-t border-border/50 pt-4">
            피드백이 저장됐어요. 감사합니다!
          </p>
        )}

        {feedback === "error" && (
          <p className="text-center text-sm text-destructive border-t border-border/50 pt-4">
            저장 실패 — 잠시 후 다시 시도해보세요
          </p>
        )}
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
