import { UploadSection } from "@/components/predict/UploadSection";
import { StatsBar } from "@/components/predict/StatsBar";

export default function HomePage() {
  return (
    <div className="relative flex flex-col items-center min-h-[calc(100vh-3.5rem)] py-10 gap-7 overflow-hidden">

      {/* Ambient atmosphere */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[700px] h-[280px] bg-amber-500/6 blur-[90px] rounded-full" />
        <div className="absolute top-48 right-1/3 w-40 h-40 bg-emerald-500/4 blur-[70px] rounded-full" />
        <div className="absolute bottom-32 left-1/3 w-32 h-32 bg-amber-500/3 blur-[60px] rounded-full" />
      </div>

      {/* Classifier label */}
      <p className="relative z-10 text-[10px] font-mono tracking-[0.35em] text-amber-500/45 uppercase mt-2">
        Korea · Lucanidae · AI Classifier
      </p>

      {/* Main specimen pod + upload */}
      <UploadSection />

      {/* Tags */}
      <div className="flex gap-2 flex-wrap justify-center relative z-10">
        {["16종 분류", "AI 동정", "피드백 학습"].map((tag) => (
          <span
            key={tag}
            className="text-[11px] px-3 py-1.5 rounded-full border border-amber-500/20 text-amber-400/55 bg-amber-500/5 font-medium"
          >
            {tag}
          </span>
        ))}
      </div>

      <StatsBar />
    </div>
  );
}
