"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";

type State = "idle" | "scanning" | "done" | "error";

const CROSSHAIR_CLASSES: Record<string, string> = {
  tl: "-top-2 -left-2 border-t-2 border-l-2",
  tr: "-top-2 -right-2 border-t-2 border-r-2",
  bl: "-bottom-2 -left-2 border-b-2 border-l-2",
  br: "-bottom-2 -right-2 border-b-2 border-r-2",
};

export function UploadSection() {
  const router   = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [state, setState]     = useState<State>("idle");
  const [preview, setPreview] = useState<string | null>(null);
  const [errMsg, setErrMsg]   = useState<string | null>(null);

  async function handleFile(file: File) {
    setPreview(URL.createObjectURL(file));
    setState("scanning");
    setErrMsg(null);

    let lat: number | null = null, lng: number | null = null;
    if (typeof navigator !== "undefined" && navigator.geolocation) {
      try {
        const pos = await new Promise<GeolocationPosition>((resolve, reject) =>
          navigator.geolocation.getCurrentPosition(resolve, reject, { timeout: 4000 })
        );
        lat = pos.coords.latitude;
        lng = pos.coords.longitude;
      } catch { /* GPS 거부/실패 시 조용히 무시 */ }
    }

    const form = new FormData();
    form.append("file", file);
    if (lat !== null && lng !== null) {
      form.append("lat", String(lat));
      form.append("lng", String(lng));
    }

    try {
      const res = await fetch("/api/v1/predict", { method: "POST", body: form });
      if (!res.ok) { setErrMsg(`서버 오류 (${res.status})`); setState("error"); return; }
      const data   = await res.json();
      const result = data[0];
      if (result?.prediction_id) {
        const entry = { ...result, analyzed_at: new Date().toISOString() };
        sessionStorage.setItem(result.prediction_id, JSON.stringify(entry));
        localStorage.setItem(result.prediction_id, JSON.stringify(entry));
        const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") || "[]");
        ids.unshift(result.prediction_id);
        localStorage.setItem("prediction_history", JSON.stringify(ids.slice(0, 100)));
        router.push(`/result/${result.prediction_id}`);
      }
    } catch {
      setErrMsg("서버에 연결할 수 없어요.");
      setState("error");
    }
  }

  const canInteract = state === "idle" || state === "error";

  return (
    <div className="relative z-10 flex flex-col items-center gap-6">

      {/* ──── SPECIMEN POD ──── */}
      <div
        className="relative select-none"
        style={{ cursor: canInteract ? "pointer" : "default" }}
        onClick={() => canInteract && inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          const file = e.dataTransfer.files[0];
          if (file && canInteract) handleFile(file);
        }}
      >
        {/* Pulse rings — scanning only */}
        {state === "scanning" && (
          <>
            <div className="absolute inset-0 rounded-full border border-amber-400/50"
              style={{ animation: "pulse-ring 1.6s ease-out infinite" }} />
            <div className="absolute inset-0 rounded-full border border-amber-400/30"
              style={{ animation: "pulse-ring 1.6s ease-out 0.55s infinite" }} />
          </>
        )}

        {/* Main circle */}
        <div className={`
          relative w-56 h-56 rounded-full overflow-hidden transition-all duration-500
          ${state === "scanning"
            ? "shadow-[0_0_0_2px_hsl(43_96%_56%/0.9),0_0_50px_hsl(43_96%_56%/0.45),0_0_100px_hsl(43_96%_56%/0.18)]"
            : state === "error"
            ? "shadow-[0_0_0_2px_hsl(0_72%_51%/0.7),0_0_30px_hsl(0_72%_51%/0.25)]"
            : "animate-float shadow-[0_0_0_2px_hsl(43_96%_56%/0.35),0_0_35px_hsl(43_96%_56%/0.18)] hover:shadow-[0_0_0_2px_hsl(43_96%_56%/0.65),0_0_55px_hsl(43_96%_56%/0.3)]"}
        `}>
          {/* scanlines overlay */}
          <div className="absolute inset-0 z-20 pointer-events-none scanlines" />

          {/* scan sweep — scanning only */}
          {state === "scanning" && (
            <div
              className="scan-line absolute inset-0 z-30 pointer-events-none"
              style={{ animation: "scanMove 1.4s ease-in-out infinite" }}
            />
          )}

          {/* Image */}
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt="preview"
              className={`w-full h-full object-cover transition-all duration-500
                ${state === "scanning" ? "opacity-75 scale-95" : "opacity-100 scale-100"}`}
            />
          ) : (
            <div className="specimen-pod w-full h-full flex items-center justify-center">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src="/lucanus.png" alt="Lucanidae"
                className="w-44 h-44 object-contain transition-transform duration-300 hover:scale-105"
              />
            </div>
          )}
        </div>

        {/* Corner crosshairs */}
        {Object.entries(CROSSHAIR_CLASSES).map(([pos, cls]) => (
          <div key={pos}
            className={`absolute w-5 h-5 border-amber-500/65 rounded-sm ${cls}`}
          />
        ))}

        {/* Specimen tag */}
        <div className="absolute -right-4 top-5 bg-card/90 backdrop-blur-sm border border-amber-500/25 px-2 py-0.5 rounded text-[10px] font-mono text-amber-400/60 tracking-widest">
          LUC·16
        </div>
      </div>

      {/* ──── TEXT BELOW POD ──── */}
      <div className="text-center space-y-1.5">
        {state === "idle" && (
          <>
            <h1 className="text-4xl font-black tracking-tight leading-none">
              <span className="text-amber-400 text-glow">비틀</span>덱스
            </h1>
            <p className="text-sm text-muted-foreground">
              사진 한 장으로 한국 사슴벌레 16종을 즉시 동정
            </p>
          </>
        )}

        {state === "scanning" && (
          <div className="space-y-2.5">
            <p className="text-sm font-bold tracking-widest text-amber-400 uppercase">
              AI 분석 중
            </p>
            <div className="flex gap-1.5 justify-center items-end h-6">
              {[0, 1, 2, 3, 4].map((i) => (
                <div key={i}
                  className="w-1 rounded-full bg-amber-500"
                  style={{
                    height: "100%",
                    animation: `bounce-bar 0.9s ${i * 0.12}s ease-in-out infinite`,
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {state === "error" && (
          <p className="text-sm text-destructive/90">{errMsg}</p>
        )}
      </div>

      {/* ──── ACTION ──── */}
      {state === "idle" && (
        <>
          <Button
            className="h-11 w-56 font-bold text-sm bg-amber-500 hover:bg-amber-400 text-amber-950 glow-amber-sm transition-all"
            onClick={() => inputRef.current?.click()}
          >
            사진으로 동정하기
          </Button>
          <p className="text-xs text-muted-foreground -mt-3">JPG · PNG · 드래그&드롭 지원</p>
        </>
      )}

      {state === "error" && (
        <Button
          variant="outline"
          className="w-56 border-amber-500/30 text-amber-400 hover:bg-amber-500/10"
          onClick={() => { setState("idle"); setPreview(null); }}
        >
          다시 시도
        </Button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      <style>{`
        .specimen-pod {
          background: radial-gradient(circle at 40% 38%, #f9f6f0, #e9dfd2);
        }
        @keyframes pulse-ring {
          0%   { transform: scale(1);    opacity: 0.55; }
          100% { transform: scale(1.45); opacity: 0; }
        }
        @keyframes scanMove {
          0%   { transform: translateY(-100%); }
          100% { transform: translateY(200%); }
        }
        @keyframes bounce-bar {
          0%, 80%, 100% { transform: scaleY(0.3); opacity: 0.3; }
          40%           { transform: scaleY(1);   opacity: 1; }
        }
      `}</style>
    </div>
  );
}
