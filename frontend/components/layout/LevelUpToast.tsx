"use client";

import { useEffect, useRef, useState } from "react";
import { useAuth, authHeader } from "@/lib/auth";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const SPECIALTY_ICON: Record<string, string> = {
  "레어헌터":   "💎",
  "분류학자":   "🔬",
  "도감탐험가": "🗺️",
  "채집왕":     "👑",
};

export function LevelUpToast() {
  const { isLoggedIn, backendToken, grade, specialty } = useAuth();
  const prevGrade    = useRef<string | null>(null);
  const prevSpecialty = useRef<string | null>(null);
  const [toast, setToast] = useState<{ grade: string; specialty: string | null } | null>(null);

  useEffect(() => {
    if (!isLoggedIn) return;

    const stored = sessionStorage.getItem("known_grade");
    const storedSpecialty = sessionStorage.getItem("known_specialty");

    // 세션 내 첫 로드면 저장만 하고 알림 없음
    if (!stored) {
      if (grade) sessionStorage.setItem("known_grade", grade);
      if (specialty) sessionStorage.setItem("known_specialty", specialty ?? "");
      prevGrade.current    = grade;
      prevSpecialty.current = specialty;
      return;
    }

    // 등급 또는 직업이 바뀌었을 때
    const gradeChanged    = grade && grade !== stored;
    const specialtyChanged = specialty && specialty !== storedSpecialty;

    if (gradeChanged || specialtyChanged) {
      setToast({ grade: grade ?? stored, specialty: specialty ?? null });
      sessionStorage.setItem("known_grade", grade ?? stored);
      sessionStorage.setItem("known_specialty", specialty ?? "");
      setTimeout(() => setToast(null), 5000);
    }

    prevGrade.current    = grade;
    prevSpecialty.current = specialty;
  }, [grade, specialty, isLoggedIn]);

  if (!toast) return null;

  const icon         = toast.specialty ? SPECIALTY_ICON[toast.specialty] : "⬆️";
  const displayGrade = toast.specialty ?? toast.grade;

  return (
    <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-[100] animate-in slide-in-from-bottom-4 duration-300">
      <div className="flex items-center gap-3 px-5 py-3.5 rounded-2xl border border-amber-500/40 bg-card/90 backdrop-blur-xl shadow-[0_0_30px_rgba(245,158,11,0.15)]">
        <span className="text-2xl">{icon}</span>
        <div>
          <p className="text-xs text-amber-400/70 font-mono uppercase tracking-widest">등급 상승</p>
          <p className="font-black text-lg text-amber-400">{displayGrade}</p>
        </div>
        <button onClick={() => setToast(null)} className="ml-2 text-muted-foreground hover:text-foreground text-lg">×</button>
      </div>
    </div>
  );
}
