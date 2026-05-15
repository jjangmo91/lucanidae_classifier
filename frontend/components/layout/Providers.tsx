"use client";
import { useEffect, useRef } from "react";
import { SessionProvider } from "next-auth/react";
import { LevelUpToast } from "./LevelUpToast";
import { useAuth, authHeader } from "@/lib/auth";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function ClaimOnLogin() {
  const { isLoggedIn, backendToken } = useAuth();
  const claimed = useRef(false);

  useEffect(() => {
    if (!isLoggedIn || !backendToken || claimed.current) return;
    claimed.current = true;

    const ids: string[] = JSON.parse(localStorage.getItem("prediction_history") ?? "[]");
    if (ids.length === 0) return;

    fetch(`${API}/api/v1/my/claim-specimens`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeader(backendToken) },
      body: JSON.stringify({ specimen_ids: ids }),
    })
      .then(r => r.json())
      .then(data => {
        if (data.claimed > 0) {
          // 이전된 항목은 로컬에서 제거 (DB가 정본)
          localStorage.removeItem("prediction_history");
        }
      })
      .catch(() => { /* 네트워크 오류 시 무시 — 다음 로그인에 재시도 */ });
  }, [isLoggedIn, backendToken]);

  return null;
}

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <ClaimOnLogin />
      {children}
      <LevelUpToast />
    </SessionProvider>
  );
}
