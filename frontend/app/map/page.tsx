"use client";

import { useEffect, useState } from "react";
import { useAuth, authHeader } from "@/lib/auth";
import { signIn } from "next-auth/react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Pin {
  id:          string;
  lat:         number;
  lng:         number;
  species:     string | null;
  image_url:   string | null;
  upload_time: string | null;
}

export default function MapPage() {
  const { isLoggedIn, isLoading, backendToken } = useAuth();
  const [pins, setPins]         = useState<Pin[]>([]);
  const [selected, setSelected] = useState<Pin | null>(null);

  useEffect(() => {
    if (!isLoggedIn || !backendToken) return;
    fetch(`${API}/api/v1/my/map`, { headers: authHeader(backendToken) })
      .then(r => r.json())
      .then(setPins);
  }, [isLoggedIn, backendToken]);

  if (isLoading) return null;

  if (!isLoggedIn) {
    return (
      <div className="flex flex-col items-center justify-center py-32 gap-4">
        <p className="text-lg text-muted-foreground">지도는 로그인 후 이용할 수 있습니다</p>
        <button
          onClick={() => signIn("google")}
          className="px-6 py-2 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 transition-all"
        >
          Google 로그인
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div>
        <h1 className="text-2xl font-black text-amber-400">내 채집 지도</h1>
        <p className="text-sm text-muted-foreground mt-1">
          내가 올린 사진만 표시됩니다. 다른 유저의 위치는 공개되지 않습니다.
        </p>
      </div>

      {pins.length === 0 ? (
        <div className="text-center text-muted-foreground py-20 border border-dashed border-white/10 rounded-xl">
          <p>GPS 정보가 있는 사진이 없습니다</p>
          <p className="text-xs mt-2">사진 업로드 시 GPS 메타데이터가 포함되면 지도에 표시됩니다</p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="border border-white/10 rounded-xl overflow-hidden">
            <div className="bg-card/50 p-3 border-b border-white/5">
              <p className="text-sm text-muted-foreground">총 {pins.length}개 채집 기록</p>
            </div>
            <div className="divide-y divide-white/5">
              {pins.map(pin => (
                <button
                  key={pin.id}
                  onClick={() => setSelected(selected?.id === pin.id ? null : pin)}
                  className="w-full flex items-center gap-3 p-3 hover:bg-white/5 transition-all text-left"
                >
                  {pin.image_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={pin.image_url} alt="" className="w-12 h-12 rounded-lg object-cover shrink-0" />
                  ) : (
                    <div className="w-12 h-12 rounded-lg bg-white/10 shrink-0" />
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">
                      {pin.species?.replace(/_/g, " ") ?? "종 불명"}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {pin.lat.toFixed(4)}°N · {pin.lng.toFixed(4)}°E
                    </p>
                    {pin.upload_time && (
                      <p className="text-xs text-muted-foreground">
                        {new Date(pin.upload_time).toLocaleDateString("ko-KR")}
                      </p>
                    )}
                  </div>
                  {selected?.id === pin.id && <span className="text-amber-400 text-xs shrink-0">▼</span>}
                </button>
              ))}
            </div>
          </div>

          {selected && (
            <div className="p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 space-y-2">
              <p className="font-semibold text-amber-400">
                {selected.species?.replace(/_/g, " ") ?? "종 불명"}
              </p>
              <p className="text-sm text-muted-foreground">
                위도 {selected.lat.toFixed(6)} · 경도 {selected.lng.toFixed(6)}
              </p>
              <a
                href={`https://maps.google.com/?q=${selected.lat},${selected.lng}`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block text-xs text-amber-400 hover:underline"
              >
                Google 지도에서 열기 →
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
