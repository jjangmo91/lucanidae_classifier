export default function MapPage() {
  return (
    <div className="py-8 space-y-4">
      <div>
        <h1 className="text-2xl font-bold">관찰 지도</h1>
        <p className="text-muted-foreground text-sm mt-1">
          커뮤니티 관찰 기록 — 1km 격자 히트맵 (GPS 보호)
        </p>
      </div>

      {/* TODO: Leaflet 또는 react-map-gl 히트맵 */}
      <div className="rounded-xl border bg-muted flex items-center justify-center h-[500px] text-muted-foreground text-sm">
        지도 컴포넌트 (Phase 3)
      </div>
    </div>
  );
}
