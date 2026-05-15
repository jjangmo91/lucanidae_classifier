import { ImageResponse } from "next/og";
import { SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

export const runtime = "edge";
export const contentType = "image/png";
export const size = { width: 1200, height: 630 };

interface Props {
  params: { id: string };
}

const BACKEND = process.env.BACKEND_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const TIER_COLOR: Record<number, string> = {
  4: "#ef4444",
  3: "#f59e0b",
  2: "#22c55e",
  1: "#94a3b8",
};
const TIER_LABEL: Record<number, string> = { 4: "S", 3: "A", 2: "B", 1: "C" };

export default async function OGImage({ params }: Props) {
  let speciesName = "사슴벌레";
  let confidence  = "";
  let rarity      = 1;
  let imageUrl: string | null = null;

  try {
    const res = await fetch(`${BACKEND}/api/v1/specimens/${params.id}`, {
      next: { revalidate: 3600 },
    });
    if (res.ok) {
      const data   = await res.json();
      const spInfo = data.species ? SPECIES_BY_MODEL_KEY[data.species] : null;
      speciesName  = spInfo?.ko ?? data.species?.replace(/_/g, " ") ?? "사슴벌레";
      confidence   = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "";
      rarity       = spInfo?.rarity ?? 1;
      imageUrl     = data.image_url
        ? `${process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000"}${data.image_url}`
        : null;
    }
  } catch { /* 기본값 사용 */ }

  const tierColor = TIER_COLOR[rarity];
  const tierLabel = TIER_LABEL[rarity];

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          background: "#0c0a06",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* 왼쪽: 텍스트 영역 */}
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            padding: "60px 64px",
            flex: 1,
            gap: 0,
          }}
        >
          {/* 사이트명 */}
          <div
            style={{
              fontSize: 20,
              color: "rgba(245,158,11,0.55)",
              letterSpacing: "6px",
              marginBottom: 32,
              textTransform: "uppercase",
            }}
          >
            BeetleDex · 동정 결과
          </div>

          {/* 희귀도 배지 */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 12,
              marginBottom: 20,
            }}
          >
            <div
              style={{
                padding: "4px 14px",
                borderRadius: 6,
                border: `2px solid ${tierColor}`,
                color: tierColor,
                fontSize: 22,
                fontWeight: 900,
                letterSpacing: "2px",
              }}
            >
              {tierLabel} TIER
            </div>
          </div>

          {/* 종 이름 */}
          <div
            style={{
              fontSize: speciesName.length > 7 ? 64 : 80,
              fontWeight: 900,
              color: "#ffffff",
              lineHeight: 1.1,
              marginBottom: 20,
            }}
          >
            {speciesName}
          </div>

          {/* 신뢰도 */}
          {confidence && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                marginTop: 8,
              }}
            >
              <span style={{ fontSize: 52, fontWeight: 900, color: "#f59e0b" }}>
                {confidence}
              </span>
              <span style={{ fontSize: 20, color: "rgba(255,255,255,0.4)", marginTop: 12 }}>
                AI 신뢰도
              </span>
            </div>
          )}
        </div>

        {/* 오른쪽: 이미지 또는 글로우 */}
        <div
          style={{
            width: 420,
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            position: "relative",
            overflow: "hidden",
          }}
        >
          {/* 배경 글로우 */}
          <div
            style={{
              position: "absolute",
              width: 400,
              height: 400,
              borderRadius: "50%",
              background: `radial-gradient(circle, ${tierColor}22 0%, transparent 70%)`,
            }}
          />
          {imageUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={imageUrl}
              alt=""
              style={{
                width: "100%",
                height: "100%",
                objectFit: "cover",
                opacity: 0.85,
              }}
            />
          ) : (
            <div
              style={{
                fontSize: 120,
                color: `${tierColor}60`,
                fontWeight: 900,
              }}
            >
              {tierLabel}
            </div>
          )}
          {/* 왼쪽 페이드 */}
          <div
            style={{
              position: "absolute",
              left: 0,
              top: 0,
              width: 120,
              height: "100%",
              background: "linear-gradient(to right, #0c0a06, transparent)",
            }}
          />
        </div>
      </div>
    ),
    { ...size },
  );
}
