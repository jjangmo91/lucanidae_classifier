import { ImageResponse } from "next/og";

export const runtime = "edge";
export const contentType = "image/png";
export const size = { width: 1200, height: 630 };

export default function OGImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #0c0a06 0%, #1c1408 50%, #0c0a06 100%)",
          position: "relative",
        }}
      >
        {/* 배경 글로우 */}
        <div
          style={{
            position: "absolute",
            width: 600,
            height: 600,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(245,158,11,0.12) 0%, transparent 70%)",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
          }}
        />

        {/* 로고 텍스트 */}
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 0,
          }}
        >
          <span
            style={{
              fontSize: 100,
              fontWeight: 900,
              color: "#f59e0b",
              letterSpacing: "-2px",
            }}
          >
            비틀
          </span>
          <span
            style={{
              fontSize: 100,
              fontWeight: 900,
              color: "rgba(255,255,255,0.85)",
              letterSpacing: "-2px",
            }}
          >
            덱스
          </span>
        </div>

        {/* 영문 서브타이틀 */}
        <div
          style={{
            fontSize: 36,
            fontWeight: 400,
            color: "rgba(245,158,11,0.55)",
            letterSpacing: "12px",
            marginTop: 8,
            textTransform: "uppercase",
          }}
        >
          BeetleDex
        </div>

        {/* 설명 */}
        <div
          style={{
            fontSize: 24,
            color: "rgba(255,255,255,0.4)",
            marginTop: 36,
            letterSpacing: "1px",
          }}
        >
          한국 사슴벌레 16종 AI 동정 서비스
        </div>

        {/* 하단 장식선 */}
        <div
          style={{
            position: "absolute",
            bottom: 60,
            display: "flex",
            alignItems: "center",
            gap: 16,
          }}
        >
          <div style={{ width: 60, height: 1, background: "rgba(245,158,11,0.3)" }} />
          <span style={{ fontSize: 14, color: "rgba(245,158,11,0.4)", letterSpacing: "4px" }}>
            LUCANIDAE · AI
          </span>
          <div style={{ width: 60, height: 1, background: "rgba(245,158,11,0.3)" }} />
        </div>
      </div>
    ),
    { ...size },
  );
}
