import { ImageResponse } from "next/og";

export const runtime = "edge";
export const size = { width: 512, height: 512 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#0c0a06",
          borderRadius: "24%",
          position: "relative",
        }}
      >
        {/* 글로우 */}
        <div
          style={{
            position: "absolute",
            width: 360,
            height: 360,
            borderRadius: "50%",
            background: "radial-gradient(circle, rgba(245,158,11,0.2) 0%, transparent 70%)",
          }}
        />
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 0,
          }}
        >
          <span
            style={{
              fontSize: 220,
              fontWeight: 900,
              color: "#f59e0b",
              lineHeight: 1,
              letterSpacing: "-8px",
            }}
          >
            비
          </span>
          <span
            style={{
              fontSize: 72,
              fontWeight: 700,
              color: "rgba(245,158,11,0.55)",
              letterSpacing: "8px",
            }}
          >
            DEX
          </span>
        </div>
      </div>
    ),
    { ...size },
  );
}
