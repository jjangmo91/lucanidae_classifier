import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "루카덱스",
    short_name: "LucaDex",
    description: "한국 사슴벌레 16종 AI 동정 서비스",
    start_url: "/",
    display: "standalone",
    background_color: "#0c0a06",
    theme_color: "#f59e0b",
    orientation: "portrait-primary",
    categories: ["lifestyle", "education"],
    icons: [
      { src: "/icon", sizes: "512x512", type: "image/png" },
      { src: "/icon", sizes: "192x192", type: "image/png" },
      { src: "/apple-icon", sizes: "180x180", type: "image/png" },
    ],
    shortcuts: [
      {
        name: "동정하기",
        url: "/",
        description: "사진으로 사슴벌레 종 동정",
      },
      {
        name: "갤러리",
        url: "/gallery",
        description: "커뮤니티 채집 갤러리",
      },
    ],
  };
}
