import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { NavBar } from "@/components/layout/NavBar";
import Providers from "@/components/layout/Providers";

const geist = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-sans",
  weight: "100 900",
});

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: { default: "루카덱스 — LucaDex", template: "%s | LucaDex" },
  description: "사진 한 장으로 한국 사슴벌레 16종을 즉시 동정. 채집 기록·랭킹·커뮤니티까지.",
  keywords: ["사슴벌레", "루카덱스", "LucaDex", "동정", "Lucanidae", "AI", "곤충"],
  manifest: "/manifest.webmanifest",
  appleWebApp: {
    capable: true,
    title: "루카덱스",
    statusBarStyle: "black-translucent",
  },
  openGraph: {
    type: "website",
    siteName: "루카덱스",
    title: "루카덱스 — LucaDex",
    description: "사진 한 장으로 한국 사슴벌레 16종을 즉시 동정. 채집 기록·랭킹·커뮤니티.",
    locale: "ko_KR",
  },
  twitter: {
    card: "summary_large_image",
    title: "루카덱스 — LucaDex",
    description: "사진 한 장으로 한국 사슴벌레 16종을 즉시 동정",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className={geist.variable}>
      <body className="min-h-screen bg-background font-sans antialiased">
        <Providers>
          <NavBar />
          <main className="container mx-auto px-4 py-6">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
