import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { NavBar } from "@/components/layout/NavBar";

const geist = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-sans",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "사슴벌레 AI — Taxonomy & Classification",
  description: "사진 한 장으로 한국 사슴벌레 16종을 AI가 동정합니다.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko" className={geist.variable}>
      <body className="min-h-screen bg-background font-sans antialiased">
        <NavBar />
        <main className="container mx-auto px-4 py-6">{children}</main>
      </body>
    </html>
  );
}
