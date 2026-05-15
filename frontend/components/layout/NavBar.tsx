"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { signIn, signOut } from "next-auth/react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/lib/auth";

const NAV_ITEMS = [
  { href: "/",        label: "홈" },
  { href: "/my",      label: "내 도감" },
  { href: "/gallery", label: "갤러리" },
  { href: "/rank",    label: "랭킹" },
  { href: "/map",     label: "지도" },
  { href: "/species", label: "종 도감" },
];

const GRADE_COLOR: Record<string, string> = {
  "딱린이":    "text-zinc-400",
  "표본수집가": "text-emerald-400",
  "주간채집러": "text-sky-400",
  "야간채집러": "text-indigo-400",
  "루카나이더": "text-violet-400",
  "6티어":     "text-amber-400",
};

export function NavBar() {
  const pathname    = usePathname();
  const { isLoggedIn, isLoading, username, avatar, grade, specialty } = useAuth();
  const [open, setOpen] = useState(false);

  const displayGrade = specialty ?? grade ?? "";
  const gradeColor   = GRADE_COLOR[grade ?? ""] ?? "text-amber-400";

  return (
    <nav className="sticky top-0 z-50 border-b border-amber-500/10 bg-card/70 backdrop-blur-xl">
      <div className="container mx-auto px-4 flex items-center justify-between h-14">

        {/* 로고 */}
        <Link href="/" className="flex items-center gap-2.5 group shrink-0" onClick={() => setOpen(false)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/lucanus.png" alt="" className="w-7 h-7 object-contain opacity-90 group-hover:opacity-100 transition-opacity" />
          <span className="font-black text-base tracking-tight">
            <span className="text-amber-400">루카</span>
            <span className="text-foreground/80">덱스</span>
          </span>
        </Link>

        {/* 데스크톱 nav */}
        <div className="hidden md:flex items-center gap-0.5">
          {NAV_ITEMS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200",
                  active ? "text-amber-400" : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                )}
              >
                {label}
                {active && (
                  <span className="absolute bottom-0.5 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full bg-amber-400" />
                )}
              </Link>
            );
          })}
        </div>

        {/* 우측: 로그인 + 햄버거 */}
        <div className="flex items-center gap-2">
          {!isLoading && (
            isLoggedIn ? (
              <button
                onClick={() => signOut()}
                className="flex items-center gap-2 px-2 py-1.5 rounded-md hover:bg-white/5 transition-all"
              >
                {avatar && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={avatar} alt="" className="w-6 h-6 rounded-full" />
                )}
                <span className="text-sm text-foreground/80 hidden sm:block">{username}</span>
                {displayGrade && (
                  <span className={cn("text-xs font-bold hidden sm:block", gradeColor)}>
                    {displayGrade}
                  </span>
                )}
              </button>
            ) : (
              <button
                onClick={() => signIn("google")}
                className="px-3 py-1.5 rounded-md text-sm font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20 hover:bg-amber-500/20 transition-all"
              >
                <span className="hidden sm:inline">Google </span>로그인
              </button>
            )
          )}

          {/* 햄버거 (모바일) */}
          <button
            className="md:hidden flex flex-col justify-center items-center w-9 h-9 gap-1.5 rounded-md hover:bg-white/5 transition-all"
            onClick={() => setOpen(v => !v)}
            aria-label="메뉴"
          >
            <span className={cn("block w-5 h-0.5 bg-foreground/70 transition-all duration-200", open && "rotate-45 translate-y-2")} />
            <span className={cn("block w-5 h-0.5 bg-foreground/70 transition-all duration-200", open && "opacity-0")} />
            <span className={cn("block w-5 h-0.5 bg-foreground/70 transition-all duration-200", open && "-rotate-45 -translate-y-2")} />
          </button>
        </div>
      </div>

      {/* 모바일 드롭다운 */}
      {open && (
        <div className="md:hidden border-t border-amber-500/10 bg-card/95 backdrop-blur-xl">
          <div className="container mx-auto px-2 py-2 flex flex-col">
            {NAV_ITEMS.map(({ href, label }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={() => setOpen(false)}
                  className={cn(
                    "px-4 py-3 rounded-lg text-sm font-medium transition-all",
                    active ? "text-amber-400 bg-amber-500/8" : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                  )}
                >
                  {label}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </nav>
  );
}
