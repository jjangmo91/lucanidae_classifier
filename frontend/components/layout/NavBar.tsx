"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/",        label: "홈" },
  { href: "/my",      label: "내 도감" },
  { href: "/map",     label: "지도" },
  { href: "/species", label: "종 도감" },
];

export function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="sticky top-0 z-50 border-b border-amber-500/10 bg-card/70 backdrop-blur-xl">
      <div className="container mx-auto px-4 flex items-center justify-between h-14">

        <Link href="/" className="flex items-center gap-2.5 group">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/lucanus.png" alt="" className="w-7 h-7 object-contain opacity-90 group-hover:opacity-100 transition-opacity" />
          <span className="font-black text-base tracking-tight">
            <span className="text-amber-400">사슴벌레</span>
            <span className="text-foreground/70 font-light text-sm ml-1 hidden sm:inline">AI</span>
          </span>
        </Link>

        <div className="flex items-center gap-0.5">
          {NAV_ITEMS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "relative px-3 py-1.5 rounded-md text-sm font-medium transition-all duration-200",
                  active
                    ? "text-amber-400"
                    : "text-muted-foreground hover:text-foreground hover:bg-white/5"
                )}
              >
                {label}
                {active && (
                  <span className="absolute bottom-0.5 left-1/2 -translate-x-1/2 w-4 h-0.5 rounded-full bg-amber-400 glow-amber-sm" />
                )}
              </Link>
            );
          })}
        </div>

      </div>
    </nav>
  );
}
