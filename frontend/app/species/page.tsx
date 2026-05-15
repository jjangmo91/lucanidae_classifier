import Link from "next/link";

const SPECIES_LIST = [
  { slug: "dorcus-hopei-binodulosus",                   ko: "왕사슴벌레",       en: "Dorcus hopei binodulosus",                   rarity: "S" },
  { slug: "dorcus-titanus-castanicolor",                 ko: "넓적사슴벌레",     en: "Dorcus titanus castanicolor",                 rarity: "A" },
  { slug: "lucanus-maculifemoratus-dybowskyi",           ko: "사슴벌레",        en: "Lucanus maculifemoratus dybowskyi",           rarity: "A" },
  { slug: "prosopocoilus-inclinatus-inclinatus",         ko: "톱사슴벌레",      en: "Prosopocoilus inclinatus inclinatus",         rarity: "B" },
  { slug: "prosopocoilus-astacoides-blanchardi",         ko: "두점박이사슴벌레", en: "Prosopocoilus astacoides blanchardi",         rarity: "B" },
  { slug: "dorcus-rectus-rectus",                       ko: "애사슴벌레",      en: "Dorcus rectus rectus",                       rarity: "B" },
  { slug: "dorcus-rubrofemoratus-rubrofemoratus",        ko: "홍다리사슴벌레",  en: "Dorcus rubrofemoratus rubrofemoratus",        rarity: "C" },
  { slug: "prismognathus-dauricus",                     ko: "다우리아사슴벌레", en: "Prismognathus dauricus",                     rarity: "C" },
  { slug: "dorcus-carinulatus-koreanus",                 ko: "털보왕사슴벌레",  en: "Dorcus carinulatus koreanus",                rarity: "C" },
  { slug: "platycerus-hongwonpyoi-hongwonpyoi",          ko: "원표애보라사슴벌레", en: "Platycerus hongwonpyoi hongwonpyoi",       rarity: "S" },
  { slug: "dorcus-consentaneus-consentaneus",            ko: "참넓적사슴벌레",  en: "Dorcus consentaneus consentaneus",           rarity: "C" },
  { slug: "aegus-laevicollis-subnitidus",                ko: "꼬마넓적사슴벌레", en: "Aegus laevicollis subnitidus",              rarity: "B" },
  { slug: "nigidius-miwai",                             ko: "뿔꼬마사슴벌레",  en: "Nigidius miwai",                             rarity: "A" },
  { slug: "dorcus-tenuihirsutus",                       ko: "엷은털왕사슴벌레", en: "Dorcus tenuihirsutus",                      rarity: "C" },
  { slug: "figulus-punctatus",                          ko: "길쭉꼬마사슴벌레", en: "Figulus punctatus",                         rarity: "B" },
  { slug: "figulus-binodulus",                          ko: "큰꼬마사슴벌레",  en: "Figulus binodulus",                          rarity: "B" },
];

const RARITY_STYLE: Record<string, string> = {
  S: "text-amber-300 border-amber-400/50 bg-amber-500/10",
  A: "text-violet-300 border-violet-400/40 bg-violet-500/8",
  B: "text-sky-300   border-sky-400/40    bg-sky-500/8",
  C: "text-slate-400 border-slate-500/30  bg-slate-500/6",
};

export default function SpeciesListPage() {
  return (
    <div className="py-8 space-y-6">

      <div className="space-y-1">
        <p className="text-[10px] font-mono tracking-[0.3em] text-amber-500/45 uppercase">Korea Lucanidae</p>
        <h1 className="text-2xl font-black">종 도감</h1>
        <p className="text-sm text-muted-foreground">
          한국 사슴벌레과 16종 — 국립생물자원관 국가생물종목록 기준
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
        {SPECIES_LIST.map((s) => (
          <Link
            key={s.slug}
            href={`/species/${s.slug}`}
            className="group rounded-xl border border-border bg-card p-4 hover:border-amber-500/30 hover:bg-amber-500/5 transition-all duration-200 space-y-2"
          >
            <div className="flex justify-between items-start">
              <div className="w-12 h-12 rounded-lg specimen-pod flex items-center justify-center text-2xl border border-border/40">
                🪲
              </div>
              <span className={`text-[10px] font-black px-1.5 py-0.5 rounded border font-mono ${RARITY_STYLE[s.rarity]}`}>
                {s.rarity}
              </span>
            </div>
            <div>
              <p className="font-bold text-sm leading-tight group-hover:text-amber-400 transition-colors">{s.ko}</p>
              <p className="text-[10px] text-muted-foreground italic mt-0.5 leading-tight">{s.en}</p>
            </div>
          </Link>
        ))}
      </div>

      <style>{`
        .specimen-pod {
          background: radial-gradient(circle at 40% 38%, #f9f6f0, #e9dfd2);
        }
      `}</style>
    </div>
  );
}
