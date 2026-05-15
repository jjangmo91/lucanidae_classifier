import Link from "next/link";
import { notFound } from "next/navigation";
import { SPECIES_BY_SLUG, RARITY_LABEL, RARITY_COLOR } from "@/lib/species-data";

interface Props {
  params: { slug: string };
}

export default function SpeciesDetailPage({ params }: Props) {
  const sp = SPECIES_BY_SLUG[params.slug];
  if (!sp) notFound();

  const stars = "★".repeat(sp.rarity) + "☆".repeat(4 - sp.rarity);

  return (
    <div className="max-w-2xl mx-auto py-8 space-y-6">

      {/* 헤더 */}
      <div className="space-y-1">
        <Link href="/species" className="text-xs text-muted-foreground hover:underline">
          ← 종 도감으로
        </Link>
        <h1 className="text-3xl font-bold mt-2">{sp.ko}</h1>
        <p className="text-base italic text-muted-foreground">{sp.scientific}</p>
        <span className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full ${RARITY_COLOR[sp.rarity]}`}>
          {stars} {RARITY_LABEL[sp.rarity]}
        </span>
      </div>

      {/* 기본 정보 */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <InfoCard label="수컷 크기" value={sp.sizeM} />
        <InfoCard label="암컷 크기" value={sp.sizeF} />
        <InfoCard label="활동 시기" value={sp.season} />
        <InfoCard label="희귀도" value={RARITY_LABEL[sp.rarity]} />
      </div>

      {/* 서식지 */}
      <Section title="서식지">
        <p className="text-sm">{sp.habitat}</p>
      </Section>

      {/* 종 설명 */}
      <Section title="특징">
        <p className="text-sm leading-relaxed">{sp.description}</p>
      </Section>

      {/* 암수 구별 */}
      <Section title="암수 구별">
        <p className="text-sm leading-relaxed">{sp.dimorphism}</p>
      </Section>

      {/* 동정 팁 */}
      <Section title="동정 팁">
        <p className="text-sm leading-relaxed text-amber-700 bg-amber-50 rounded-lg px-4 py-3">
          {sp.tip}
        </p>
      </Section>

      {/* AI 분류 버튼 */}
      <Link
        href="/"
        className="block w-full text-center rounded-xl bg-primary text-primary-foreground py-3 text-sm font-semibold hover:opacity-90 transition-opacity"
      >
        🪲 내 사진이 {sp.ko}인지 확인하기
      </Link>
    </div>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border p-3 text-center space-y-1">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="text-sm font-semibold">{value}</p>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wide">{title}</h2>
      {children}
    </div>
  );
}
