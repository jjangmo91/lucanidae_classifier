import type { Metadata } from "next";
import { ResultCard } from "@/components/predict/ResultCard";
import { SPECIES_BY_MODEL_KEY } from "@/lib/species-data";

interface Props {
  params: { id: string };
}

const BACKEND = process.env.BACKEND_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  try {
    const res = await fetch(`${BACKEND}/api/v1/specimens/${params.id}`, {
      next: { revalidate: 3600 },
    });
    if (!res.ok) throw new Error();
    const data = await res.json();

    const spInfo = data.species ? SPECIES_BY_MODEL_KEY[data.species] : null;
    const name   = spInfo?.ko ?? data.species?.replace(/_/g, " ") ?? "사슴벌레";
    const pct    = data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "";
    const title  = `${name} 동정 결과`;
    const desc   = `AI 신뢰도 ${pct} · 루카덱스에서 동정한 ${name}`;

    return {
      title,
      openGraph: { title: `${name} | LucaDex`, description: desc },
      twitter:   { title: `${name} | LucaDex`, description: desc },
    };
  } catch {
    return { title: "동정 결과" };
  }
}

export default function ResultPage({ params }: Props) {
  return (
    <div className="max-w-lg mx-auto py-8">
      <ResultCard predictionId={params.id} />
    </div>
  );
}
