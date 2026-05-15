import { ResultCard } from "@/components/predict/ResultCard";

interface Props {
  params: { id: string };
}

export default function ResultPage({ params }: Props) {
  return (
    <div className="max-w-lg mx-auto py-8">
      <ResultCard predictionId={params.id} />
    </div>
  );
}
