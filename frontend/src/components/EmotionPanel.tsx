import type { EmotionState } from "@/types";

export function EmotionPanel({ emotion }: { emotion: EmotionState | null }) {
  if (!emotion || Object.keys(emotion).length === 0)
    return <p className="text-muted-foreground text-sm">감정 데이터 없음</p>;

  return (
    <div className="space-y-3">
      {Object.entries(emotion).map(([key, value]) => (
        <div key={key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{key}</span>
            <span className="text-muted-foreground">
              {(value * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${Math.min(value * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
