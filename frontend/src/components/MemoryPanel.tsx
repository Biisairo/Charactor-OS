import { Badge } from "@/components/ui/badge";
import type { MemoryEntry } from "@/types";

export function MemoryPanel({
  stats,
  memories,
}: {
  stats: { count: number } | null;
  memories: MemoryEntry[];
}) {
  if (!stats)
    return <p className="text-muted-foreground text-sm">기억 데이터 없음</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">저장된 기억</span>
        <Badge variant="secondary">{stats.count}개</Badge>
      </div>
      {memories.length > 0 && (
        <div className="space-y-2 max-h-[300px] overflow-auto">
          {memories.map((m) => (
            <div key={m.id} className="p-2 bg-muted rounded text-xs space-y-1">
              <p className="text-foreground">{m.content}</p>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span>가중치: {m.weight}</span>
                <span>접근: {m.access_count}회</span>
              </div>
              {Object.keys(m.emotion_tags).length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {Object.entries(m.emotion_tags).map(([k, v]) => (
                    <Badge key={k} variant="outline" className="text-[10px] px-1 py-0">
                      {k}: {(v * 100).toFixed(0)}%
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
