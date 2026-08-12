import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { PerformanceData } from "@/types";
import { STAGE_LABELS } from "@/constants";

export function PerformancePanel({
  performance,
  onRefresh,
}: {
  performance: PerformanceData | null;
  onRefresh: () => void;
}) {
  if (!performance) {
    return (
      <div className="space-y-2">
        <p className="text-muted-foreground text-sm">데이터 없음</p>
        <Button variant="outline" size="sm" className="w-full text-xs" onClick={onRefresh}>
          새로고침
        </Button>
      </div>
    );
  }

  const stages = performance.trace?.stages ?? [];
  const totalMs = performance.trace?.total_duration_ms;
  const maxDuration = stages.reduce((max, s) => Math.max(max, s.duration_ms), 0);

  return (
    <div className="space-y-3">
      {totalMs != null && (
        <div className="text-center">
          <div className="text-2xl font-bold tabular-nums">{totalMs.toFixed(0)}</div>
          <div className="text-xs text-muted-foreground">총 지연 시간 (ms)</div>
        </div>
      )}
      {stages.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-medium text-muted-foreground">단계별 분석</span>
          {stages.map((stage, i) => {
            const pct = maxDuration > 0 ? (stage.duration_ms / maxDuration) * 100 : 0;
            return (
              <div key={i} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span>{STAGE_LABELS[stage.name] || stage.name}</span>
                  <span className="text-muted-foreground tabular-nums">
                    {stage.duration_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
      <div className="flex gap-4">
        {performance.memory_count != null && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">기억:</span>
            <Badge variant="outline" className="text-xs">{performance.memory_count}</Badge>
          </div>
        )}
        {performance.history_count != null && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">기록:</span>
            <Badge variant="outline" className="text-xs">{performance.history_count}</Badge>
          </div>
        )}
      </div>
      <Button variant="outline" size="sm" className="w-full text-xs" onClick={onRefresh}>
        새로고침
      </Button>
    </div>
  );
}
