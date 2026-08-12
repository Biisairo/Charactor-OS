import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { PerformanceData, TurnMetrics } from "@/types";
import { LLM_LABELS, STAGE_LABELS } from "@/constants";
import { formatCost, formatTokens } from "@/lib/format";

/** 턴당 LLM 호출·토큰·비용. TASK-04가 만든 정보가 여기서 화면에 닿는다. */
function LlmMetrics({ metrics }: { metrics: TurnMetrics }) {
  const labels = Object.entries(metrics.by_label);
  // 토큰 합계가 하한이면 그렇게 보여야 한다. 확정값처럼 두면
  // 거부·미상 호출이 "토큰 0"으로 뭉개진다.
  const approx = metrics.tokens_are_lower_bound ? "≥ " : "";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">LLM 호출</span>
        {metrics.model && (
          <span className="text-[10px] text-muted-foreground tabular-nums">{metrics.model}</span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-sm font-semibold tabular-nums">{metrics.calls}</div>
          <div className="text-[10px] text-muted-foreground">호출</div>
        </div>
        <div>
          <div className="text-sm font-semibold tabular-nums">
            {approx}
            {formatTokens(metrics.total_tokens)}
          </div>
          <div className="text-[10px] text-muted-foreground">토큰</div>
        </div>
        <div>
          <div className="text-sm font-semibold tabular-nums">{formatCost(metrics.cost_usd)}</div>
          <div className="text-[10px] text-muted-foreground">추정 비용</div>
        </div>
      </div>

      <div className="text-[10px] text-muted-foreground text-center tabular-nums">
        입력 {approx}
        {formatTokens(metrics.prompt_tokens)} · 출력 {approx}
        {formatTokens(metrics.completion_tokens)}
      </div>

      {(metrics.tokens_are_lower_bound ||
        metrics.refused_calls > 0 ||
        metrics.failed_calls > 0) && (
        <div className="space-y-1 rounded border border-amber-500/40 bg-amber-500/10 p-2 text-[10px]">
          {metrics.tokens_are_lower_bound && (
            <div>
              usage가 없는 호출 {metrics.unknown_usage_calls}건 — 토큰·비용은 <b>하한</b>입니다.
            </div>
          )}
          {metrics.refused_calls > 0 && <div>프로바이더 거부 {metrics.refused_calls}건</div>}
          {metrics.failed_calls > 0 && <div>실패 {metrics.failed_calls}건</div>}
        </div>
      )}

      {labels.length > 0 && (
        <div className="space-y-1">
          {labels.map(([label, m]) => (
            <div key={label} className="flex items-center justify-between text-[11px]">
              <span className="text-muted-foreground">
                {LLM_LABELS[label] || label}
                <span className="ml-1 tabular-nums">×{m.calls}</span>
              </span>
              <span className="tabular-nums text-muted-foreground">
                {formatTokens(m.prompt_tokens)} / {formatTokens(m.completion_tokens)}
              </span>
            </div>
          ))}
          <div className="text-[10px] text-muted-foreground text-right">입력 / 출력</div>
        </div>
      )}
    </div>
  );
}

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
  const metrics = performance.trace?.metrics;

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
      {metrics && metrics.calls > 0 && (
        <>
          <div className="border-t" />
          <LlmMetrics metrics={metrics} />
          <div className="border-t" />
        </>
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
