import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { EmotionPanel } from "@/components/EmotionPanel";
import { MemoryPanel } from "@/components/MemoryPanel";
import { PerformancePanel } from "@/components/PerformancePanel";
import { stripAnsi } from "@/lib/format";
import type { EmotionState, MemoryEntry, PerformanceData } from "@/types";

interface Props {
  emotion: EmotionState | null;
  memoryStats: { count: number } | null;
  memories: MemoryEntry[];
  performance: PerformanceData | null;
  refreshPerformance: () => void;
  logs: string[];
  debugOpen: boolean;
  setLogModalOpen: (open: boolean) => void;
}

/**
 * 캐릭터의 내부 상태를 드러내는 패널 모음.
 *
 * 응답만 보여주는 챗봇 UI와 달리 "왜 그렇게 답했는지"를 확인할 수 있게 하는 것이
 * 이 사이드바의 목적이다.
 */
export function Sidebar({
  emotion,
  memoryStats,
  memories,
  performance,
  refreshPerformance,
  logs,
  debugOpen,
  setLogModalOpen,
}: Props) {
  return (
  <aside className="hidden lg:flex flex-col w-72 border-l shrink-0">
    <div className="p-4 flex-1 overflow-auto space-y-4">
      <Card className="border border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">감정 상태</CardTitle>
        </CardHeader>
        <CardContent>
          <EmotionPanel emotion={emotion} />
        </CardContent>
      </Card>

      <Card className="border border-border">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">기억</CardTitle>
        </CardHeader>
        <CardContent>
          <MemoryPanel stats={memoryStats} memories={memories} />
        </CardContent>
      </Card>

      {debugOpen && (
        <>
          <Card className="border border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">성능</CardTitle>
            </CardHeader>
            <CardContent>
              <PerformancePanel performance={performance} onRefresh={refreshPerformance} />
            </CardContent>
          </Card>

          <Card className="border border-border">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-sm">시스템 로그</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-xs"
                  onClick={() => setLogModalOpen(true)}
                >
                  전체 보기
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {/* 최근 5줄 미리보기 */}
              <div className="font-mono text-[10px] leading-4 text-muted-foreground space-y-0.5">
                {logs.length === 0 ? (
                  <p>로그 없음</p>
                ) : (
                  logs.slice(-5).map((line, i) => (
                    <div key={i} className="truncate">{stripAnsi(line)}</div>
                  ))
                )}
              </div>
              <p className="text-[10px] text-muted-foreground mt-2">총 {logs.length}줄</p>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  </aside>
  );
}
