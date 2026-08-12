import { useState } from "react";
import { Button } from "@/components/ui/button";

export function ResetPanel({
  onReset,
}: {
  onReset: (opts: { memory: boolean; emotion: boolean; history: boolean }) => void;
}) {
  const [memory, setMemory] = useState(true);
  const [emotion, setEmotion] = useState(true);
  const [history, setHistory] = useState(false);

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        캐릭터의 상태를 초기화합니다. 이 작업은 되돌릴 수 없습니다.
      </p>
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={memory}
            onChange={(e) => setMemory(e.target.checked)}
          />
          기억 초기화
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={emotion}
            onChange={(e) => setEmotion(e.target.checked)}
          />
          감정 초기화
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={history}
            onChange={(e) => setHistory(e.target.checked)}
          />
          대화 기록 초기화
        </label>
      </div>
      <Button
        variant="destructive"
        onClick={() => onReset({ memory, emotion, history })}
        className="w-full"
      >
        초기화
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
