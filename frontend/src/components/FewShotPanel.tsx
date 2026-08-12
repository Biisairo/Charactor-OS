import { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import type { FewShotGroup } from "@/types";
import { apiGet } from "@/lib/api";

export function FewShotPanel() {
  const [groups, setGroups] = useState<FewShotGroup[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<{ groups: FewShotGroup[] }>("/api/fewshot")
      .then((data) => setGroups(data.groups))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-sm text-muted-foreground">로딩 중...</p>;

  return (
    <div className="space-y-3 max-h-[50vh] overflow-y-auto">
      {groups.length === 0 && (
        <p className="text-sm text-muted-foreground">예시 없음</p>
      )}
      {groups.map((group) => (
        <div key={group.tag} className="border rounded-md p-3 space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant="secondary">{group.tag}</Badge>
            <span className="text-xs text-muted-foreground">{group.count}개</span>
          </div>
          {group.examples.map((ex, i) => (
            <div key={i} className="pl-3 border-l-2 border-muted text-sm space-y-1">
              <div>
                <span className="text-muted-foreground">사용자: </span>
                {ex.user}
              </div>
              <div>
                <span className="text-muted-foreground">캐릭터: </span>
                {ex.character}
              </div>
              {ex.emotion_state.length > 0 && (
                <div className="flex gap-1">
                  {ex.emotion_state.map((e) => (
                    <Badge key={e} variant="outline" className="text-xs">
                      {e}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}
