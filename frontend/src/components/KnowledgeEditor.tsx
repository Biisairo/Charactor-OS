import { useState } from "react";
import { Button } from "@/components/ui/button";
import type { KnowledgeEntry } from "@/types";

export function KnowledgeEditor({
  entries,
  onLoad,
  onSave,
}: {
  entries: KnowledgeEntry[];
  onLoad: (name: string) => Promise<string>;
  onSave: (name: string, content: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSelect = async (name: string) => {
    setSelected(name);
    setLoading(true);
    try {
      const data = await onLoad(name);
      setContent(data);
    } catch {
      setContent("(로드 실패)");
    }
    setLoading(false);
  };

  const typeLabels: Record<string, string> = {
    world: "🌍 세계관",
    character: "👤 캐릭터",
    relationships: "🔗 관계",
    timeline: "📅 타임라인",
    locations: "📍 장소",
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        {entries.map((entry) => (
          <div
            key={entry.name}
            className={`p-2 rounded cursor-pointer text-sm ${
              selected === entry.name
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            }`}
            onClick={() => handleSelect(entry.name)}
          >
            <div className="font-medium">
              {entry.type ? typeLabels[entry.type] || entry.type : ""}
              {" "}
              {entry.name}
            </div>
            <div className="text-xs opacity-70">
              {entry.count !== undefined
                ? `${entry.count}개`
                : entry.size !== undefined
                  ? `${entry.size}자`
                  : ""}
            </div>
          </div>
        ))}
        {entries.length === 0 && (
          <p className="text-muted-foreground text-sm">지식 파일 없음</p>
        )}
      </div>

      {selected && (
        <div className="space-y-2">
          <label className="text-sm font-medium">{selected}</label>
          {loading ? (
            <p className="text-sm text-muted-foreground">로딩 중...</p>
          ) : (
            <>
              <textarea
                className="w-full min-h-[200px] rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
              <Button
                onClick={() => onSave(selected, content)}
                className="w-full"
              >
                저장
              </Button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
