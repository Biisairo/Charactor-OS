import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { CharacterInfo } from "@/types";

interface Props {
  characters: CharacterInfo[];
  activeCharacter: string;
  switching: boolean;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
  onCreate: (name: string, identity: string) => Promise<boolean>;
}

/**
 * 캐릭터 선택·생성·삭제 드롭다운.
 *
 * 열림 상태와 생성 폼 입력은 이 컴포넌트 밖에서 쓸 일이 없으므로 안에 둔다.
 * 바깥 클릭으로 닫는 처리도 여기 함께 둔다 — 상태와 그 정리는 붙어 있어야 한다.
 */
export function CharacterSwitcher({
  characters,
  activeCharacter,
  switching,
  onSwitch,
  onDelete,
  onCreate,
}: Props) {
  const [showCharManager, setShowCharManager] = useState(false);
  const [showNewCharacter, setShowNewCharacter] = useState(false);
  const [newName, setNewName] = useState("");
  const [newIdentity, setNewIdentity] = useState("");

  useEffect(() => {
    if (!showCharManager) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-char-manager]")) {
        setShowCharManager(false);
        setShowNewCharacter(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showCharManager]);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    if (await onCreate(newName, newIdentity)) {
      setNewName("");
      setNewIdentity("");
      setShowNewCharacter(false);
    }
  };

  const switchCharacter = onSwitch;
  const deleteCharacter = onDelete;

  return (
    <div className="relative" data-char-manager>
      <button
        onClick={() => setShowCharManager(!showCharManager)}
        disabled={switching}
        className="flex items-center gap-1.5 h-8 rounded-md border border-input bg-background px-3 text-sm font-medium hover:bg-accent transition-colors disabled:opacity-50"
      >
        {switching ? (
          <>
            <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
            <span>전환 중...</span>
          </>
        ) : (
          <>
            <span>{characters.find(c => c.id === activeCharacter)?.name || "캐릭터 선택"}</span>
            <span className="text-xs text-muted-foreground">▼</span>
          </>
        )}
      </button>
      {showCharManager && !switching && (
        <div className="absolute top-full left-0 mt-1 z-50 bg-popover border rounded-lg shadow-lg p-1 min-w-[220px]">
          {characters.map((c) => (
            <div
              key={c.id}
              className={`flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors ${
                c.id === activeCharacter
                  ? "bg-primary text-primary-foreground"
                  : "hover:bg-muted cursor-pointer"
              }`}
            >
              <button
                className="flex-1 text-left truncate"
                onClick={() => {
                  if (c.id !== activeCharacter) switchCharacter(c.id);
                  setShowCharManager(false);
                }}
              >
                {c.name}
                {c.identity && (
                  <span className={`block text-xs truncate ${
                    c.id === activeCharacter ? "opacity-80" : "text-muted-foreground"
                  }`}>
                    {c.identity}
                  </span>
                )}
              </button>
              {c.id !== activeCharacter && (
                <button
                  className="ml-2 p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                  title="삭제"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteCharacter(c.id);
                  }}
                >
                  ×
                </button>
              )}
            </div>
          ))}
          <div className="border-t mt-1 pt-1">
            {showNewCharacter ? (
              <div className="px-2 py-2 space-y-2">
                <Input
                  placeholder="캐릭터 이름"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="h-7 text-xs"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && newName.trim()) handleCreate();
                    if (e.key === "Escape") {
                      setShowNewCharacter(false);
                      setNewName("");
                      setNewIdentity("");
                    }
                  }}
                />
                <Input
                  placeholder="한줄 소개 (선택)"
                  value={newIdentity}
                  onChange={(e) => setNewIdentity(e.target.value)}
                  className="h-7 text-xs"
                />
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    className="h-7 flex-1 text-xs"
                    onClick={handleCreate}
                    disabled={!newName.trim()}
                  >
                    만들기
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-7 text-xs"
                    onClick={() => {
                      setShowNewCharacter(false);
                      setNewName("");
                      setNewIdentity("");
                    }}
                  >
                    취소
                  </Button>
                </div>
              </div>
            ) : (
              <button
                className="flex items-center gap-2 w-full px-3 py-2 rounded-md text-sm hover:bg-muted transition-colors"
                onClick={() => setShowNewCharacter(true)}
              >
                <span>+</span>
                <span>새 캐릭터</span>
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
