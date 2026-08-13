import { useEffect, useState } from "react";

import type { CharacterInfo } from "@/types";

interface Props {
  characters: CharacterInfo[];
  activeCharacter: string;
  switching: boolean;
  onSwitch: (id: string) => void;
  onDelete: (id: string) => void;
  /** 기존 캐릭터를 질문지 위저드로 다시 연다. */
  onEdit: (id: string) => void;
  /** 새 캐릭터 질문지 위저드를 연다. 생성 응답은 위저드가 위임받아 처리한다. */
  onStartWizard: () => void;
}

/**
 * 캐릭터 선택·삭제 드롭다운.
 *
 * 열림 상태는 이 컴포넌트 밖에서 쓸 일이 없으므로 안에 둔다.
 * 바깥 클릭으로 닫는 처리도 여기 함께 둔다 — 상태와 그 정리는 붙어 있어야 한다.
 * 새 캐릭터 생성은 인라인 폼 대신 질문지 위저드(CharacterWizard)로 위임한다.
 */
export function CharacterSwitcher({
  characters,
  activeCharacter,
  switching,
  onSwitch,
  onDelete,
  onEdit,
  onStartWizard,
}: Props) {
  const [showCharManager, setShowCharManager] = useState(false);

  useEffect(() => {
    if (!showCharManager) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-char-manager]")) {
        setShowCharManager(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showCharManager]);

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
              <button
                className="ml-1 p-1 rounded hover:bg-accent/60 text-muted-foreground hover:text-foreground transition-colors"
                title="질문지로 열기"
                onClick={(e) => {
                  e.stopPropagation();
                  setShowCharManager(false);
                  onEdit(c.id);
                }}
              >
                ✎
              </button>
            </div>
          ))}
          <div className="border-t mt-1 pt-1">
            <button
              className="flex items-center gap-2 w-full px-3 py-2 rounded-md text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              onClick={() => {
                setShowCharManager(false);
                onStartWizard();
              }}
            >
              <span>＋</span>
              <span>새 캐릭터 만들기</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
