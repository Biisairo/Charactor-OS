import { useCallback, useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { CharacterSwitcher } from "@/components/CharacterSwitcher";
import { ChatPanel } from "@/components/ChatPanel";
import { LogModal } from "@/components/LogModal";
import { SettingsSheet } from "@/components/SettingsSheet";
import { Sidebar } from "@/components/Sidebar";
import { useChat } from "@/hooks/useChat";
import { useDarkMode } from "@/hooks/useDarkMode";
import { useServerState } from "@/hooks/useServerState";
import type { Message } from "@/types";

/**
 * 앱 셸 — 화면 조각을 배치하고 훅을 잇는다.
 *
 * 서버에서 오는 상태는 `useServerState`, 대화는 `useChat`이 갖는다.
 * 여기 남는 상태는 **어느 패널이 열려 있는가** 뿐이다.
 */
export default function App() {
  const { dark, setDark } = useDarkMode();

  const [sheetOpen, setSheetOpen] = useState(false);
  const [debugOpen, setDebugOpen] = useState(true);
  const [logModalOpen, setLogModalOpen] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);
  const logScrollRef = useRef<HTMLDivElement>(null);

  // 두 훅은 서로의 내부 상태를 모른다. 서버가 대화 기록을 읽어오거나 대화를
  // 비워야 할 때만 이 ref를 통해 알린다. `useServerState`가 먼저 선언되어야
  // `useChat`이 그 갱신 함수를 받을 수 있어, 한쪽 방향은 ref로 잇는다.
  const setMessagesRef = useRef<(messages: Message[]) => void>(() => {});

  const server = useServerState({
    onHistoryLoaded: useCallback((loaded: Message[]) => setMessagesRef.current(loaded), []),
    onConversationCleared: useCallback(() => setMessagesRef.current([]), []),
  });

  const { refreshPerformance, fetchLogs, setError } = server;
  const chat = useChat({
    onTurnComplete: useCallback(() => {
      refreshPerformance();
      fetchLogs();
    }, [refreshPerformance, fetchLogs]),
    onError: setError,
  });

  // 렌더 중에 잇는다. 서버 조회는 effect에서 시작하므로 그 전에 채워진다.
  setMessagesRef.current = chat.setMessages;

  // 새 메시지가 오면 아래로 붙인다.
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [chat.messages]);

  // 로그는 이미 아래를 보고 있을 때만 따라 내린다 — 위를 읽는 중이면 방해가 된다.
  useEffect(() => {
    const el = logScrollRef.current;
    if (!el) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 100) {
      el.scrollTop = el.scrollHeight;
    }
  }, [server.logs]);

  return (
    <div className="flex h-screen bg-background text-foreground">
      <div className="flex-1 flex flex-col min-w-0">
        <header className="border-b px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold">Character OS</h1>
            <CharacterSwitcher
              characters={server.characters}
              activeCharacter={server.activeCharacter}
              switching={server.switching}
              onSwitch={server.switchCharacter}
              onDelete={server.deleteCharacter}
              onCreate={server.createCharacter}
            />
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDark(!dark)}
              title={dark ? "라이트 모드" : "다크 모드"}
            >
              {dark ? "☀️" : "🌙"}
            </Button>
            <Button
              variant={debugOpen ? "default" : "ghost"}
              size="sm"
              onClick={() => {
                server.toggleDebug();
                setDebugOpen(!debugOpen);
              }}
              title="디버그 모드"
            >
              🐛
            </Button>
            <Button variant="outline" size="sm" onClick={chat.exportChat}>
              내보내기
            </Button>
            <SettingsSheet
              open={sheetOpen}
              onOpenChange={setSheetOpen}
              emotion={server.emotion}
              memoryStats={server.memoryStats}
              memories={server.memories}
              persona={server.persona}
              knowledgeEntries={server.knowledgeEntries}
              savePersona={server.savePersona}
              saveKnowledge={server.saveKnowledge}
              loadKnowledge={server.loadKnowledge}
              resetCharacter={server.resetCharacter}
            />
          </div>
        </header>

        <ChatPanel
          messages={chat.messages}
          input={chat.input}
          setInput={chat.setInput}
          isSending={chat.isSending}
          inputRef={chat.inputRef}
          sendMessage={chat.sendMessage}
          scrollRef={scrollRef}
          error={server.error}
          setError={server.setError}
        />
      </div>

      <Sidebar
        emotion={server.emotion}
        memoryStats={server.memoryStats}
        memories={server.memories}
        performance={server.performance}
        refreshPerformance={server.refreshPerformance}
        logs={server.logs}
        debugOpen={debugOpen}
        setLogModalOpen={setLogModalOpen}
      />

      {logModalOpen && (
        <LogModal
          logs={server.logs}
          logScrollRef={logScrollRef}
          setLogModalOpen={setLogModalOpen}
        />
      )}
    </div>
  );
}
