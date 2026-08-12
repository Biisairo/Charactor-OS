// 대화 화면의 상태와 전송 동작.
//
// 대화 경로는 POST /api/chat 하나다. 이전에는 WebSocket으로 토큰을 받아
// 점진 렌더했으나, 그 경로에는 Reflection 검토가 없어 검토를 거치지 않은
// 초안이 사용자에게 도달했다. 스트리밍이 흘려보내는 것은 확정 응답이 아니라
// 검토기가 반려할지 모르는 초안이므로 둘은 양립하지 않는다 (TASK-11).

import { useCallback, useRef, useState } from "react";

import { PENDING_ID } from "@/constants";
import { apiGet, apiPost } from "@/lib/api";
import type { Message } from "@/types";

interface Options {
  /** 턴이 끝났을 때 (성공·실패 모두) 서버 상태를 다시 읽기 위한 훅. */
  onTurnComplete: () => void;
  onError: (message: string | null) => void;
}

export function useChat({ onTurnComplete, onError }: Options) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const sendMessage = useCallback(async () => {
    const currentInput = inputRef.current?.value || input;
    if (!currentInput.trim() || isSending) return;

    const debugData = await apiGet<{ logs: string[] }>("/api/debug").catch(() => null);
    const logStart = debugData?.logs.length ?? 0;

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content: currentInput,
        timestamp: new Date(),
      },
      // 응답을 기다리는 동안 자리를 잡아두는 빈 메시지 (로딩 표시용)
      {
        id: PENDING_ID,
        role: "assistant",
        content: "",
        timestamp: new Date(),
      },
    ]);
    setInput("");
    setIsSending(true);
    onError(null);

    try {
      const data = await apiPost<{ response: string; emotion: Record<string, number> }>(
        "/api/chat",
        { message: currentInput },
      );
      const newLogs = await apiGet<{ logs: string[] }>("/api/debug")
        .then((d) => d.logs.slice(logStart))
        .catch(() => []);

      setMessages((prev) =>
        prev.map((m) =>
          m.id === PENDING_ID
            ? { ...m, id: `${Date.now()}-a`, content: data.response, debugLogs: newLogs }
            : m,
        ),
      );
    } catch {
      // 실패한 턴은 캐릭터 발화가 아니다. 자리표시 메시지를 지우고 오류로 알린다.
      setMessages((prev) => prev.filter((m) => m.id !== PENDING_ID));
      onError("응답 생성에 실패했습니다. 잠시 후 다시 시도해 주세요.");
    } finally {
      setIsSending(false);
      onTurnComplete();
    }
  }, [input, isSending, onTurnComplete, onError]);

  const exportChat = useCallback(() => {
    const text = messages.map((m) => `[${m.role}] ${m.content}`).join("\n\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  return { messages, setMessages, input, setInput, isSending, inputRef, sendMessage, exportChat };
}
