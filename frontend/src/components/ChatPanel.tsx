import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChatMessage } from "@/components/ChatMessage";
import type { Message } from "@/types";

interface Props {
  messages: Message[];
  input: string;
  setInput: (value: string) => void;
  isSending: boolean;
  inputRef: React.RefObject<HTMLInputElement | null>;
  sendMessage: () => void;
  scrollRef: React.RefObject<HTMLDivElement | null>;
  error: string | null;
  setError: (value: string | null) => void;
}

/** 오류 배너 · 메시지 목록 · 입력창. */
export function ChatPanel({
  messages,
  input,
  setInput,
  isSending,
  inputRef,
  sendMessage,
  scrollRef,
  error,
  setError,
}: Props) {
  return (
    <>
  {/* Error banner */}
  {error && (
    <div className="bg-destructive/10 border-b border-destructive/30 px-4 py-2 flex items-center justify-between shrink-0">
      <span className="text-sm text-destructive">{error}</span>
      <Button
        variant="ghost"
        size="sm"
        className="h-6 px-2 text-xs text-destructive"
        onClick={() => setError(null)}
      >
        닫기
      </Button>
    </div>
  )}

  {/* Messages */}
  <div className="flex-1 overflow-auto p-4" ref={scrollRef}>
    {messages.length === 0 ? (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <p>대화를 시작하세요</p>
      </div>
    ) : (
      messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} isSending={isSending} />
      ))
    )}
  </div>

  {/* Input */}
  <div className="border-t p-4 shrink-0">
    <div className="flex gap-2 max-w-3xl mx-auto">
      <Input
        ref={inputRef}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
          if (e.key === "Enter" && !e.nativeEvent.isComposing) {
            e.preventDefault();
            sendMessage();
          }
        }}
        placeholder={isSending ? "응답을 기다리는 중..." : "메시지를 입력하세요..."}
        disabled={isSending}
        className={`flex-1 ${isSending ? "opacity-60" : ""}`}
      />
      <Button onClick={sendMessage} disabled={isSending} className="min-w-[70px]">
        {isSending ? (
          <div className="flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.3s]" />
            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.15s]" />
            <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" />
          </div>
        ) : (
          "전송"
        )}
      </Button>
    </div>
    {isSending && (
      <p className="text-xs text-muted-foreground text-center mt-2 animate-pulse">
        캐릭터가 응답을 생성하고 있습니다...
      </p>
    )}
  </div>
    </>
  );
}
