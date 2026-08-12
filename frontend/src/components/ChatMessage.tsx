import type { Message } from "@/types";
import { PENDING_ID } from "@/constants";
import { stripAnsi } from "@/lib/format";

export function ChatMessage({
  message,
  isSending,
}: {
  message: Message;
  isSending: boolean;
}) {
  const isUser = message.role === "user";
  const isLoading = isSending && message.id === PENDING_ID && !message.content;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        {!isUser && message.debugLogs && message.debugLogs.length > 0 && (
          <div className="mb-1 opacity-50 hover:opacity-80 transition-opacity">
            {message.debugLogs.map((log, i) => (
              <div key={i} className="text-[9px] text-muted-foreground leading-tight">
                {stripAnsi(log)}
              </div>
            ))}
          </div>
        )}
        <div
          className={`rounded-lg px-4 py-2 ${
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted text-muted-foreground"
          }`}
        >
          {isLoading ? (
            <div className="flex items-center gap-1.5 py-1">
              <div className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.3s]" />
              <div className="w-2 h-2 bg-current rounded-full animate-bounce [animation-delay:-0.15s]" />
              <div className="w-2 h-2 bg-current rounded-full animate-bounce" />
            </div>
          ) : (
            <>
              <p className="whitespace-pre-wrap text-sm">{message.content}</p>
              <p className="text-xs opacity-70 mt-1">
                {message.timestamp.toLocaleTimeString()}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
