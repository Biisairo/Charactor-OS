import { useState } from "react";
import { Input } from "@/components/ui/input";
import { ERROR_KEYWORDS, LOG_COLORS } from "@/constants";
import { stripAnsi } from "@/lib/format";

export function LogViewer({
  logs,
  logScrollRef,
}: {
  logs: string[];
  logScrollRef: React.RefObject<HTMLDivElement | null>;
}) {
  const [searchText, setSearchText] = useState("");
  const [showErrorOnly, setShowErrorOnly] = useState(false);

  const filtered = logs.filter((line) => {
    const stripped = stripAnsi(line);
    if (showErrorOnly && !ERROR_KEYWORDS.some((k) => stripped.includes(k))) return false;
    if (searchText && !stripped.toLowerCase().includes(searchText.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="flex flex-col h-full space-y-2">
      {/* 검색 + 필터 */}
      <div className="flex gap-2">
        <Input
          placeholder="로그 검색..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          className="h-8 text-xs flex-1"
        />
        <button
          className={`px-3 h-8 rounded-md text-xs font-medium transition-colors ${
            showErrorOnly
              ? "bg-destructive text-destructive-foreground"
              : "bg-muted text-muted-foreground hover:bg-muted/80"
          }`}
          onClick={() => setShowErrorOnly(!showErrorOnly)}
        >
          에러
        </button>
      </div>
      {/* 로그 수 */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>{filtered.length}줄 표시</span>
        <span>총 {logs.length}줄</span>
      </div>
      {/* 로그 내용 */}
      <div
        ref={logScrollRef}
        className="flex-1 min-h-0 overflow-auto rounded-md bg-muted/50 border"
      >
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
            로그 없음
          </div>
        ) : (
          <div className="p-1">
            {filtered.map((line, i) => {
              const stripped = stripAnsi(line);
              const colorRule = LOG_COLORS.find((c) => c.pattern.test(stripped));
              const isError = ERROR_KEYWORDS.some((k) => stripped.includes(k));
              return (
                <div
                  key={i}
                  className={`font-mono text-[11px] leading-[18px] px-2 py-0.5 rounded hover:bg-muted transition-colors whitespace-pre-wrap break-all ${
                    isError ? "bg-red-500/5" : ""
                  } ${colorRule?.cls || "text-foreground"}`}
                >
                  {stripped}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
