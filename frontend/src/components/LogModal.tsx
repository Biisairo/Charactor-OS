import { LogViewer } from "@/components/LogViewer";

interface Props {
  logs: string[];
  logScrollRef: React.RefObject<HTMLDivElement | null>;
  setLogModalOpen: (open: boolean) => void;
}

/** 시스템 로그 전체 보기. */
export function LogModal({ logs, logScrollRef, setLogModalOpen }: Props) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={() => setLogModalOpen(false)}
    >
      <div
        className="bg-background border rounded-lg shadow-xl w-[90vw] max-w-[1200px] h-[85vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* 모달 헤더 */}
        <div className="flex items-center justify-between px-6 py-4 border-b shrink-0">
          <h2 className="text-lg font-semibold">시스템 로그</h2>
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">총 {logs.length}줄</span>
            <button
              className="text-muted-foreground hover:text-foreground text-xl leading-none"
              onClick={() => setLogModalOpen(false)}
            >
              ✕
            </button>
          </div>
        </div>
        {/* 모달 본체 */}
        <div className="flex-1 overflow-hidden p-4">
          <LogViewer
            logs={logs}
            logScrollRef={logScrollRef}
          />
        </div>
      </div>
    </div>
  );
}
