// 화면 전반에서 쓰는 표시 규칙.

export const STAGE_LABELS: Record<string, string> = {
  context: "컨텍스트 수집",
  response: "응답 생성",
  postprocess: "후처리",
};

// LLM 호출 지점(계측 라벨) → 화면 표기. `src/character_os.py`의 `_meter.wrap()` 라벨과 맞춘다.
export const LLM_LABELS: Record<string, string> = {
  response: "응답 생성",
  reflection: "검토",
  emotion: "감정 갱신",
  memory: "기억 추출",
};

export const ERROR_KEYWORDS = ["실패", "오류", "error", "FAIL"];

// 응답을 기다리는 동안 자리를 잡아두는 assistant 메시지의 ID
export const PENDING_ID = "pending";

export const LOG_COLORS: { pattern: RegExp; cls: string }[] = [
  { pattern: /실패|오류|error|FAIL/i, cls: "text-red-400" },
  { pattern: /\[Orchestrator\]/, cls: "text-blue-400" },
  { pattern: /\[ReAct\]/, cls: "text-cyan-400" },
  { pattern: /\[Response\]/, cls: "text-yellow-400" },
  { pattern: /\[PostProcess\]/, cls: "text-green-400" },
  { pattern: /Stage [123]/, cls: "text-purple-400" },
  { pattern: /={4,}/, cls: "text-muted-foreground" },
  { pattern: /-{4,}/, cls: "text-muted-foreground" },
];
