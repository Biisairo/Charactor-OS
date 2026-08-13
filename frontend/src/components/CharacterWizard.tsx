// 새 캐릭터 질문지 위저드.
//
// "새 캐릭터 만들기"와 "질문지로 열기"(기존 캐릭터 수정) 두 모드가 있다.
// 응답은 모두 클라이언트에 모았다가 마지막에 한 번에 서버로 보낸다 — 중간에
// 반쯤 만든 캐릭터가 생기지 않는다. 작성 중 내용은 localStorage에 자동 저장해
// 실수로 닫아도 되살릴 수 있다.

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  BasicStep,
  BackgroundStep,
  BehaviorStep,
  EmotionTriggersStep,
  FreeformStep,
  GraphStep,
  InnerWorldStep,
  LocationsStep,
  PersonalityStep,
  PersonaExamplesStep,
  RelationshipsStep,
  ScenarioStep,
  SpeakingStyleStep,
  SummaryPanel,
  TimelineStep,
  ValuesStep,
  WorldStep,
} from "@/components/CharacterWizardSteps";
import { createEmptyDraft, isDraftDirty } from "@/lib/characterDraft";
import type {
  CharacterDraft,
  CharacterDraftExamples,
  CharacterDraftKnowledge,
  CharacterDraftPersona,
} from "@/types";

const STEPS: { title: string; hint: string }[] = [
  { title: "기본 정보", hint: "이름 · 한 줄 소개 · 첫 메시지 · 나이 · 성별 · 직업" },
  { title: "성격", hint: "성격 특성과 Big Five 점수" },
  { title: "말투", hint: "톤 · 어휘 · 문장 패턴 · 말버릇 · 샘플 문장" },
  { title: "가치관", hint: "캐릭터가 지키는 핵심 가치" },
  { title: "배경", hint: "과거 이야기 · 좋아함 · 싫어함 · 두려움 · 목표" },
  { title: "행동 지침", hint: "상황별 반응 · 주제별 태도 · 절대 규칙" },
  { title: "감정 트리거", hint: "특정 주제가 나왔을 때의 감정 변화" },
  { title: "관계 설정", hint: "사용자와의 관계가 가장 중요 · 그 외 주요 인물" },
  { title: "내면 상태", hint: "숨기는 생각과 감정 · 비밀 · 약점" },
  { title: "대화 예시", hint: "핵심 성격을 보여주는 예시 대화" },
  { title: "세계관", hint: "캐릭터가 사는 세계의 이름 · 시대 · 규칙 · 메타 인식" },
  { title: "장소", hint: "이야기의 배경이 되는 공간" },
  { title: "관계 그래프", hint: "세계 안 인물들 사이의 관계" },
  { title: "타임라인", hint: "인생의 주요 사건을 시간순으로" },
  { title: "추가 지식", hint: "위 분류에 안 들어가는 자유 서술" },
  { title: "시나리오 예시", hint: "상황별 few-shot 대화 — 말투 일관성의 핵심" },
];

const STORAGE_KEY = "char-wizard:create";

function renderStep(
  step: number,
  draft: CharacterDraft,
  setDraft: (fn: (d: CharacterDraft) => CharacterDraft) => void,
) {
  const setPersona = (persona: CharacterDraftPersona) => setDraft((d) => ({ ...d, persona }));
  const setKnowledge = (knowledge: CharacterDraftKnowledge) =>
    setDraft((d) => ({ ...d, knowledge }));
  const setExamples = (examples: CharacterDraftExamples) =>
    setDraft((d) => ({ ...d, examples }));

  switch (step) {
    case 0:
      return <BasicStep value={draft.persona} onChange={setPersona} />;
    case 1:
      return <PersonalityStep value={draft.persona} onChange={setPersona} />;
    case 2:
      return <SpeakingStyleStep value={draft.persona} onChange={setPersona} />;
    case 3:
      return <ValuesStep value={draft.persona} onChange={setPersona} />;
    case 4:
      return <BackgroundStep value={draft.persona} onChange={setPersona} />;
    case 5:
      return <BehaviorStep value={draft.persona} onChange={setPersona} />;
    case 6:
      return <EmotionTriggersStep value={draft.persona} onChange={setPersona} />;
    case 7:
      return <RelationshipsStep value={draft.persona} onChange={setPersona} />;
    case 8:
      return <InnerWorldStep value={draft.persona} onChange={setPersona} />;
    case 9:
      return <PersonaExamplesStep value={draft.persona} onChange={setPersona} />;
    case 10:
      return (
        <WorldStep
          value={draft.knowledge}
          persona={draft.persona}
          onPersonaChange={setPersona}
          onChange={setKnowledge}
        />
      );
    case 11:
      return <LocationsStep value={draft.knowledge} onChange={setKnowledge} />;
    case 12:
      return <GraphStep value={draft.knowledge} onChange={setKnowledge} />;
    case 13:
      return <TimelineStep value={draft.knowledge} onChange={setKnowledge} />;
    case 14:
      return <FreeformStep value={draft.knowledge} onChange={setKnowledge} />;
    case 15:
      return (
        <div className="space-y-4">
          <SummaryPanel draft={draft} />
          <ScenarioStep value={draft.examples} onChange={setExamples} />
        </div>
      );
    default:
      return null;
  }
}

interface Props {
  open: boolean;
  /** create: 새 캐릭터 생성 · edit: 기존 캐릭터 질문지 수정 */
  mode: "create" | "edit";
  /** edit 모드에서 서버에서 읽어온 기존 응답. */
  initialDraft?: CharacterDraft;
  onClose: () => void;
  onCreate: (draft: CharacterDraft) => Promise<boolean>;
  onUpdate?: (draft: CharacterDraft) => Promise<boolean>;
}

export function CharacterWizard({ open, mode, initialDraft, onClose, onCreate, onUpdate }: Props) {
  // create 모드는 localStorage에서 이전 작성분을 되살린다. edit 모드는 서버가 원본이다.
  const [state, setState] = useState<{ step: number; draft: CharacterDraft }>(() => {
    if (initialDraft) return { step: 0, draft: initialDraft };
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved) as { step?: number; draft?: CharacterDraft };
        if (parsed.draft) return { step: Math.min(parsed.step ?? 0, STEPS.length - 1), draft: parsed.draft };
      }
    } catch {
      // 손상된 저장분은 무시하고 빈 응답지로 시작한다.
    }
    return { step: 0, draft: createEmptyDraft() };
  });
  const { step, draft } = state;
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const saveTimer = useRef<number | null>(null);

  // 작성 중 내용을 localStorage에 자동 저장 (create 모드만 — 서버에 없는 유일한 사본).
  useEffect(() => {
    if (!open || mode !== "create") return;
    if (saveTimer.current) window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ step, draft }));
    }, 300);
    return () => {
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
    };
  }, [open, mode, step, draft]);

  const requestClose = () => {
    if (submitting) return;
    // create는 빈 응답지, edit은 서버에서 읽은 원본이 기준 — 원본 대비 변경이 있을 때만 묻는다.
    const baseline = mode === "edit" ? initialDraft : undefined;
    if (isDraftDirty(draft, baseline) && !window.confirm("작성 중인 내용이 사라집니다. 닫을까요?")) return;
    onClose();
  };

  // Esc · Enter 단축키. Enter는 입력 폼에서 다음 단계로 — textarea에서는 새 줄.
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        requestClose();
        return;
      }
      if (e.key === "Enter" && !submitting) {
        const target = e.target as HTMLElement;
        if (target.tagName === "TEXTAREA") return;
        const canNext = step > 0 || draft.persona.name.trim().length > 0;
        if (step < STEPS.length - 1 && canNext) {
          e.preventDefault();
          setState((s) => ({ ...s, step: s.step + 1 }));
        } else if (step === STEPS.length - 1 && canNext) {
          e.preventDefault();
          void handleSubmit();
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, submitting, step, draft]);

  if (!open) return null;

  const total = STEPS.length;
  const meta = STEPS[step];
  const isFirst = step === 0;
  const isLast = step === total - 1;
  const canNext = step > 0 || draft.persona.name.trim().length > 0;

  // 스텝 폼은 draft만 갱신한다 — step은 여기가 갖는다.
  const setDraft = (fn: (d: CharacterDraft) => CharacterDraft) =>
    setState((s) => ({ ...s, draft: fn(s.draft) }));

  const handleSubmit = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const ok = mode === "edit" ? await onUpdate?.(draft) : await onCreate(draft);
      if (ok) {
        if (mode === "create") localStorage.removeItem(STORAGE_KEY);
        onClose();
      } else {
        setError(mode === "edit" ? "캐릭터 저장에 실패했습니다." : "캐릭터 생성에 실패했습니다. 서버 상태를 확인하세요.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  const startOver = () => {
    if (!window.confirm("작성 내용을 버리고 처음부터 시작할까요?")) return;
    localStorage.removeItem(STORAGE_KEY);
    setState({ step: 0, draft: createEmptyDraft() });
    setError(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div className="flex flex-col w-full max-w-3xl h-[90vh] rounded-xl border bg-background shadow-xl overflow-hidden">
        {/* 헤더 — 제목 · 단계 칩 · 진행 바 */}
        <header className="border-b px-6 pt-4 pb-3 space-y-3 shrink-0">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-semibold">
                {mode === "edit" ? "캐릭터 수정" : "새 캐릭터 만들기"}
              </h2>
              <p className="text-sm text-muted-foreground">
                {step + 1} / {total} — {meta.title}
              </p>
            </div>
            <div className="flex items-center gap-1">
              {mode === "create" && (
                <Button variant="ghost" size="sm" className="h-8 text-xs" onClick={startOver}>
                  새로 시작
                </Button>
              )}
              <button
                onClick={requestClose}
                disabled={submitting}
                className="h-8 w-8 rounded-md hover:bg-accent transition-colors text-muted-foreground disabled:opacity-50"
                title="닫기"
              >
                ×
              </button>
            </div>
          </div>
          {/* 단계 칩 — 어디쯤인지 보고 점프할 수 있다 */}
          <div className="flex gap-1.5 overflow-x-auto pb-1 -mx-1 px-1">
            {STEPS.map((s, i) => (
              <button
                key={s.title}
                onClick={() => setState((cur) => ({ ...cur, step: i }))}
                className={`shrink-0 h-6 rounded-full px-2.5 text-xs transition-colors ${
                  i === step
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted text-muted-foreground hover:bg-accent hover:text-foreground"
                }`}
              >
                {i + 1}. {s.title}
              </button>
            ))}
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-300"
              style={{ width: `${((step + 1) / total) * 100}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">{meta.hint}</p>
        </header>

        {/* 본문 — 현재 단계 폼 */}
        <div className="flex-1 overflow-y-auto px-6 py-5">
          {error && (
            <p className="mb-3 rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          {renderStep(step, draft, setDraft)}
        </div>

        {/* 푸터 — 이전 / 맨 끝으로 / 다음·저장 */}
        <footer className="border-t px-6 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              onClick={() => setState((s) => ({ ...s, step: Math.max(0, s.step - 1) }))}
              disabled={isFirst || submitting}
            >
              이전
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setState((s) => ({ ...s, step: total - 1 }))}
              disabled={isLast || submitting}
            >
              맨 끝으로
            </Button>
          </div>
          {isLast ? (
            <Button onClick={handleSubmit} disabled={submitting}>
              {submitting ? "저장 중..." : mode === "edit" ? "저장" : "캐릭터 생성"}
            </Button>
          ) : (
            <Button
              onClick={() => setState((s) => ({ ...s, step: Math.min(total - 1, s.step + 1) }))}
              disabled={!canNext || submitting}
            >
              다음 {!canNext && "(이름 입력 필요)"}
            </Button>
          )}
        </footer>
      </div>
    </div>
  );
}
