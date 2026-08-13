// 새 캐릭터 질문지의 각 단계 폼.
//
// 위저드 셸(CharacterWizard)이 단계 메타데이터와 진행 상태를 갖고,
// 여기는 **한 단계의 응답을 받아 draft 조각을 갱신**하는 폼만 둔다.
// 값은 항상 상위에서 내려주고, 변경은 콜백으로 올린다 (controlled).

import { useState } from "react";

import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ArrayInput } from "@/components/ArrayInput";
import { cn } from "@/lib/utils";
import type {
  Big5Scores,
  CharacterDraft,
  CharacterDraftExamples,
  CharacterDraftKnowledge,
  CharacterDraftPersona,
  ScenarioGroup,
} from "@/types";

// ─── 공용 폼 조각 ───────────────────────────────────────────────────────

function Field({
  label,
  hint,
  required,
  children,
}: {
  label: string;
  hint?: string;
  required?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label className="text-sm font-medium">
        {label}
        {required && <span className="ml-1 text-destructive">*</span>}
      </label>
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
      {children}
    </div>
  );
}

function Textarea({ className, ...props }: React.ComponentProps<"textarea">) {
  return (
    <textarea
      className={cn(
        "w-full min-w-0 rounded-lg border border-input bg-transparent px-2.5 py-1.5 text-sm transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30",
        className,
      )}
      {...props}
    />
  );
}

export interface FieldSpec {
  key: string;
  label: string;
  placeholder?: string;
  type?: "text" | "textarea" | "range" | "tags" | "select";
  options?: string[];
  min?: number;
  max?: number;
  step?: number;
  /** range 기본값 — 미지정 시 min(기본 0)이 된다. */
  default?: number;
}

function emptyRow(fields: FieldSpec[]): Record<string, unknown> {
  const row: Record<string, unknown> = {};
  for (const f of fields) {
    if (f.type === "range") row[f.key] = f.default ?? f.min ?? 0;
    else if (f.type === "tags") row[f.key] = [];
    else if (f.type === "select") row[f.key] = f.options?.[0] ?? "";
    else row[f.key] = "";
  }
  return row;
}

function FieldControl({
  field,
  value,
  onChange,
}: {
  field: FieldSpec;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  if (field.type === "range") {
    const num = typeof value === "number" ? value : (field.min ?? 0);
    return (
      <div>
        <div className="flex items-center justify-between">
          <label className="text-xs text-muted-foreground">{field.label}</label>
          <span className="text-xs font-medium tabular-nums">{num.toFixed(2)}</span>
        </div>
        <input
          type="range"
          min={field.min ?? 0}
          max={field.max ?? 1}
          step={field.step ?? 0.1}
          value={num}
          onChange={(e) => onChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
    );
  }
  if (field.type === "tags") {
    const arr = Array.isArray(value) ? (value as string[]) : [];
    return (
      <div>
        <label className="text-xs text-muted-foreground">{field.label} (쉼표로 구분)</label>
        <Input
          value={arr.join(", ")}
          placeholder={field.placeholder}
          onChange={(e) =>
            onChange(e.target.value.split(",").map((s) => s.trim()).filter(Boolean))
          }
        />
      </div>
    );
  }
  if (field.type === "select") {
    return (
      <div>
        <label className="text-xs text-muted-foreground">{field.label}</label>
        <select
          value={String(value ?? field.options?.[0] ?? "")}
          onChange={(e) => onChange(e.target.value)}
          className="h-8 w-full rounded-lg border border-input bg-transparent px-2.5 text-sm outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {(field.options ?? []).map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </div>
    );
  }
  if (field.type === "textarea") {
    return (
      <div className="sm:col-span-2">
        <label className="text-xs text-muted-foreground">{field.label}</label>
        <Textarea
          rows={2}
          value={String(value ?? "")}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
        />
      </div>
    );
  }
  return (
    <div>
      <label className="text-xs text-muted-foreground">{field.label}</label>
      <Input
        value={String(value ?? "")}
        placeholder={field.placeholder}
        onChange={(e) => onChange(e.target.value)}
      />
    </div>
  );
}

/** 목록 형태 응답 — 행 단위로 추가·제거하며 편집한다. 건너뛰면 빈 배열로 남는다. */
export function RowsEditor<T extends object>({
  label,
  hint,
  fields,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  fields: FieldSpec[];
  value: T[];
  onChange: (rows: T[]) => void;
}) {
  const update = (i: number, key: string, v: unknown) => {
    onChange(
      value.map((row, j) => (j === i ? ({ ...row, [key]: v } as T) : row)),
    );
  };
  const remove = (i: number) => onChange(value.filter((_, j) => j !== i));
  const add = () => onChange([...value, emptyRow(fields) as T]);

  return (
    <div className="space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-0.5">
          <h4 className="text-sm font-medium">{label}</h4>
          {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
        </div>
        <Button variant="ghost" size="sm" className="h-6 px-2 text-xs shrink-0" onClick={add}>
          + 항목 추가
        </Button>
      </div>
      {value.length === 0 && (
        <p className="text-xs text-muted-foreground">아직 항목이 없습니다. 필요하면 추가하세요. (선택)</p>
      )}
      {value.map((row, i) => (
        <div key={i} className="rounded-lg border border-input p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">항목 {i + 1}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
              onClick={() => remove(i)}
            >
              제거
            </Button>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            {fields.map((f) => (
              <FieldControl
                key={f.key}
                field={f}
                value={(row as Record<string, unknown>)[f.key]}
                onChange={(v) => update(i, f.key, v)}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

const BIG5_FIELDS: { key: keyof Big5Scores; label: string; desc: string }[] = [
  { key: "openness", label: "개방성", desc: "새로운 경험·아이디어에 얼마나 열려있는가" },
  { key: "conscientiousness", label: "성실성", desc: "계획적이고 책임감이 강한가" },
  { key: "extraversion", label: "외향성", desc: "사람들과의 교류에서 에너지를 얻는가" },
  { key: "agreeableness", label: "친화성", desc: "타인에게 협조적이고 공감적인가" },
  { key: "neuroticism", label: "신경성", desc: "부정적 감정을 자주·강하게 느끼는가" },
];

/** 슬라이더 값에 따른 행동 앵커 — 추상적인 숫자에 감각을 붙인다. */
function big5Anchor(key: keyof Big5Scores, v: number): string {
  if (key === "openness") return v < 0.4 ? "익숙한 것 선호" : v < 0.7 ? "때때로 새로운 시도" : "새 경험에 적극적";
  if (key === "conscientiousness") return v < 0.4 ? "즉흥적·되는 대로" : v < 0.7 ? "때와 일에 따라" : "계획적·철저함";
  if (key === "extraversion") return v < 0.4 ? "혼자서 에너지 충전" : v < 0.7 ? "상황에 따라" : "사람에게서 에너지";
  if (key === "agreeableness") return v < 0.4 ? "직설적·날카로움" : v < 0.7 ? "보통" : "배려심 깊음";
  return v < 0.4 ? "차분·감정 안정" : v < 0.7 ? "보통" : "감정 기복이 큼";
}

function Big5Editor({
  value,
  onChange,
}: {
  value: Big5Scores;
  onChange: (v: Big5Scores) => void;
}) {
  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium">Big Five 성격 모델 (0.0 ~ 1.0)</h4>
      {BIG5_FIELDS.map((f) => (
        <div key={f.key} className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium">{f.label}</span>
            <span className="text-muted-foreground tabular-nums">{value[f.key].toFixed(2)}</span>
          </div>
          <p className="text-xs text-muted-foreground">
            {f.desc} — <span className="text-foreground">{big5Anchor(f.key, value[f.key])}</span>
          </p>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={value[f.key]}
            onChange={(e) => onChange({ ...value, [f.key]: Number(e.target.value) })}
            className="w-full"
          />
        </div>
      ))}
    </div>
  );
}

// ─── 단계 폼 ────────────────────────────────────────────────────────────

/** 1. 기본 정보 */
export function BasicStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona>) => onChange({ ...value, ...patch });
  return (
    <div className="space-y-4">
      <Field label="이름" required hint="필수 — 캐릭터를 식별하는 이름. 저장 디렉토리 이름이 된다.">
        <Input
          value={value.name}
          onChange={(e) => set({ name: e.target.value })}
          placeholder="예: 홍길동"
        />
      </Field>
      <Field label="한 줄 소개" hint="캐릭터를 가장 잘 나타내는 한 문장">
        <Input
          value={value.identity}
          onChange={(e) => set({ identity: e.target.value })}
          placeholder="예: 아버지를 아버지라 부르지 못하는 서자. 조선시대의 의적."
        />
      </Field>
      <Field
        label="첫 메시지"
        hint="사용자가 채팅을 시작하면 캐릭터가 건네는 첫 한마디. (선택이지만 채우면 첫인상이 산다)"
      >
        <Textarea
          rows={2}
          value={value.first_message}
          onChange={(e) => set({ first_message: e.target.value })}
          placeholder="예: 왔어? 앉아 앉아ㅋㅋ 오늘은 좀 늦었네?"
        />
      </Field>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Field label="나이" hint="숫자 또는 나이대">
          <Input
            value={value.age}
            onChange={(e) => set({ age: e.target.value })}
            placeholder="예: 24 또는 20대 초반"
          />
        </Field>
        <Field label="성별">
          <Input value={value.gender} onChange={(e) => set({ gender: e.target.value })} placeholder="예: 남성" />
        </Field>
        <Field label="직업/역할">
          <Input value={value.occupation} onChange={(e) => set({ occupation: e.target.value })} placeholder="예: 의적" />
        </Field>
      </div>
    </div>
  );
}

/** 2. 성격 */
export function PersonalityStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona>) => onChange({ ...value, ...patch });
  return (
    <div className="space-y-5">
      <ArrayInput
        label="성격 특성"
        values={value.personality.traits}
        onChange={(traits) => set({ personality: { ...value.personality, traits } })}
      />
      <p className="text-xs text-muted-foreground -mt-3">
        힌트: 모순되는 특성을 섞으면 캐릭터가 살아난다 — 예: "외향적"과 "혼자 있을 때 가라앉는",
        "친절한"과 "가끔 날카로운".
      </p>
      <Big5Editor
        value={value.personality.big5}
        onChange={(big5) => set({ personality: { ...value.personality, big5 } })}
      />
    </div>
  );
}

/** 3. 말투 */
export function SpeakingStyleStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona>) => onChange({ ...value, ...patch });
  const s = value.speaking_style;
  const setStyle = (patch: Partial<CharacterDraftPersona["speaking_style"]>) =>
    set({ speaking_style: { ...s, ...patch } });
  return (
    <div className="space-y-4">
      <Field label="말투 한 줄 요약">
        <Textarea
          rows={2}
          value={s.summary}
          onChange={(e) => setStyle({ summary: e.target.value })}
          placeholder="예: 빠른 구어체 반말. 리액션이 먼저 나가고 설명이 따라온다."
        />
      </Field>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Field label="전체 톤">
          <Input value={s.tone} onChange={(e) => setStyle({ tone: e.target.value })} placeholder="예: 밝고 경쾌한" />
        </Field>
        <Field label="어휘 수준">
          <Input value={s.vocabulary} onChange={(e) => setStyle({ vocabulary: e.target.value })} placeholder="예: 일상어" />
        </Field>
        <Field label="문장 패턴">
          <Input value={s.sentence_pattern} onChange={(e) => setStyle({ sentence_pattern: e.target.value })} placeholder="예: 짧은 문장 선호" />
        </Field>
        <Field label="이모지 사용">
          <Input value={s.emojis} onChange={(e) => setStyle({ emojis: e.target.value })} placeholder="예: 자주 사용, 거의 안 함" />
        </Field>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ArrayInput
          label="말버릇·추임새"
          values={s.fillers}
          onChange={(fillers) => setStyle({ fillers })}
        />
        <ArrayInput
          label="문미 패턴 (말투 끝맺음)"
          values={s.endings}
          onChange={(endings) => setStyle({ endings })}
        />
      </div>
      <Field
        label="말투 샘플 문장"
        hint="이 말투로 실제로 말하는 것처럼 한 문장. 목소리가 잡히면 나중에 대화 예시 품질이 올라간다."
      >
        <Input
          value={s.sample}
          onChange={(e) => setStyle({ sample: e.target.value })}
          placeholder="예: 아니 그니까 잠깐만, 나 어제 진짜 개캐리했거든? ㅋㅋ"
        />
      </Field>
    </div>
  );
}

/** 4. 가치관 */
export function ValuesStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  return (
    <ArrayInput
      label="핵심 가치관"
      values={value.values}
      onChange={(values) => onChange({ ...value, values })}
    />
  );
}

/** 5. 배경 */
export function BackgroundStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona>) => onChange({ ...value, ...patch });
  return (
    <div className="space-y-4">
      <Field
        label="배경 이야기"
        hint="어디서 자랐는가, 인생을 바꾼 사건(전환점), 지금의 모습이 된 이유, 상처가 된 일. 자유롭게."
      >
        <Textarea
          rows={8}
          value={value.backstory}
          onChange={(e) => set({ backstory: e.target.value })}
          placeholder="예: 조선시대 양반 가문의 서자로 태어났다. 아버지를 아버지라 부르지 못하고..."
        />
      </Field>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <ArrayInput label="좋아하는 것" values={value.likes} onChange={(likes) => set({ likes })} />
        <ArrayInput label="싫어하는 것" values={value.dislikes} onChange={(dislikes) => set({ dislikes })} />
        <ArrayInput label="두려워하는 것" values={value.fears} onChange={(fears) => set({ fears })} />
        <ArrayInput label="현재 목표" values={value.goals} onChange={(goals) => set({ goals })} />
      </div>
    </div>
  );
}

/** 6. 행동 지침 */
export function BehaviorStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona>) => onChange({ ...value, ...patch });
  const b = value.behavior;
  const setBehavior = (patch: Partial<CharacterDraftPersona["behavior"]>) =>
    set({ behavior: { ...b, ...patch } });
  return (
    <div className="space-y-5">
      <RowsEditor
        label="상황별 반응"
        hint="어떤 상황에서 어떻게 반응하는가 — 이게 캐릭터를 살아있게 만든다."
        fields={[
          { key: "trigger", label: "상황", placeholder: "예: 사용자가 고민을 털어놓을 때" },
          { key: "action", label: "반응", placeholder: "예: 경청하고 공감 먼저, 조언은 나중에" },
        ]}
        value={b.situations}
        onChange={(situations) => setBehavior({ situations })}
      />
      <RowsEditor
        label="주제별 태도"
        hint="특정 주제가 나왔을 때의 태도 — 대화를 피할 주제도 여기에 담는다. 예: 정치 → 회피하려 함."
        fields={[
          { key: "name", label: "주제", placeholder: "예: 정치" },
          { key: "stance", label: "태도", placeholder: "예: 대화를 회피하려 함" },
        ]}
        value={b.topics}
        onChange={(topics) => setBehavior({ topics })}
      />
      <ArrayInput
        label="절대 규칙"
        values={b.rules}
        onChange={(rules) => setBehavior({ rules })}
      />
      <p className="text-xs text-muted-foreground -mt-3">
        힌트: 절대 규칙에는 "답하지 않을 질문"도 넣는다 — 예: "작년 휴방 이유를 먼저 꺼내지 않는다".
      </p>
    </div>
  );
}

/** 7. 감정 트리거 */
export function EmotionTriggersStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  return (
    <RowsEditor
      label="감정 트리거"
      hint="특정 단어·주제가 나왔을 때 유발되는 감정과 강도."
      fields={[
        { key: "keyword", label: "키워드/주제", placeholder: "예: 악플" },
        { key: "emotion", label: "유발 감정", placeholder: "예: 분노" },
        { key: "intensity", label: "강도", type: "range", min: 0, max: 1, step: 0.1, default: 0.5 },
      ]}
      value={value.emotion_triggers}
      onChange={(emotion_triggers) => onChange({ ...value, emotion_triggers })}
    />
  );
}

/** 8. 관계 설정 */
export function RelationshipsStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  // "사용자" 행은 전용 입력으로 관리한다 — 챗봇 캐릭터에서 가장 중요한 축이다.
  const userRow = value.relationships.find((r) => r.target === "사용자") ?? {
    target: "사용자",
    type: "",
    description: "",
  };
  const others = value.relationships.filter((r) => r.target !== "사용자");

  const setUserRow = (patch: Partial<{ type: string; description: string }>) => {
    const next = { ...userRow, ...patch };
    const rows = [...(next.type || next.description ? [next] : []), ...others];
    onChange({ ...value, relationships: rows });
  };

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-input p-3 space-y-3">
        <h4 className="text-sm font-medium">사용자와의 관계 (가장 중요)</h4>
        <Field label="어떻게 대하는가" hint="호칭·거리감·태도 — 첫 대화의 톤을 정한다">
          <Input
            value={userRow.type}
            onChange={(e) => setUserRow({ type: e.target.value })}
            placeholder="예: 시청자처럼 반말로 편하게, 처음 보는 동료처럼"
          />
        </Field>
        <Field label="이 대화에서 원하는 것" hint="캐릭터가 사용자에게서 얻고 싶은 것">
          <Input
            value={userRow.description}
            onChange={(e) => setUserRow({ description: e.target.value })}
            placeholder="예: 사용자의 하루를 듣고 싶어함. 누군가 말을 걸어주는 게 그리움"
          />
        </Field>
      </div>
      <RowsEditor
        label="그 외 주요 인물"
        hint="사용자 외 가족·동료·라이벌 등."
        fields={[
          { key: "target", label: "대상", placeholder: "예: 엄마, 재현" },
          { key: "type", label: "관계 유형", placeholder: "예: 모녀, 동료" },
          { key: "description", label: "관계 설명" },
        ]}
        value={others}
        onChange={(others) =>
          onChange({
            ...value,
            relationships: [...(userRow.type || userRow.description ? [userRow] : []), ...others],
          })
        }
      />
    </div>
  );
}

/** 9. 내면 상태 */
export function InnerWorldStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  const set = (patch: Partial<CharacterDraftPersona["inner_world"]>) =>
    onChange({ ...value, inner_world: { ...value.inner_world, ...patch } });
  return (
    <div className="space-y-4">
      <Field label="지금 하고 있는 생각" hint="혼자 있을 때 무슨 생각을 하는가">
        <Textarea rows={3} value={value.inner_world.current_thought} onChange={(e) => set({ current_thought: e.target.value })} placeholder="예: 이번 주 동접이 계속 떨어지는데..." />
      </Field>
      <Field label="숨기는 감정">
        <Textarea rows={3} value={value.inner_world.hidden_feelings} onChange={(e) => set({ hidden_feelings: e.target.value })} placeholder="예: 사람들이 좋아하는 건 진짜 내가 아니라는 생각." />
      </Field>
      <Field label="하고 싶지만 못하는 말">
        <Textarea rows={3} value={value.inner_world.wants_to_say} onChange={(e) => set({ wants_to_say: e.target.value })} placeholder="예: 가끔은 텐션 안 올리고도 방송하고 싶어." />
      </Field>
      <ArrayInput
        label="남들이 모르는 비밀·약점"
        values={value.secrets}
        onChange={(secrets) => onChange({ ...value, secrets })}
      />
    </div>
  );
}

/** 10. 내장 대화 예시 */
const SCENARIO_OPTIONS = ["인사", "위로", "갈등", "유머", "일상"];

export function PersonaExamplesStep({
  value,
  onChange,
}: {
  value: CharacterDraftPersona;
  onChange: (v: CharacterDraftPersona) => void;
}) {
  return (
    <RowsEditor
      label="대화 예시"
      hint="핵심 성격을 보여주는 예시 대화 — 인사·위로·갈등 상황을 권장한다."
      fields={[
        { key: "user", label: "사용자", placeholder: "예: 안녕!" },
        { key: "character", label: "캐릭터 응답", placeholder: "예: 왔어? 앉아 앉아ㅋㅋ" },
        { key: "scenario", label: "상황", type: "select", options: SCENARIO_OPTIONS },
      ]}
      value={value.examples}
      onChange={(examples) => onChange({ ...value, examples })}
    />
  );
}

/** 11. 세계관 */
export function WorldStep({
  value,
  persona,
  onPersonaChange,
  onChange,
}: {
  value: CharacterDraftKnowledge;
  persona: CharacterDraftPersona;
  onPersonaChange: (v: CharacterDraftPersona) => void;
  onChange: (v: CharacterDraftKnowledge) => void;
}) {
  const set = (patch: Partial<CharacterDraftKnowledge["world"]>) =>
    onChange({ ...value, world: { ...value.world, ...patch } });
  const w = value.world;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Field label="세계 이름">
          <Input value={w.name} onChange={(e) => set({ name: e.target.value })} placeholder="예: 개인방송 플랫폼 생태계" />
        </Field>
        <Field label="시대/배경">
          <Input value={w.era} onChange={(e) => set({ era: e.target.value })} placeholder="예: 2020년대 후반 대한민국 서울" />
        </Field>
        <Field label="기술 수준">
          <Input value={w.technology_level} onChange={(e) => set({ technology_level: e.target.value })} placeholder="예: 현대, 전근대" />
        </Field>
        <Field label="사회 구조">
          <Input value={w.social_structure} onChange={(e) => set({ social_structure: e.target.value })} placeholder="예: 플랫폼 – 소속사 – 스트리머" />
        </Field>
      </div>
      <Field label="세계관 설명" hint="이 세계에서 캐릭터가 사는 방식">
        <Textarea rows={4} value={w.description} onChange={(e) => set({ description: e.target.value })} placeholder="예: 개인이 실시간 방송을 송출하고, 시청자가 후원과 구독으로 수익을 만드는 시장이다." />
      </Field>
      <ArrayInput
        label="세계의 규칙·법칙"
        values={w.rules}
        onChange={(rules) => set({ rules })}
      />
      <Field
        label="메타 인식"
        hint="캐릭터가 자신이 가상 인물임을 아는가? 현실 세계(2026년, 인터넷 등)에 대해 아는가? — 모르는 게 기본값이면 비워둬도 된다."
      >
        <Textarea
          rows={3}
          value={persona.meta_awareness}
          onChange={(e) => onPersonaChange({ ...persona, meta_awareness: e.target.value })}
          placeholder="예: 자신이 AI인 줄 모른다. '인터넷' 같은 현대 개념은 들어도 이해하지 못한다."
        />
      </Field>
    </div>
  );
}

/** 12. 장소 */
export function LocationsStep({
  value,
  onChange,
}: {
  value: CharacterDraftKnowledge;
  onChange: (v: CharacterDraftKnowledge) => void;
}) {
  return (
    <RowsEditor
      label="중요한 장소"
      hint="스토리의 배경이 되는 공간과 그 의미."
      fields={[
        { key: "name", label: "장소 이름", placeholder: "예: 자취방 방송실" },
        { key: "description", label: "묘사", placeholder: "예: 6평 원룸의 절반을 차지한 방송 세팅" },
        { key: "significance", label: "의미", placeholder: "예: 하루의 대부분을 보내는 곳" },
        { key: "characters_present", label: "이곳에 있는 인물", type: "tags", placeholder: "예: 한소민, 재현" },
      ]}
      value={value.locations}
      onChange={(locations) => onChange({ ...value, locations })}
    />
  );
}

/** 13. 관계 그래프 */
export function GraphStep({
  value,
  onChange,
}: {
  value: CharacterDraftKnowledge;
  onChange: (v: CharacterDraftKnowledge) => void;
}) {
  return (
    <RowsEditor
      label="캐릭터 간 관계"
      hint="페르소나의 관계보다 넓게 — 세계 안 인물들 사이의 관계."
      fields={[
        { key: "from", label: "출발", placeholder: "예: 홍길동" },
        { key: "to", label: "대상", placeholder: "예: 아버지" },
        { key: "type", label: "관계 유형", placeholder: "예: 부자, 연인, 적" },
        { key: "sentiment", label: "감정", placeholder: "예: 분노, 그리움" },
        { key: "description", label: "설명" },
        { key: "strength", label: "강도", type: "range", min: 0, max: 1, step: 0.1 },
      ]}
      value={value.relationships}
      onChange={(relationships) => onChange({ ...value, relationships })}
    />
  );
}

/** 14. 타임라인 */
export function TimelineStep({
  value,
  onChange,
}: {
  value: CharacterDraftKnowledge;
  onChange: (v: CharacterDraftKnowledge) => void;
}) {
  return (
    <RowsEditor
      label="인생의 주요 사건"
      hint="시간순으로 — 어린 시절부터 현재까지."
      fields={[
        { key: "time", label: "시점", placeholder: "예: 20세" },
        { key: "event", label: "사건", placeholder: "예: 첫 방송 송출, 동접 3명" },
        { key: "characters_involved", label: "관련 인물", type: "tags", placeholder: "예: 한소민, 재현" },
        { key: "impact", label: "영향", placeholder: "예: 방송의 시작" },
      ]}
      value={value.timeline}
      onChange={(timeline) => onChange({ ...value, timeline })}
    />
  );
}

/** 15. 추가 지식 */
export function FreeformStep({
  value,
  onChange,
}: {
  value: CharacterDraftKnowledge;
  onChange: (v: CharacterDraftKnowledge) => void;
}) {
  return (
    <Field
      label="추가 지식"
      hint="위 분류에 들어가지 않는 중요 정보 — 스케줄, 문화, 배경 설정 등. knowledge/notes.md로 저장된다."
    >
      <Textarea
        rows={12}
        value={value.freeform}
        onChange={(e) => onChange({ ...value, freeform: e.target.value })}
        placeholder="예: 주 5회 방송. 시청자를 '얘들아'라고 부른다. 도네이션은 후반으로 미룬다."
      />
    </Field>
  );
}

/** 마지막 단계 위에 붙는 작성 요약 — 뭘 채웠고 뭐가 비었는지 한눈에. */
export function SummaryPanel({ draft }: { draft: CharacterDraft }) {
  const p = draft.persona;
  const k = draft.knowledge;

  const rows: { label: string; done: boolean; detail: string }[] = [
    {
      label: "기본 정보",
      done: Boolean(p.name.trim() && p.identity.trim()),
      detail: p.name ? `${p.name} · ${p.identity || "소개 없음"}` : "이름 없음",
    },
    {
      label: "첫 메시지",
      done: Boolean(p.first_message.trim()),
      detail: p.first_message.trim() ? "작성됨" : "미작성",
    },
    {
      label: "성격",
      done: p.personality.traits.length > 0,
      detail: `특성 ${p.personality.traits.length}개`,
    },
    {
      label: "말투",
      done: Boolean(p.speaking_style.summary.trim()),
      detail: p.speaking_style.summary.trim() ? "요약 있음" : "요약 없음",
    },
    {
      label: "배경",
      done: Boolean(p.backstory.trim()),
      detail: p.backstory.trim() ? "작성됨" : "미작성",
    },
    {
      label: "행동 지침",
      done: p.behavior.situations.length > 0 || p.behavior.rules.length > 0,
      detail: `상황 ${p.behavior.situations.length} · 주제 ${p.behavior.topics.length} · 규칙 ${p.behavior.rules.length}`,
    },
    {
      label: "감정 트리거",
      done: p.emotion_triggers.length > 0,
      detail: `${p.emotion_triggers.length}개`,
    },
    {
      label: "관계",
      done: p.relationships.length > 0,
      detail: `${p.relationships.length}개 (사용자 포함)`,
    },
    {
      label: "내면 상태",
      done: Boolean(p.inner_world.current_thought.trim()),
      detail: [
        p.inner_world.current_thought.trim() && "생각",
        p.inner_world.hidden_feelings.trim() && "숨김",
        p.inner_world.wants_to_say.trim() && "못한 말",
        p.secrets.length > 0 && `비밀 ${p.secrets.length}`,
      ]
        .filter(Boolean)
        .join(" · ") || "미작성",
    },
    {
      label: "세계관",
      done: Boolean(k.world.name.trim()),
      detail: k.world.name.trim() ? `${k.world.name}` : "미작성",
    },
    {
      label: "장소·관계·타임라인",
      done: k.locations.length > 0 || k.relationships.length > 0 || k.timeline.length > 0,
      detail: `장소 ${k.locations.length} · 관계 ${k.relationships.length} · 사건 ${k.timeline.length}`,
    },
    {
      label: "시나리오 예시",
      done: Object.values(draft.examples).some((g) => g.examples.length > 0),
      detail: Object.entries(draft.examples)
        .map(([, g]) => `${g.tag} ${g.examples.length}개`)
        .join(" · "),
    },
  ];

  const filled = rows.filter((r) => r.done).length;
  const noScenarios = rows[rows.length - 1].detail === "0개 · 0개 · 0개 · 0개 · 0개";

  return (
    <div className="mb-4 rounded-lg border border-input p-3 space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold">작성 요약</h4>
        <span className="text-xs text-muted-foreground">{filled} / {rows.length} 섹션</span>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-baseline justify-between gap-2 text-xs">
            <span className={r.done ? "text-foreground" : "text-muted-foreground"}>
              {r.done ? "✓" : "○"} {r.label}
            </span>
            <span className="text-muted-foreground truncate">{r.detail}</span>
          </div>
        ))}
      </div>
      {noScenarios && (
        <p className="text-xs text-amber-600 dark:text-amber-400">
          시나리오 예시가 하나도 없습니다 — 캐릭터의 말투 일관성이 떨어질 수 있어요.
        </p>
      )}
    </div>
  );
}

/** 16. 시나리오별 예시 */
const SCENARIO_GROUPS: { key: keyof CharacterDraftExamples; label: string }[] = [
  { key: "greeting", label: "인사 (greeting.yaml)" },
  { key: "comfort", label: "위로 (comfort.yaml)" },
  { key: "conflict", label: "갈등 (conflict.yaml)" },
  { key: "humor", label: "유머 (humor.yaml)" },
  { key: "daily", label: "일상 (daily.yaml)" },
];

export function ScenarioStep({
  value,
  onChange,
}: {
  value: CharacterDraftExamples;
  onChange: (v: CharacterDraftExamples) => void;
}) {
  const [open, setOpen] = useState<string>("greeting");
  const setGroup = (key: keyof CharacterDraftExamples, group: ScenarioGroup) =>
    onChange({ ...value, [key]: group });

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        시나리오별 대화 예시 — "이런 말에 이렇게 대답한다"는 패턴을 LLM에 학습시킨다.
        최소 한 시나리오는 채우길 권장한다.
      </p>
      {SCENARIO_GROUPS.map(({ key, label }) => {
        const group = value[key];
        const count = group.examples.filter((ex) => ex.user.trim() || ex.character.trim()).length;
        const isOpen = open === key;
        return (
          <div key={key} className="rounded-lg border border-input">
            <button
              type="button"
              className="w-full flex items-center justify-between px-3 py-2 text-sm font-medium hover:bg-accent/50 transition-colors"
              onClick={() => setOpen(isOpen ? "" : key)}
            >
              <span>{label}</span>
              <span className="text-xs text-muted-foreground">
                {count}개 {isOpen ? "▲" : "▼"}
              </span>
            </button>
            {isOpen && (
              <div className="p-3 border-t border-input space-y-3">
                <ArrayInput
                  label="트리거 키워드 (선택)"
                  values={group.keywords}
                  onChange={(keywords) => setGroup(key, { ...group, keywords })}
                />
                <RowsEditor
                  label="대화 예시"
                  fields={[
                    { key: "user", label: "사용자", placeholder: "예: 오늘 시험 망했어..." },
                    { key: "character", label: "캐릭터 응답", placeholder: "예: 아... 속상하겠다. 편히 말해 보게." },
                    { key: "emotion_state", label: "감정 상태", type: "tags", placeholder: "예: 슬픔, 실망" },
                  ]}
                  value={group.examples}
                  onChange={(examples) => setGroup(key, { ...group, examples })}
                />
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
