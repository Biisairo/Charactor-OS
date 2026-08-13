// 질문지 응답(CharacterDraft)의 빈 값과 서버 응답 정규화.
//
// 위저지(CharacterWizard)와 훅(useServerState)이 함께 쓴다 — 드래프트의
// 기본 형태는 한 곳에서만 정의해야 어긋나지 않는다.

import type {
  CharacterDraft,
  CharacterDraftResponse,
} from "@/types";

export function createEmptyDraft(): CharacterDraft {
  return {
    persona: {
      name: "",
      identity: "",
      age: "",
      gender: "",
      occupation: "",
      first_message: "",
      personality: {
        traits: [],
        big5: {
          openness: 0.5,
          conscientiousness: 0.5,
          extraversion: 0.5,
          agreeableness: 0.5,
          neuroticism: 0.5,
        },
      },
      speaking_style: {
        summary: "",
        tone: "",
        vocabulary: "",
        sentence_pattern: "",
        fillers: [],
        emojis: "",
        endings: [],
        sample: "",
      },
      values: [],
      backstory: "",
      likes: [],
      dislikes: [],
      fears: [],
      goals: [],
      behavior: { situations: [], topics: [], rules: [] },
      emotion_triggers: [],
      relationships: [],
      inner_world: { current_thought: "", hidden_feelings: "", wants_to_say: "" },
      secrets: [],
      meta_awareness: "",
      examples: [],
    },
    knowledge: {
      world: {
        name: "",
        era: "",
        description: "",
        rules: [],
        technology_level: "",
        social_structure: "",
      },
      locations: [],
      relationships: [],
      timeline: [],
      freeform: "",
    },
    examples: {
      greeting: { tag: "인사", keywords: [], examples: [] },
      comfort: { tag: "위로", keywords: [], examples: [] },
      conflict: { tag: "갈등", keywords: [], examples: [] },
      humor: { tag: "유머", keywords: [], examples: [] },
      daily: { tag: "일상", keywords: [], examples: [] },
    },
  };
}

/** 서버가 돌려준 부분 응답을 빈 값으로 메워 온전한 드래프트로 만든다. */
export function normalizeDraft(raw: CharacterDraftResponse): CharacterDraft {
  const empty = createEmptyDraft();
  const p = raw.persona ?? {};
  const persona = {
    ...empty.persona,
    ...p,
    personality: { ...empty.persona.personality, ...(p.personality ?? {}) },
    speaking_style: { ...empty.persona.speaking_style, ...(p.speaking_style ?? {}) },
    behavior: { ...empty.persona.behavior, ...(p.behavior ?? {}) },
    inner_world: { ...empty.persona.inner_world, ...(p.inner_world ?? {}) },
  };

  const k = raw.knowledge ?? {};
  const knowledge = {
    ...empty.knowledge,
    world: { ...empty.knowledge.world, ...(k.world ?? {}) },
    locations: k.locations ?? [],
    relationships: k.relationships ?? [],
    timeline: k.timeline ?? [],
    freeform: k.freeform ?? "",
  };

  const examples = { ...empty.examples };
  for (const key of ["greeting", "comfort", "conflict", "humor", "daily"] as const) {
    const g = raw.examples?.[key];
    if (g) {
      examples[key] = {
        ...empty.examples[key],
        ...g,
        examples: (g.examples ?? []).map((ex) => ({
          user: ex.user ?? "",
          character: ex.character ?? "",
          emotion_state: ex.emotion_state ?? [],
        })),
      };
    }
  }

  return { persona, knowledge, examples };
}

/** 드래프트에 입력된 내용이 기준(기본값)과 다른가 — 닫기 확인·저장 여부 판정. */
export function isDraftDirty(draft: CharacterDraft, baseline?: CharacterDraft): boolean {
  return JSON.stringify(draft) !== JSON.stringify(baseline ?? createEmptyDraft());
}
