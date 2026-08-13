// 서버 API가 돌려주는 값의 형태. 화면 전용 상태는 각 컴포넌트가 갖는다.

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  debugLogs?: string[];
}

export interface EmotionState {
  [key: string]: number;
}

export interface PersonaData {
  name: string;
  identity?: string;
  age?: number | string;
  gender?: string;
  occupation?: string;
  first_message?: string;
  personality?: {
    traits?: string[];
    big5?: {
      openness: number;
      conscientiousness: number;
      extraversion: number;
      agreeableness: number;
      neuroticism: number;
    };
  };
  speaking_style?: {
    summary?: string;
    tone?: string;
    vocabulary?: string;
    sentence_pattern?: string;
    fillers?: string[];
    emojis?: string;
    endings?: string[];
    sample?: string;
  };
  values?: string[];
  backstory?: string;
  likes?: string[];
  dislikes?: string[];
  fears?: string[];
  goals?: string[];
  behavior?: {
    situations?: { trigger: string; action: string }[];
    topics?: { name: string; stance: string }[];
    rules?: string[];
  };
  emotion_triggers?: { keyword: string; emotion: string; intensity: number }[];
  relationships?: { target: string; type: string; description: string }[];
  inner_world?: {
    current_thought?: string;
    hidden_feelings?: string;
    wants_to_say?: string;
  };
  secrets?: string[];
  meta_awareness?: string;
  examples?: { user: string; character: string; scenario?: string }[];
}

export interface FewShotGroup {
  tag: string;
  count: number;
  examples: { user: string; character: string; emotion_state: string[] }[];
}

export interface MemoryEntry {
  id: string;
  content: string;
  weight: number;
  emotion_tags: Record<string, number>;
  access_count: number;
  created_at: number;
}

export interface KnowledgeEntry {
  name: string;
  size?: number;
  preview?: string;
  type?: string;
  data?: unknown;
  count?: number;
}

export interface CharacterInfo {
  id: string;
  name: string;
  identity: string;
}

// ─── 새 캐릭터 질문지 (CharacterWizard) ────────────────────────────────
// 위저지 응답 한 벌이 static/ 전체를 이룬다. 모든 필드는 편집 중 빈 값이
// 가능해야 하므로 문자열·배열 기본값을 갖는다 — 서버로 보낼 때 빈 섹션을 뺀다.

export interface Big5Scores {
  openness: number;
  conscientiousness: number;
  extraversion: number;
  agreeableness: number;
  neuroticism: number;
}

export interface SituationRule {
  trigger: string;
  action: string;
}

export interface TopicRule {
  name: string;
  stance: string;
}

export interface EmotionTrigger {
  keyword: string;
  emotion: string;
  intensity: number;
}

export interface PersonaRelationship {
  target: string;
  type: string;
  description: string;
}

export interface PersonaExample {
  user: string;
  character: string;
  scenario?: string;
}

export interface CharacterDraftPersona {
  name: string;
  identity: string;
  age: string;
  gender: string;
  occupation: string;
  first_message: string;
  personality: {
    traits: string[];
    big5: Big5Scores;
  };
  speaking_style: {
    summary: string;
    tone: string;
    vocabulary: string;
    sentence_pattern: string;
    fillers: string[];
    emojis: string;
    endings: string[];
    sample: string;
  };
  values: string[];
  backstory: string;
  likes: string[];
  dislikes: string[];
  fears: string[];
  goals: string[];
  behavior: {
    situations: SituationRule[];
    topics: TopicRule[];
    rules: string[];
  };
  emotion_triggers: EmotionTrigger[];
  relationships: PersonaRelationship[];
  inner_world: {
    current_thought: string;
    hidden_feelings: string;
    wants_to_say: string;
  };
  secrets: string[];
  meta_awareness: string;
  examples: PersonaExample[];
}

export interface WorldData {
  name: string;
  era: string;
  description: string;
  rules: string[];
  technology_level: string;
  social_structure: string;
}

export interface LocationData {
  name: string;
  description: string;
  significance: string;
  characters_present: string[];
}

export interface KnowledgeRelationship {
  from: string;
  to: string;
  type: string;
  sentiment: string;
  description: string;
  strength: number;
}

export interface TimelineEvent {
  time: string;
  event: string;
  characters_involved: string[];
  impact: string;
}

export interface CharacterDraftKnowledge {
  world: WorldData;
  locations: LocationData[];
  relationships: KnowledgeRelationship[];
  timeline: TimelineEvent[];
  freeform: string;
}

export interface ScenarioExample {
  user: string;
  character: string;
  emotion_state: string[];
}

export interface ScenarioGroup {
  tag: string;
  keywords: string[];
  examples: ScenarioExample[];
}

export interface CharacterDraftExamples {
  greeting: ScenarioGroup;
  comfort: ScenarioGroup;
  conflict: ScenarioGroup;
  humor: ScenarioGroup;
  daily: ScenarioGroup;
}

export interface CharacterDraft {
  persona: CharacterDraftPersona;
  knowledge: CharacterDraftKnowledge;
  examples: CharacterDraftExamples;
}

/** GET /api/characters/{id}/draft 응답 — 질문지 재오픈용 원본 형태. */
export interface CharacterDraftResponse {
  persona: Partial<CharacterDraftPersona>;
  knowledge: Partial<CharacterDraftKnowledge>;
  examples: Partial<Record<keyof CharacterDraftExamples, ScenarioGroup>>;
}

export interface LabelMetrics {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  duration_ms: number;
}

export interface TurnMetrics {
  calls: number;
  failed_calls: number;
  refused_calls: number;
  /** 성공했고 거부도 아닌데 usage가 없는 호출 수. */
  unknown_usage_calls: number;
  /** 참이면 토큰 합계는 하한이다 — 화면에서 반드시 그렇게 표시한다. */
  tokens_are_lower_bound: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  /** 단가표에 없는 모델이면 null. 0과 구별해야 한다. */
  cost_usd: number | null;
  model?: string;
  by_label: Record<string, LabelMetrics>;
}

export interface PerformanceData {
  trace?: {
    stages?: { name: string; duration_ms: number }[];
    total_duration_ms?: number;
    metrics?: TurnMetrics;
  };
  emotion_state?: EmotionState;
  memory_count?: number;
  history_count?: number;
}
