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

export interface PerformanceData {
  trace?: {
    stages?: { name: string; duration_ms: number }[];
    total_duration_ms?: number;
  };
  emotion_state?: EmotionState;
  memory_count?: number;
  history_count?: number;
}
