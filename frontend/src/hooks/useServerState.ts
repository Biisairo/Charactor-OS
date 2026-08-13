// 서버에서 오는 상태와 그것을 갱신하는 동작.
//
// 한 훅에 모은 이유가 있다. 이 값들은 독립적이지 않다 — 캐릭터를 전환하거나
// 페르소나를 저장하면 감정·기억·지식·로그가 **함께** 다시 읽혀야 한다.
// 도메인별로 쪼개면 그 동시성이 호출부로 새어 나가, 한 곳만 빠뜨려도
// 화면이 조용히 낡은 값을 보여준다.
//
// 화면 전용 상태(시트 열림, 탭 선택 등)는 여기 두지 않는다.

import { useCallback, useEffect, useState } from "react";

import type {
  CharacterDraft,
  CharacterDraftResponse,
  CharacterInfo,
  EmotionState,
  KnowledgeEntry,
  MemoryEntry,
  Message,
  PerformanceData,
  PersonaData,
} from "@/types";
import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";
import { normalizeDraft } from "@/lib/characterDraft";

interface Options {
  /** 서버에 저장된 대화 기록을 읽었을 때 호출된다. */
  onHistoryLoaded: (messages: Message[]) => void;
  /** 캐릭터 전환·초기화처럼 대화를 비워야 할 때 호출된다. */
  onConversationCleared: () => void;
}

export function useServerState({ onHistoryLoaded, onConversationCleared }: Options) {
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [memoryStats, setMemoryStats] = useState<{ count: number } | null>(null);
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [persona, setPersona] = useState<PersonaData | null>(null);
  const [knowledgeEntries, setKnowledgeEntries] = useState<KnowledgeEntry[]>([]);
  const [characters, setCharacters] = useState<CharacterInfo[]>([]);
  const [activeCharacter, setActiveCharacter] = useState("");
  const [logs, setLogs] = useState<string[]>([]);
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [debugEnabled, setDebugEnabled] = useState(true);
  const [switching, setSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshData = useCallback(async () => {
    try {
      const [e, m, p, k, mem, hist, charsResp, logResp] = await Promise.all([
        apiGet<EmotionState>("/api/emotion"),
        apiGet<{ count: number }>("/api/memory/stats"),
        apiGet<PersonaData>("/api/persona"),
        apiGet<{ entries: KnowledgeEntry[] }>("/api/knowledge"),
        apiGet<{ memories: MemoryEntry[] }>("/api/memory"),
        apiGet<{ turns: { role: string; content: string; timestamp: number }[] }>("/api/history"),
        apiGet<{ characters: CharacterInfo[]; active: string }>("/api/characters"),
        apiGet<{ logs: string[]; total: number }>("/api/logs?level=all&limit=200"),
      ]);
      setEmotion(e);
      setMemoryStats(m);
      setPersona(p);
      setKnowledgeEntries(k.entries);
      setMemories(mem.memories);
      setCharacters(charsResp.characters);
      setActiveCharacter(charsResp.active);
      setLogs(logResp.logs);
      if (hist.turns.length > 0) {
        onHistoryLoaded(
          hist.turns.map((turn, i) => ({
            id: `history-${i}`,
            role: turn.role === "user" ? "user" : "assistant",
            content: turn.content,
            timestamp: new Date(turn.timestamp * 1000),
          })),
        );
      }
    } catch (err) {
      setError(`데이터 로드 실패: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [onHistoryLoaded]);

  const refreshPerformance = useCallback(async () => {
    try {
      setPerformance(await apiGet<PerformanceData>("/api/performance"));
    } catch {
      // 성능 패널은 보조 정보다. 실패해도 대화를 막지 않는다.
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const data = await apiGet<{ logs: string[]; total: number }>(
        "/api/logs?level=all&limit=500",
      );
      setLogs(data.logs);
    } catch {
      // 로그 조회 실패가 화면을 막아서는 안 된다.
    }
  }, []);

  const fetchDebugLogs = useCallback(async () => {
    try {
      const data = await apiGet<{ logs: string[]; enabled: boolean }>("/api/debug");
      setDebugEnabled(data.enabled);
    } catch {
      // 상동
    }
  }, []);

  const toggleDebug = useCallback(async () => {
    try {
      const data = await apiPost<{ enabled: boolean }>("/api/debug/toggle", {});
      setDebugEnabled(data.enabled);
      if (data.enabled) await fetchDebugLogs();
    } catch {
      // 상동
    }
  }, [fetchDebugLogs]);

  useEffect(() => {
    refreshData();
    fetchDebugLogs();
    // 성능 패널은 대화 완료 시에만 갱신되었다. 새로고침 직후에는 직전 턴의
    // 트레이스가 서버에 남아 있는데도 화면이 "데이터 없음"이었다 (REQ-13-1).
    refreshPerformance();
  }, [refreshData, fetchDebugLogs, refreshPerformance]);

  // 디버그가 켜져 있을 때만 로그를 주기적으로 당겨온다.
  useEffect(() => {
    if (!debugEnabled) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [debugEnabled, fetchLogs]);

  const savePersona = useCallback(
    async (data: PersonaData) => {
      try {
        await apiPut("/api/persona", data);
        await refreshData();
      } catch {
        setError("페르소나 저장 실패");
      }
    },
    [refreshData],
  );

  const saveKnowledge = useCallback(
    async (name: string, content: string) => {
      try {
        await apiPut(`/api/knowledge/${name}`, { content });
        await refreshData();
      } catch {
        setError("지식 저장 실패");
      }
    },
    [refreshData],
  );

  const loadKnowledge = useCallback(async (name: string) => {
    const data = await apiGet<{ content: string }>(`/api/knowledge/${name}`);
    return data.content;
  }, []);

  const resetCharacter = useCallback(
    async (opts: { memory: boolean; emotion: boolean; history: boolean }) => {
      if (!confirm("정말 초기화하시겠습니까?")) return;
      try {
        await apiPost("/api/character/reset", opts);
        if (opts.history) onConversationCleared();
        await refreshData();
      } catch {
        setError("캐릭터 초기화 실패");
      }
    },
    [refreshData, onConversationCleared],
  );

  const switchCharacter = useCallback(
    async (characterId: string) => {
      if (characterId === activeCharacter) return;
      if (!confirm("캐릭터를 전환하면 현재 대화가 초기화됩니다. 계속하시겠습니까?")) return;

      setSwitching(true);
      try {
        await apiPost("/api/character/switch", { character_id: characterId });
        onConversationCleared();
        await refreshData();
      } catch {
        setError("캐릭터 전환 실패");
      } finally {
        setSwitching(false);
      }
    },
    [activeCharacter, refreshData, onConversationCleared],
  );

  const createCharacterFromDraft = useCallback(
    async (draft: CharacterDraft): Promise<boolean> => {
      const name = draft.persona.name.trim();
      if (!name) return false;
      try {
        const res = await apiPost<{ status: string; character: string }>("/api/characters", {
          name,
          identity: draft.persona.identity,
          static_data: buildStaticPayload(draft),
        });
        // 새 캐릭터로 바로 전환 — 대화를 비우는 건 생성 의도에 이미 포함돼 있으므로
        // 확인 없이 진행한다. 목록·상태는 refreshData가 다시 읽는다.
        await apiPost("/api/character/switch", { character_id: res.character });
        onConversationCleared();
        await refreshData();
        return true;
      } catch {
        setError("캐릭터 생성 실패");
        return false;
      }
    },
    [refreshData, onConversationCleared],
  );

  const loadCharacterDraft = useCallback(
    async (characterId: string): Promise<CharacterDraft | null> => {
      try {
        const raw = await apiGet<CharacterDraftResponse>(`/api/characters/${characterId}/draft`);
        return normalizeDraft(raw);
      } catch {
        setError("질문지 로드 실패");
        return null;
      }
    },
    [],
  );

  const updateCharacterStatic = useCallback(
    async (characterId: string, draft: CharacterDraft): Promise<boolean> => {
      if (!draft.persona.name.trim()) return false;
      try {
        await apiPut(`/api/characters/${characterId}/static`, buildStaticPayload(draft));
        await refreshData();
        return true;
      } catch {
        setError("캐릭터 저장 실패");
        return false;
      }
    },
    [refreshData],
  );

  const deleteCharacter = useCallback(
    async (characterId: string) => {
      if (!confirm("정말 삭제하시겠습니까?")) return;
      try {
        await apiDelete(`/api/characters/${characterId}`);
        await refreshData();
      } catch {
        setError("캐릭터 삭제 실패");
      }
    },
    [refreshData],
  );

  return {
    emotion,
    memoryStats,
    memories,
    persona,
    knowledgeEntries,
    characters,
    activeCharacter,
    logs,
    performance,
    debugEnabled,
    switching,
    error,
    setError,
    refreshData,
    refreshPerformance,
    fetchLogs,
    toggleDebug,
    savePersona,
    saveKnowledge,
    loadKnowledge,
    resetCharacter,
    switchCharacter,
    createCharacterFromDraft,
    loadCharacterDraft,
    updateCharacterStatic,
    deleteCharacter,
  };
}

// ─── 질문지 응답 → POST /api/characters 페이로드 ─────────────────────────
// 위저지의 빈 응답지를 서버에 그대로 보내면 YAML에 빈 껍데기가 남는다.
// 여기서 빈 문자열·빈 배열 항목을 걸러, 건너뛴 섹션은 아예 보내지 않는다.

function hasContent(v: unknown): boolean {
  if (Array.isArray(v)) return v.length > 0;
  return Boolean(v);
}

function cleanRows<T extends object>(rows: T[]): T[] {
  return rows.filter((r) => Object.values(r as Record<string, unknown>).some(hasContent));
}

function cleanStrings(items: string[]): string[] {
  return items.map((s) => s.trim()).filter(Boolean);
}

function buildStaticPayload(draft: CharacterDraft): Record<string, unknown> {
  const p = draft.persona;
  const persona: Record<string, unknown> = {
    ...p,
    values: cleanStrings(p.values),
    likes: cleanStrings(p.likes),
    dislikes: cleanStrings(p.dislikes),
    fears: cleanStrings(p.fears),
    goals: cleanStrings(p.goals),
    personality: {
      ...p.personality,
      traits: cleanStrings(p.personality.traits),
    },
    speaking_style: {
      ...p.speaking_style,
      fillers: cleanStrings(p.speaking_style.fillers),
      endings: cleanStrings(p.speaking_style.endings),
    },
    behavior: {
      situations: cleanRows(p.behavior.situations),
      topics: cleanRows(p.behavior.topics),
      rules: cleanStrings(p.behavior.rules),
    },
    emotion_triggers: cleanRows(p.emotion_triggers),
    relationships: cleanRows(p.relationships),
    examples: cleanRows(p.examples),
  };

  const k = draft.knowledge;
  const knowledge: Record<string, unknown> = {};
  if (Object.values(k.world).some(hasContent)) {
    knowledge.world = { ...k.world, rules: cleanStrings(k.world.rules) };
  }
  const locations = cleanRows(k.locations);
  if (locations.length) knowledge.locations = locations;
  const relationships = cleanRows(k.relationships);
  if (relationships.length) knowledge.relationships = relationships;
  const timeline = cleanRows(k.timeline);
  if (timeline.length) knowledge.timeline = timeline;
  if (k.freeform.trim()) knowledge.freeform = k.freeform;

  const examples: Record<string, unknown> = {};
  for (const key of ["greeting", "comfort", "conflict", "humor", "daily"] as const) {
    const group = draft.examples[key];
    const cleaned = group.examples
      .filter((ex) => ex.user.trim() || ex.character.trim())
      .map((ex) => ({ ...ex, emotion_state: cleanStrings(ex.emotion_state) }));
    if (cleaned.length) {
      examples[key] = {
        tag: group.tag,
        keywords: cleanStrings(group.keywords),
        examples: cleaned,
      };
    }
  }

  return { persona, knowledge, examples };
}
