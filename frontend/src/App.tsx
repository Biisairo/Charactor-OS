import { useState, useEffect, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  debugLogs?: string[];
}

interface EmotionState {
  [key: string]: number;
}

interface PersonaData {
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

interface FewShotGroup {
  tag: string;
  count: number;
  examples: { user: string; character: string; emotion_state: string[] }[];
}

interface MemoryEntry {
  id: string;
  content: string;
  weight: number;
  emotion_tags: Record<string, number>;
  access_count: number;
  created_at: number;
}

interface KnowledgeEntry {
  name: string;
  size?: number;
  preview?: string;
  type?: string;
  data?: unknown;
  count?: number;
}

interface CharacterInfo {
  id: string;
  name: string;
  identity: string;
}

interface PerformanceData {
  trace?: {
    stages?: { name: string; duration_ms: number }[];
    total_duration_ms?: number;
  };
  emotion_state?: EmotionState;
  memory_count?: number;
  history_count?: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STAGE_LABELS: Record<string, string> = {
  context: "컨텍스트 수집",
  response: "응답 생성",
  postprocess: "후처리",
};

const ERROR_KEYWORDS = ["실패", "오류", "error", "FAIL"];

const LOG_COLORS: { pattern: RegExp; cls: string }[] = [
  { pattern: /실패|오류|error|FAIL/i, cls: "text-red-400" },
  { pattern: /\[Orchestrator\]/, cls: "text-blue-400" },
  { pattern: /\[ReAct\]/, cls: "text-cyan-400" },
  { pattern: /\[Response\]/, cls: "text-yellow-400" },
  { pattern: /\[PostProcess\]/, cls: "text-green-400" },
  { pattern: /Stage [123]/, cls: "text-purple-400" },
  { pattern: /={4,}/, cls: "text-muted-foreground" },
  { pattern: /-{4,}/, cls: "text-muted-foreground" },
];

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

const API_BASE = import.meta.env.VITE_API_URL || "";

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function apiDelete<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Dark mode hook
// ---------------------------------------------------------------------------

function useDarkMode() {
  const [dark, setDark] = useState(() => {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return { dark, setDark };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

function ChatMessage({
  message,
  isStreaming,
}: {
  message: Message;
  isStreaming: boolean;
}) {
  const isUser = message.role === "user";
  const isLoading = isStreaming && message.id === "streaming" && !message.content;

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

function EmotionPanel({ emotion }: { emotion: EmotionState | null }) {
  if (!emotion || Object.keys(emotion).length === 0)
    return <p className="text-muted-foreground text-sm">감정 데이터 없음</p>;

  return (
    <div className="space-y-3">
      {Object.entries(emotion).map(([key, value]) => (
        <div key={key} className="space-y-1">
          <div className="flex justify-between text-sm">
            <span>{key}</span>
            <span className="text-muted-foreground">
              {(value * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-primary rounded-full transition-all duration-500"
              style={{ width: `${Math.min(value * 100, 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function MemoryPanel({
  stats,
  memories,
}: {
  stats: { count: number } | null;
  memories: MemoryEntry[];
}) {
  if (!stats)
    return <p className="text-muted-foreground text-sm">기억 데이터 없음</p>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">저장된 기억</span>
        <Badge variant="secondary">{stats.count}개</Badge>
      </div>
      {memories.length > 0 && (
        <div className="space-y-2 max-h-[300px] overflow-auto">
          {memories.map((m) => (
            <div key={m.id} className="p-2 bg-muted rounded text-xs space-y-1">
              <p className="text-foreground">{m.content}</p>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span>가중치: {m.weight}</span>
                <span>접근: {m.access_count}회</span>
              </div>
              {Object.keys(m.emotion_tags).length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {Object.entries(m.emotion_tags).map(([k, v]) => (
                    <Badge key={k} variant="outline" className="text-[10px] px-1 py-0">
                      {k}: {(v * 100).toFixed(0)}%
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LogViewer({
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

function PerformancePanel({
  performance,
  onRefresh,
}: {
  performance: PerformanceData | null;
  onRefresh: () => void;
}) {
  if (!performance) {
    return (
      <div className="space-y-2">
        <p className="text-muted-foreground text-sm">데이터 없음</p>
        <Button variant="outline" size="sm" className="w-full text-xs" onClick={onRefresh}>
          새로고침
        </Button>
      </div>
    );
  }

  const stages = performance.trace?.stages ?? [];
  const totalMs = performance.trace?.total_duration_ms;
  const maxDuration = stages.reduce((max, s) => Math.max(max, s.duration_ms), 0);

  return (
    <div className="space-y-3">
      {totalMs != null && (
        <div className="text-center">
          <div className="text-2xl font-bold tabular-nums">{totalMs.toFixed(0)}</div>
          <div className="text-xs text-muted-foreground">총 지연 시간 (ms)</div>
        </div>
      )}
      {stages.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs font-medium text-muted-foreground">단계별 분석</span>
          {stages.map((stage, i) => {
            const pct = maxDuration > 0 ? (stage.duration_ms / maxDuration) * 100 : 0;
            return (
              <div key={i} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span>{STAGE_LABELS[stage.name] || stage.name}</span>
                  <span className="text-muted-foreground tabular-nums">
                    {stage.duration_ms.toFixed(1)} ms
                  </span>
                </div>
                <div className="h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
      <div className="flex gap-4">
        {performance.memory_count != null && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">기억:</span>
            <Badge variant="outline" className="text-xs">{performance.memory_count}</Badge>
          </div>
        )}
        {performance.history_count != null && (
          <div className="flex items-center gap-1">
            <span className="text-xs text-muted-foreground">기록:</span>
            <Badge variant="outline" className="text-xs">{performance.history_count}</Badge>
          </div>
        )}
      </div>
      <Button variant="outline" size="sm" className="w-full text-xs" onClick={onRefresh}>
        새로고침
      </Button>
    </div>
  );
}

function ArrayInput({
  label,
  values,
  onChange,
}: {
  label: string;
  values: string[];
  onChange: (vals: string[]) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 px-2 text-xs"
          onClick={() => onChange([...values, ""])}
        >
          + 추가
        </Button>
      </div>
      <div className="space-y-1">
        {values.map((val, i) => (
          <div key={i} className="flex gap-1">
            <Input
              value={val}
              onChange={(e) => {
                const next = [...values];
                next[i] = e.target.value;
                onChange(next);
              }}
              className="flex-1 h-8 text-sm"
            />
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
              onClick={() => onChange(values.filter((_, j) => j !== i))}
            >
              ×
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}

function PersonaEditor({
  persona,
  onSave,
}: {
  persona: PersonaData | null;
  onSave: (data: PersonaData) => void;
}) {
  const [editData, setEditData] = useState<PersonaData>({
    name: "",
    personality: { traits: [] },
    speaking_style: { summary: "" },
    values: [],
    backstory: "",
    likes: [],
    dislikes: [],
    fears: [],
    goals: [],
    behavior: { situations: [], topics: [], rules: [] },
    emotion_triggers: [],
    relationships: [],
    inner_world: {},
    examples: [],
  });
  const [expandedSection, setExpandedSection] = useState<string>("basic");

  useEffect(() => {
    if (persona) setEditData(persona);
  }, [persona]);

  if (!persona)
    return <p className="text-muted-foreground text-sm">페르소나 로딩 중...</p>;

  const toggleSection = (s: string) =>
    setExpandedSection(expandedSection === s ? "" : s);

  const SectionHeader = ({ id, label }: { id: string; label: string }) => (
    <button
      className="w-full flex items-center justify-between py-2 text-sm font-semibold border-b"
      onClick={() => toggleSection(id)}
    >
      {label}
      <span>{expandedSection === id ? "▲" : "▼"}</span>
    </button>
  );

  return (
    <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-2">
      {/* 기본 정보 */}
      <SectionHeader id="basic" label="기본 정보" />
      {expandedSection === "basic" && (
        <div className="space-y-2 pl-2">
          <div className="space-y-1">
            <label className="text-xs font-medium">이름</label>
            <Input
              value={editData.name}
              onChange={(e) => setEditData({ ...editData, name: e.target.value })}
            />
          </div>
          <div className="space-y-1">
            <label className="text-xs font-medium">한줄 소개</label>
            <Input
              value={editData.identity || ""}
              onChange={(e) => setEditData({ ...editData, identity: e.target.value })}
              placeholder="예: 아버지를 아버지라 부르지 못하는 서자"
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <div className="space-y-1">
              <label className="text-xs font-medium">나이</label>
              <Input
                value={editData.age?.toString() || ""}
                onChange={(e) => setEditData({ ...editData, age: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">성별</label>
              <Input
                value={editData.gender || ""}
                onChange={(e) => setEditData({ ...editData, gender: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs font-medium">직업</label>
              <Input
                value={editData.occupation || ""}
                onChange={(e) => setEditData({ ...editData, occupation: e.target.value })}
              />
            </div>
          </div>
        </div>
      )}

      {/* 성격 */}
      <SectionHeader id="personality" label="성격" />
      {expandedSection === "personality" && (
        <div className="space-y-2 pl-2">
          <ArrayInput
            label="성격 특성"
            values={editData.personality?.traits || []}
            onChange={(v) =>
              setEditData({
                ...editData,
                personality: { ...editData.personality, traits: v },
              })
            }
          />
          {editData.personality?.big5 && (
            <div className="space-y-1">
              <label className="text-xs font-medium">Big Five 성격 모델</label>
              {(["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"] as const).map(
                (key) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="text-xs w-24">{key}</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={editData.personality?.big5?.[key] ?? 0.5}
                      onChange={(e) =>
                        setEditData({
                          ...editData,
                          personality: {
                            ...editData.personality,
                            big5: {
                              ...(editData.personality?.big5 ?? {
                                openness: 0.5, conscientiousness: 0.5,
                                extraversion: 0.5, agreeableness: 0.5, neuroticism: 0.5,
                              }),
                              [key]: parseFloat(e.target.value),
                            },
                          },
                        })
                      }
                      className="flex-1"
                    />
                    <span className="text-xs w-8 text-right">
                      {(editData.personality?.big5?.[key] ?? 0.5).toFixed(1)}
                    </span>
                  </div>
                )
              )}
            </div>
          )}
        </div>
      )}

      {/* 말투 */}
      <SectionHeader id="speaking" label="말투" />
      {expandedSection === "speaking" && (
        <div className="space-y-2 pl-2">
          <div className="space-y-1">
            <label className="text-xs font-medium">말투 요약</label>
            <textarea
              className="w-full min-h-[60px] rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={editData.speaking_style?.summary || ""}
              onChange={(e) =>
                setEditData({
                  ...editData,
                  speaking_style: { ...editData.speaking_style, summary: e.target.value },
                })
              }
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            {(["tone", "vocabulary", "sentence_pattern", "emojis"] as const).map((key) => (
              <div key={key} className="space-y-1">
                <label className="text-xs font-medium">{key}</label>
                <Input
                  value={editData.speaking_style?.[key] || ""}
                  onChange={(e) =>
                    setEditData({
                      ...editData,
                      speaking_style: { ...editData.speaking_style, [key]: e.target.value },
                    })
                  }
                />
              </div>
            ))}
          </div>
          <ArrayInput
            label="말버릇"
            values={editData.speaking_style?.fillers || []}
            onChange={(v) =>
              setEditData({
                ...editData,
                speaking_style: { ...editData.speaking_style, fillers: v },
              })
            }
          />
          <ArrayInput
            label="문미 패턴"
            values={editData.speaking_style?.endings || []}
            onChange={(v) =>
              setEditData({
                ...editData,
                speaking_style: { ...editData.speaking_style, endings: v },
              })
            }
          />
        </div>
      )}

      {/* 배경 */}
      <SectionHeader id="background" label="배경" />
      {expandedSection === "background" && (
        <div className="space-y-2 pl-2">
          <ArrayInput label="가치관" values={editData.values || []} onChange={(v) => setEditData({ ...editData, values: v })} />
          <div className="space-y-1">
            <label className="text-xs font-medium">배경 이야기</label>
            <textarea
              className="w-full min-h-[100px] rounded-md border border-input bg-background px-3 py-2 text-sm"
              value={editData.backstory || ""}
              onChange={(e) => setEditData({ ...editData, backstory: e.target.value })}
            />
          </div>
          <ArrayInput label="좋아하는 것" values={editData.likes || []} onChange={(v) => setEditData({ ...editData, likes: v })} />
          <ArrayInput label="싫어하는 것" values={editData.dislikes || []} onChange={(v) => setEditData({ ...editData, dislikes: v })} />
          <ArrayInput label="두려운 것" values={editData.fears || []} onChange={(v) => setEditData({ ...editData, fears: v })} />
          <ArrayInput label="목표" values={editData.goals || []} onChange={(v) => setEditData({ ...editData, goals: v })} />
        </div>
      )}

      {/* 행동 지침 */}
      <SectionHeader id="behavior" label="행동 지침" />
      {expandedSection === "behavior" && (
        <div className="space-y-2 pl-2">
          <ArrayInput
            label="절대 규칙"
            values={editData.behavior?.rules || []}
            onChange={(v) =>
              setEditData({
                ...editData,
                behavior: { ...editData.behavior, rules: v },
              })
            }
          />
          <div className="space-y-1">
            <label className="text-xs font-medium text-muted-foreground">
              상황별/주제별 행동 지침은 YAML 파일에서 직접 편집하세요
            </label>
          </div>
        </div>
      )}

      {/* 감정 트리거 */}
      <SectionHeader id="triggers" label="감정 트리거" />
      {expandedSection === "triggers" && (
        <div className="space-y-2 pl-2">
          {(editData.emotion_triggers || []).map((t, i) => (
            <div key={i} className="flex items-center gap-2 text-sm">
              <Input
                className="flex-1"
                value={t.keyword}
                onChange={(e) => {
                  const triggers = [...(editData.emotion_triggers || [])];
                  triggers[i] = { ...t, keyword: e.target.value };
                  setEditData({ ...editData, emotion_triggers: triggers });
                }}
                placeholder="키워드"
              />
              <Input
                className="w-20"
                value={t.emotion}
                onChange={(e) => {
                  const triggers = [...(editData.emotion_triggers || [])];
                  triggers[i] = { ...t, emotion: e.target.value };
                  setEditData({ ...editData, emotion_triggers: triggers });
                }}
                placeholder="감정"
              />
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={t.intensity}
                onChange={(e) => {
                  const triggers = [...(editData.emotion_triggers || [])];
                  triggers[i] = { ...t, intensity: parseFloat(e.target.value) };
                  setEditData({ ...editData, emotion_triggers: triggers });
                }}
                className="w-16"
              />
              <button
                className="text-destructive text-xs"
                onClick={() => {
                  const triggers = [...(editData.emotion_triggers || [])];
                  triggers.splice(i, 1);
                  setEditData({ ...editData, emotion_triggers: triggers });
                }}
              >
                삭제
              </button>
            </div>
          ))}
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setEditData({
                ...editData,
                emotion_triggers: [
                  ...(editData.emotion_triggers || []),
                  { keyword: "", emotion: "", intensity: 0.5 },
                ],
              })
            }
          >
            + 트리거 추가
          </Button>
        </div>
      )}

      {/* 내면 상태 */}
      <SectionHeader id="inner" label="내면 상태" />
      {expandedSection === "inner" && (
        <div className="space-y-2 pl-2">
          {(["current_thought", "hidden_feelings", "wants_to_say"] as const).map((key) => (
            <div key={key} className="space-y-1">
              <label className="text-xs font-medium">
                {key === "current_thought" ? "현재 생각" : key === "hidden_feelings" ? "숨기는 감정" : "하고 싶은 말"}
              </label>
              <textarea
                className="w-full min-h-[60px] rounded-md border border-input bg-background px-3 py-2 text-sm"
                value={editData.inner_world?.[key] || ""}
                onChange={(e) =>
                  setEditData({
                    ...editData,
                    inner_world: { ...editData.inner_world, [key]: e.target.value },
                  })
                }
              />
            </div>
          ))}
        </div>
      )}

      <Button onClick={() => onSave(editData)} className="w-full">
        저장
      </Button>
    </div>
  );
}

function KnowledgeEditor({
  entries,
  onLoad,
  onSave,
}: {
  entries: KnowledgeEntry[];
  onLoad: (name: string) => Promise<string>;
  onSave: (name: string, content: string) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSelect = async (name: string) => {
    setSelected(name);
    setLoading(true);
    try {
      const data = await onLoad(name);
      setContent(data);
    } catch {
      setContent("(로드 실패)");
    }
    setLoading(false);
  };

  const typeLabels: Record<string, string> = {
    world: "🌍 세계관",
    character: "👤 캐릭터",
    relationships: "🔗 관계",
    timeline: "📅 타임라인",
    locations: "📍 장소",
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        {entries.map((entry) => (
          <div
            key={entry.name}
            className={`p-2 rounded cursor-pointer text-sm ${
              selected === entry.name
                ? "bg-primary text-primary-foreground"
                : "bg-muted hover:bg-muted/80"
            }`}
            onClick={() => handleSelect(entry.name)}
          >
            <div className="font-medium">
              {entry.type ? typeLabels[entry.type] || entry.type : ""}
              {" "}
              {entry.name}
            </div>
            <div className="text-xs opacity-70">
              {entry.count !== undefined
                ? `${entry.count}개`
                : entry.size !== undefined
                  ? `${entry.size}자`
                  : ""}
            </div>
          </div>
        ))}
        {entries.length === 0 && (
          <p className="text-muted-foreground text-sm">지식 파일 없음</p>
        )}
      </div>

      {selected && (
        <div className="space-y-2">
          <label className="text-sm font-medium">{selected}</label>
          {loading ? (
            <p className="text-sm text-muted-foreground">로딩 중...</p>
          ) : (
            <>
              <textarea
                className="w-full min-h-[200px] rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                value={content}
                onChange={(e) => setContent(e.target.value)}
              />
              <Button
                onClick={() => onSave(selected, content)}
                className="w-full"
              >
                저장
              </Button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function FewShotPanel() {
  const [groups, setGroups] = useState<FewShotGroup[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiGet<{ groups: FewShotGroup[] }>("/api/fewshot")
      .then((data) => setGroups(data.groups))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p className="text-sm text-muted-foreground">로딩 중...</p>;

  return (
    <div className="space-y-3 max-h-[50vh] overflow-y-auto">
      {groups.length === 0 && (
        <p className="text-sm text-muted-foreground">예시 없음</p>
      )}
      {groups.map((group) => (
        <div key={group.tag} className="border rounded-md p-3 space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant="secondary">{group.tag}</Badge>
            <span className="text-xs text-muted-foreground">{group.count}개</span>
          </div>
          {group.examples.map((ex, i) => (
            <div key={i} className="pl-3 border-l-2 border-muted text-sm space-y-1">
              <div>
                <span className="text-muted-foreground">사용자: </span>
                {ex.user}
              </div>
              <div>
                <span className="text-muted-foreground">캐릭터: </span>
                {ex.character}
              </div>
              {ex.emotion_state.length > 0 && (
                <div className="flex gap-1">
                  {ex.emotion_state.map((e) => (
                    <Badge key={e} variant="outline" className="text-xs">
                      {e}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function ResetPanel({
  onReset,
}: {
  onReset: (opts: { memory: boolean; emotion: boolean; history: boolean }) => void;
}) {
  const [memory, setMemory] = useState(true);
  const [emotion, setEmotion] = useState(true);
  const [history, setHistory] = useState(false);

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        캐릭터의 상태를 초기화합니다. 이 작업은 되돌릴 수 없습니다.
      </p>
      <div className="space-y-2">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={memory}
            onChange={(e) => setMemory(e.target.checked)}
          />
          기억 초기화
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={emotion}
            onChange={(e) => setEmotion(e.target.checked)}
          />
          감정 초기화
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={history}
            onChange={(e) => setHistory(e.target.checked)}
          />
          대화 기록 초기화
        </label>
      </div>
      <Button
        variant="destructive"
        onClick={() => onReset({ memory, emotion, history })}
        className="w-full"
      >
        초기화
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------

export default function App() {
  const { dark, setDark } = useDarkMode();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [emotion, setEmotion] = useState<EmotionState | null>(null);
  const [memoryStats, setMemoryStats] = useState<{ count: number } | null>(null);
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [persona, setPersona] = useState<PersonaData | null>(null);
  const [knowledgeEntries, setKnowledgeEntries] = useState<KnowledgeEntry[]>([]);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [settingsTab, setSettingsTab] = useState<"emotion" | "persona" | "knowledge" | "fewshot" | "reset">("emotion");
  const [debugEnabled, setDebugEnabled] = useState(true);
  const [debugOpen, setDebugOpen] = useState(true);
  const [characters, setCharacters] = useState<CharacterInfo[]>([]);
  const [activeCharacter, setActiveCharacter] = useState("");
  const [switching, setSwitching] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [logModalOpen, setLogModalOpen] = useState(false);
  const [performance, setPerformance] = useState<PerformanceData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showNewCharacter, setShowNewCharacter] = useState(false);
  const [showCharManager, setShowCharManager] = useState(false);
  const [newName, setNewName] = useState("");
  const [newIdentity, setNewIdentity] = useState("");

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const logScrollRef = useRef<HTMLDivElement>(null);
  const streamingMessageRef = useRef<string>("");
  const streamingLogCountRef = useRef<number>(0);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Auto-scroll logs (only if already near bottom)
  useEffect(() => {
    const el = logScrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    if (nearBottom) {
      el.scrollTop = el.scrollHeight;
    }
  }, [logs]);

  // Fetch data
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
        const loadedMessages: Message[] = hist.turns.map((turn, i) => ({
          id: `history-${i}`,
          role: turn.role === "user" ? "user" : "assistant",
          content: turn.content,
          timestamp: new Date(turn.timestamp * 1000),
        }));
        setMessages(loadedMessages);
      }
    } catch (err) {
      setError(`데이터 로드 실패: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, []);

  // Refresh performance
  const refreshPerformance = useCallback(async () => {
    try {
      const data = await apiGet<PerformanceData>("/api/performance");
      setPerformance(data);
    } catch {
      // ignore
    }
  }, []);

  // Debug
  const fetchDebugLogs = useCallback(async () => {
    try {
      const data = await apiGet<{ logs: string[]; enabled: boolean }>("/api/debug");
      setDebugEnabled(data.enabled);
    } catch {
      // ignore
    }
  }, []);

  const toggleDebug = useCallback(async () => {
    try {
      const data = await apiPost<{ enabled: boolean }>("/api/debug/toggle", {});
      setDebugEnabled(data.enabled);
      if (data.enabled) {
        await fetchDebugLogs();
      }
    } catch {
      // ignore
    }
  }, [fetchDebugLogs]);

  useEffect(() => {
    refreshData();
    fetchDebugLogs();
  }, [refreshData, fetchDebugLogs]);

  // Close character manager on outside click
  useEffect(() => {
    if (!showCharManager) return;
    const handler = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-char-manager]")) {
        setShowCharManager(false);
        setShowNewCharacter(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showCharManager]);

  // Fetch logs
  const fetchLogs = useCallback(async () => {
    try {
      const data = await apiGet<{ logs: string[]; total: number }>("/api/logs?level=all&limit=500");
      setLogs(data.logs);
    } catch {}
  }, []);

  // 5-second polling for logs when debug is enabled
  useEffect(() => {
    if (!debugEnabled) return;
    const interval = setInterval(fetchLogs, 5000);
    return () => clearInterval(interval);
  }, [debugEnabled, fetchLogs]);

  // WebSocket
  const connectWs = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const wsUrl =
      (API_BASE || window.location.origin).replace(/^http/, "ws") +
      "/api/ws/chat";
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = event.data;
      if (data === "[DONE]") {
        setIsStreaming(false);
        const logStart = streamingLogCountRef.current;
        apiGet<{ logs: string[] }>("/api/debug").then((debugData) => {
          const newLogs = debugData.logs.slice(logStart);
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === "assistant" && last.id === "streaming") {
              return [
                ...prev.slice(0, -1),
                { ...last, debugLogs: newLogs },
              ];
            }
            return prev;
          });
        });
        refreshPerformance();
        fetchLogs();
        return;
      }

      streamingMessageRef.current += data;
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && last.id === "streaming") {
          return [
            ...prev.slice(0, -1),
            { ...last, content: streamingMessageRef.current },
          ];
        }
        return [
          ...prev,
          {
            id: "streaming",
            role: "assistant",
            content: streamingMessageRef.current,
            timestamp: new Date(),
          },
        ];
      });
    };

    ws.onclose = () => {
      wsRef.current = null;
    };

    ws.onerror = () => {
      setIsStreaming(false);
      setError("WebSocket 연결 오류");
    };

    wsRef.current = ws;
  }, [refreshPerformance, fetchLogs]);

  // Send message
  const sendMessage = useCallback(() => {
    const currentInput = inputRef.current?.value || input;
    if (!currentInput.trim() || isStreaming) return;

    apiGet<{ logs: string[] }>("/api/debug").then((debugData) => {
      streamingLogCountRef.current = debugData.logs.length;
    });

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        role: "user",
        content: currentInput,
        timestamp: new Date(),
      },
    ]);
    setInput("");

    connectWs();
    const sendWhenReady = () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        streamingMessageRef.current = "";
        setIsStreaming(true);
        wsRef.current.send(currentInput);
      } else {
        setTimeout(sendWhenReady, 100);
      }
    };
    sendWhenReady();
  }, [input, isStreaming, connectWs]);

  // Export chat
  const exportChat = useCallback(() => {
    const text = messages.map((m) => `[${m.role}] ${m.content}`).join("\n\n");
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `chat-${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }, [messages]);

  // Save persona
  const savePersona = useCallback(
    async (data: PersonaData) => {
      try {
        await apiPut("/api/persona", data);
        await refreshData();
      } catch {
        setError("페르소나 저장 실패");
      }
    },
    [refreshData]
  );

  // Save knowledge
  const saveKnowledge = useCallback(
    async (name: string, content: string) => {
      try {
        await apiPut(`/api/knowledge/${name}`, { content });
        await refreshData();
      } catch {
        setError("지식 저장 실패");
      }
    },
    [refreshData]
  );

  // Load knowledge content
  const loadKnowledge = useCallback(async (name: string) => {
    const data = await apiGet<{ content: string }>(`/api/knowledge/${name}`);
    return data.content;
  }, []);

  // Reset character
  const resetCharacter = useCallback(
    async (opts: { memory: boolean; emotion: boolean; history: boolean }) => {
      if (!confirm("정말 초기화하시겠습니까?")) return;
      try {
        await apiPost("/api/character/reset", opts);
        if (opts.history) setMessages([]);
        await refreshData();
      } catch {
        setError("캐릭터 초기화 실패");
      }
    },
    [refreshData]
  );

  // Switch character
  const switchCharacter = useCallback(
    async (characterId: string) => {
      if (characterId === activeCharacter) return;
      if (!confirm("캐릭터를 전환하면 현재 대화가 초기화됩니다. 계속하시겠습니까?")) return;

      setSwitching(true);
      try {
        await apiPost("/api/character/switch", { character_id: characterId });
        setMessages([]);
        await refreshData();
      } catch {
        setError("캐릭터 전환 실패");
      } finally {
        setSwitching(false);
      }
    },
    [activeCharacter, refreshData]
  );

  // Create character
  const createCharacter = useCallback(async () => {
    if (!newName.trim()) return;
    try {
      await apiPost('/api/characters', { name: newName, identity: newIdentity });
      setNewName('');
      setNewIdentity('');
      setShowNewCharacter(false);
      await refreshData();
    } catch (err) {
      setError('캐릭터 생성 실패');
    }
  }, [newName, newIdentity, refreshData]);

  // Delete character
  const deleteCharacter = useCallback(async (characterId: string) => {
    if (!confirm(`정말 삭제하시겠습니까?`)) return;
    try {
      await apiDelete(`/api/characters/${characterId}`);
      await refreshData();
    } catch (err) {
      setError('캐릭터 삭제 실패');
    }
  }, [refreshData]);

  return (
    <div className="flex h-screen bg-background text-foreground">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="border-b px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold">Character OS</h1>
            {/* 캐릭터 선택 드롭다운 */}
            <div className="relative" data-char-manager>
              <button
                onClick={() => setShowCharManager(!showCharManager)}
                disabled={switching}
                className="flex items-center gap-1.5 h-8 rounded-md border border-input bg-background px-3 text-sm font-medium hover:bg-accent transition-colors disabled:opacity-50"
              >
                {switching ? (
                  <>
                    <div className="w-3 h-3 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                    <span>전환 중...</span>
                  </>
                ) : (
                  <>
                    <span>{characters.find(c => c.id === activeCharacter)?.name || "캐릭터 선택"}</span>
                    <span className="text-xs text-muted-foreground">▼</span>
                  </>
                )}
              </button>
              {showCharManager && !switching && (
                <div className="absolute top-full left-0 mt-1 z-50 bg-popover border rounded-lg shadow-lg p-1 min-w-[220px]">
                  {characters.map((c) => (
                    <div
                      key={c.id}
                      className={`flex items-center justify-between px-3 py-2 rounded-md text-sm transition-colors ${
                        c.id === activeCharacter
                          ? "bg-primary text-primary-foreground"
                          : "hover:bg-muted cursor-pointer"
                      }`}
                    >
                      <button
                        className="flex-1 text-left truncate"
                        onClick={() => {
                          if (c.id !== activeCharacter) switchCharacter(c.id);
                          setShowCharManager(false);
                        }}
                      >
                        {c.name}
                        {c.identity && (
                          <span className={`block text-xs truncate ${
                            c.id === activeCharacter ? "opacity-80" : "text-muted-foreground"
                          }`}>
                            {c.identity}
                          </span>
                        )}
                      </button>
                      {c.id !== activeCharacter && (
                        <button
                          className="ml-2 p-1 rounded hover:bg-destructive/10 text-muted-foreground hover:text-destructive transition-colors"
                          title="삭제"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteCharacter(c.id);
                          }}
                        >
                          ×
                        </button>
                      )}
                    </div>
                  ))}
                  <div className="border-t mt-1 pt-1">
                    {showNewCharacter ? (
                      <div className="px-2 py-2 space-y-2">
                        <Input
                          placeholder="캐릭터 이름"
                          value={newName}
                          onChange={(e) => setNewName(e.target.value)}
                          className="h-7 text-xs"
                          autoFocus
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && newName.trim()) createCharacter();
                            if (e.key === "Escape") {
                              setShowNewCharacter(false);
                              setNewName("");
                              setNewIdentity("");
                            }
                          }}
                        />
                        <Input
                          placeholder="한줄 소개 (선택)"
                          value={newIdentity}
                          onChange={(e) => setNewIdentity(e.target.value)}
                          className="h-7 text-xs"
                        />
                        <div className="flex gap-1">
                          <Button
                            size="sm"
                            className="h-7 flex-1 text-xs"
                            onClick={createCharacter}
                            disabled={!newName.trim()}
                          >
                            만들기
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            className="h-7 text-xs"
                            onClick={() => {
                              setShowNewCharacter(false);
                              setNewName("");
                              setNewIdentity("");
                            }}
                          >
                            취소
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <button
                        className="flex items-center gap-2 w-full px-3 py-2 rounded-md text-sm hover:bg-muted transition-colors"
                        onClick={() => setShowNewCharacter(true)}
                      >
                        <span>+</span>
                        <span>새 캐릭터</span>
                      </button>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setDark(!dark)}
              title={dark ? "라이트 모드" : "다크 모드"}
            >
              {dark ? "☀️" : "🌙"}
            </Button>
            <Button
              variant={debugOpen ? "default" : "ghost"}
              size="sm"
              onClick={() => {
                toggleDebug();
                setDebugOpen(!debugOpen);
              }}
              title="디버그 모드"
            >
              🐛
            </Button>
            <Button variant="outline" size="sm" onClick={exportChat}>
              내보내기
            </Button>
            <Sheet open={sheetOpen} onOpenChange={setSheetOpen}>
              <SheetTrigger>
                <Button variant="outline" size="sm">
                  설정
                </Button>
              </SheetTrigger>
              <SheetContent className="w-[340px] sm:w-[400px] p-0">
                <SheetHeader className="px-6 py-4 border-b">
                  <SheetTitle>설정</SheetTitle>
                </SheetHeader>
                <div className="flex-1 overflow-auto">
                  <Tabs
                    value={settingsTab}
                    onValueChange={(v) => setSettingsTab(v as typeof settingsTab)}
                    className="w-full"
                  >
                    <div className="px-6 pt-4">
                      <TabsList className="w-full grid grid-cols-5">
                        <TabsTrigger value="emotion">감정</TabsTrigger>
                        <TabsTrigger value="persona">페르소나</TabsTrigger>
                        <TabsTrigger value="knowledge">지식</TabsTrigger>
                        <TabsTrigger value="fewshot">예시</TabsTrigger>
                        <TabsTrigger value="reset">초기화</TabsTrigger>
                      </TabsList>
                    </div>
                    <div className="px-6 py-4">
                      <TabsContent value="emotion">
                        <div className="space-y-4">
                          <Card className="border border-border">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm">감정 상태</CardTitle>
                            </CardHeader>
                            <CardContent>
                              <EmotionPanel emotion={emotion} />
                            </CardContent>
                          </Card>
                          <Card className="border border-border">
                            <CardHeader className="pb-2">
                              <CardTitle className="text-sm">기억</CardTitle>
                            </CardHeader>
                            <CardContent>
                              <MemoryPanel stats={memoryStats} memories={memories} />
                            </CardContent>
                          </Card>
                        </div>
                      </TabsContent>
                      <TabsContent value="persona">
                        <Card className="border border-border">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">페르소나 수정</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <PersonaEditor persona={persona} onSave={savePersona} />
                          </CardContent>
                        </Card>
                      </TabsContent>
                      <TabsContent value="knowledge">
                        <Card className="border border-border">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">지식 파일</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <KnowledgeEditor
                              entries={knowledgeEntries}
                              onLoad={loadKnowledge}
                              onSave={saveKnowledge}
                            />
                          </CardContent>
                        </Card>
                      </TabsContent>
                      <TabsContent value="fewshot">
                        <Card className="border border-border">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">Few-shot 예시</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <FewShotPanel />
                          </CardContent>
                        </Card>
                      </TabsContent>
                      <TabsContent value="reset">
                        <Card className="border border-border">
                          <CardHeader className="pb-2">
                            <CardTitle className="text-sm">캐릭터 초기화</CardTitle>
                          </CardHeader>
                          <CardContent>
                            <ResetPanel onReset={resetCharacter} />
                          </CardContent>
                        </Card>
                      </TabsContent>
                    </div>
                  </Tabs>
                </div>
              </SheetContent>
            </Sheet>
          </div>
        </header>

        {/* Error banner */}
        {error && (
          <div className="bg-destructive/10 border-b border-destructive/30 px-4 py-2 flex items-center justify-between shrink-0">
            <span className="text-sm text-destructive">{error}</span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-xs text-destructive"
              onClick={() => setError(null)}
            >
              닫기
            </Button>
          </div>
        )}

        {/* Messages */}
        <div className="flex-1 overflow-auto p-4" ref={scrollRef}>
          {messages.length === 0 ? (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              <p>대화를 시작하세요</p>
            </div>
          ) : (
            messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg} isStreaming={isStreaming} />
            ))
          )}
        </div>

        {/* Input */}
        <div className="border-t p-4 shrink-0">
          <div className="flex gap-2 max-w-3xl mx-auto">
            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {
                if (e.key === "Enter" && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              placeholder={isStreaming ? "응답을 기다리는 중..." : "메시지를 입력하세요..."}
              disabled={isStreaming}
              className={`flex-1 ${isStreaming ? "opacity-60" : ""}`}
            />
            <Button onClick={sendMessage} disabled={isStreaming} className="min-w-[70px]">
              {isStreaming ? (
                <div className="flex items-center gap-1.5">
                  <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-1.5 h-1.5 bg-current rounded-full animate-bounce" />
                </div>
              ) : (
                "전송"
              )}
            </Button>
          </div>
          {isStreaming && (
            <p className="text-xs text-muted-foreground text-center mt-2 animate-pulse">
              캐릭터가 응답을 생성하고 있습니다...
            </p>
          )}
        </div>
      </div>

      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex flex-col w-72 border-l shrink-0">
        <div className="p-4 flex-1 overflow-auto space-y-4">
          <Card className="border border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">감정 상태</CardTitle>
            </CardHeader>
            <CardContent>
              <EmotionPanel emotion={emotion} />
            </CardContent>
          </Card>

          <Card className="border border-border">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">기억</CardTitle>
            </CardHeader>
            <CardContent>
              <MemoryPanel stats={memoryStats} memories={memories} />
            </CardContent>
          </Card>

          {debugOpen && (
            <>
              <Card className="border border-border">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">성능</CardTitle>
                </CardHeader>
                <CardContent>
                  <PerformancePanel performance={performance} onRefresh={refreshPerformance} />
                </CardContent>
              </Card>

              <Card className="border border-border">
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">시스템 로그</CardTitle>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-2 text-xs"
                      onClick={() => setLogModalOpen(true)}
                    >
                      전체 보기
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* 최근 5줄 미리보기 */}
                  <div className="font-mono text-[10px] leading-4 text-muted-foreground space-y-0.5">
                    {logs.length === 0 ? (
                      <p>로그 없음</p>
                    ) : (
                      logs.slice(-5).map((line, i) => (
                        <div key={i} className="truncate">{stripAnsi(line)}</div>
                      ))
                    )}
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2">총 {logs.length}줄</p>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </aside>

      {/* 로그 모달 */}
      {logModalOpen && (
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
      )}
    </div>
  );
}
