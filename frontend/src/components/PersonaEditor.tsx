import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { PersonaData } from "@/types";
import { ArrayInput } from "@/components/ArrayInput";

export function PersonaEditor({
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
