import { useState } from "react";

import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { EmotionPanel } from "@/components/EmotionPanel";
import { MemoryPanel } from "@/components/MemoryPanel";
import { PersonaEditor } from "@/components/PersonaEditor";
import { KnowledgeEditor } from "@/components/KnowledgeEditor";
import { FewShotPanel } from "@/components/FewShotPanel";
import { ResetPanel } from "@/components/ResetPanel";
import type { EmotionState, KnowledgeEntry, MemoryEntry, PersonaData } from "@/types";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  emotion: EmotionState | null;
  memoryStats: { count: number } | null;
  memories: MemoryEntry[];
  persona: PersonaData | null;
  knowledgeEntries: KnowledgeEntry[];
  savePersona: (data: PersonaData) => Promise<void>;
  saveKnowledge: (name: string, content: string) => Promise<void>;
  loadKnowledge: (name: string) => Promise<string>;
  resetCharacter: (opts: { memory: boolean; emotion: boolean; history: boolean }) => Promise<void>;
}

/** 감정·페르소나·지식·예시·초기화를 한데 모은 설정 패널. */
export function SettingsSheet({
  open,
  onOpenChange,
  emotion,
  memoryStats,
  memories,
  persona,
  knowledgeEntries,
  savePersona,
  saveKnowledge,
  loadKnowledge,
  resetCharacter,
}: Props) {
  const [settingsTab, setSettingsTab] = useState<
    "emotion" | "persona" | "knowledge" | "fewshot" | "reset"
  >("emotion");
  const sheetOpen = open;
  const setSheetOpen = onOpenChange;

  return (
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
  );
}
