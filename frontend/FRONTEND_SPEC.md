# 프론트엔드 기능 명세서

> 현재 상태 기반 문제점 + 개선 명세

---

## 1. 레이아웃 구조

### 현재 문제

- **설정 시트 탭 7개** (`grid-cols-7`): 모바일에서 탭 텍스트 잘림, 클릭 영역 부족
- **데스크톱 사이드바**: 감정/기억만 표시. 로그·성능은 설정 시트에만 있음
- **로그·성능 패널 중복**: 사이드바와 시트에 동일 컴포넌트 복사
- **`debugOpen` 상태 미사용**: `debugOpen` state가 있지만 로그 표시 토글에 연결 안 됨

### 개선 명세

```
┌─────────────────────────────────────────────────────────┐
│ Header                                                  │
│ [Character OS] [캐릭터 선택 ▼] [🌙] [🐛] [내보내기] [설정]│
├──────────────────────────────────┬──────────────────────┤
│                                  │ 사이드바 (데스크톱)   │
│          채팅 영역                │ ┌──────────────────┐ │
│                                  │ │ 감정 상태         │ │
│                                  │ ├──────────────────┤ │
│                                  │ │ 기억              │ │
│                                  │ ├──────────────────┤ │
│                                  │ │ 성능 (🐛토글 시)  │ │
│                                  │ ├──────────────────┤ │
│                                  │ │ 로그 (🐛토글 시)  │ │
│                                  │ └──────────────────┘ │
├──────────────────────────────────┴──────────────────────┤
│ [메시지 입력...]                              [전송]     │
└─────────────────────────────────────────────────────────┘
```

**변경점:**
- 사이드바에 성능·로그 패널 추가 (🐛 디버그 토글로 표시/숨김)
- `debugOpen` → 로그·성능 패널 visibility에 연결
- 설정 시트 탭 수 축소 (감정/페르소나/지식/예시/초기화 = 5개)

---

## 2. 캐릭터 선택

### 현재 문제

- `<select>` 드롭다운이 헤더에 있지만, 전환 시 피드백 없음
- 캐릭터 전환 시 대화 기록이 남아있음 (이전 캐릭터 대화)
- 전환 중 로딩 상태 없음

### 개선 명세

```tsx
// 상태
const [switching, setSwitching] = useState(false);

// 전환 로직
const switchCharacter = async (characterId: string) => {
  if (characterId === activeCharacter) return;
  if (!confirm("캐릭터를 전환하면 현재 대화가 초기화됩니다. 계속하시겠습니까?")) return;
  
  setSwitching(true);
  try {
    await apiPost("/api/character/switch", { character_id: characterId });
    setMessages([]);  // 대화 초기화
    await refreshData();
  } catch (err) {
    alert("캐릭터 전환 실패");
  } finally {
    setSwitching(false);
  }
};
```

**UI:**
- 전환 중 `<select>` 비활성화 + 로딩 인디케이터
- 현재 활성 캐릭터 이름을 헤더 배지에 표시 (이미 있음)
- 캐릭터 목록이 1개일 때는 `<select>` 숨김

---

## 3. 로그 뷰어

### 현재 문제

- 로그가 실시간 갱신 안 됨 (채팅 후에만 갱신)
- 로그 라인 구분 없음 (시간/레벨/모듈)
- 필터가 "전체/에러"만 있음
- 로그가 너무 많으면 성능 저하 (가상 스크롤 없음)

### 개선 명세

```tsx
// 로그 파싱 (모듈별 색상)
function parseLogLine(line: string): { module: string; message: string; level: string } {
  const stripped = stripAnsi(line);
  // [모듈명] 메시지 형태 파싱
  const match = stripped.match(/^\[?(\w+)\]?\s*(.*)/);
  if (match) {
    const module = match[1].toLowerCase();
    const message = match[2];
    const level = message.includes("실패") || message.includes("오류") ? "error" : "info";
    return { module, message, level };
  }
  return { module: "system", message: stripped, level: "info" };
}

// 모듈별 색상
const MODULE_COLORS: Record<string, string> = {
  orchestrator: "text-blue-500",
  react: "text-cyan-500",
  response: "text-yellow-500",
  postprocess: "text-green-500",
  reflection: "text-purple-500",
};
```

**UI:**
- 로그 라인: `[시간] [모듈] 메시지` 형태로 포맷
- 모듈별 색상 구분 (ANSI 색상 대신 Tailwind 클래스)
- 필터: 전체 / 에러 / 모듈별
- 최대 200줄 표시 (older logs auto-trim)
- 자동 스크롤 (최신 로그 하단 고정)

---

## 4. 성능 대시보드

### 현재 문제

- 성능 데이터가 채팅 후에만 갱신
- 수동 갱신 버튼 없음
- 스테이지 이름이 영어 (`context`, `response`, `postprocess`)
- 이력(history) 그래프 없음

### 개선 명세

```tsx
// 스테이지 이름 한글화
const STAGE_LABELS: Record<string, string> = {
  context: "컨텍스트 수집",
  response: "응답 생성",
  postprocess: "후처리",
};

// 성능 데이터 갱신 (수동 + 자동)
const refreshPerformance = useCallback(async () => {
  try {
    const data = await apiGet<PerformanceData>("/api/performance");
    setPerformance(data);
  } catch {}
}, []);
```

**UI:**
- 총 지연 시간: 큰 숫자 + 단위 (ms)
- 스테이지별 바 차트 (가로 막대, 비율 표시)
- 기억/기록 수 표시
- "새로고침" 버튼 (수동 갱신)
- 이전 대화 성능 기록 (최근 5회, sparkline)

---

## 5. 설정 시트 정리

### 현재 문제

- 탭 7개 → 모바일에서 잘림
- 로그·성능이 시트와 사이드바에 중복

### 개선 명세

**시트 탭 5개로 축소:**
1. 감정 (감정 상태 + 기억)
2. 페르소나
3. 지식
4. 예시
5. 초기화

로그·성능은 사이드바에서만 표시 (🐛 토글로 제어)

---

## 6. 기타 수정 사항

| 항목 | 현재 | 개선 |
|------|------|------|
| `debugOpen` 미사용 | state만 존재 | 로그·성능 패널 visibility에 연결 |
| 캐릭터 전환 확인 | 없음 | confirm 다이얼로그 |
| 전환 로딩 | 없음 | `<select>` 비활성화 + 스피너 |
| 에러 알림 | `console.error`만 | 토스트 또는 인라인 에러 메시지 |
| 로그 자동 갱신 | 채팅 후만 | 5초 폴링 (debug 모드일 때) |
| 성능 새로고침 | 없음 | 수동 버튼 + 채팅 후 자동 |

---

## 7. 구현 우선순위

1. **_layout fix_**: 사이드바에 로그·성능 추가, `debugOpen` 연결, 시트 탭 축소
2. **_character switch_**: 확인 다이얼로그, 대화 초기화, 로딩 상태
3. **_log viewer_**: 파싱, 모듈별 색상, 필터, 자동 갱신
4. **_performance_**: 한글 라벨, 바 차트, 수동 새로고침
5. **_error handling_**: 인라인 에러 메시지

---

*작성일: 2026-08-11*
