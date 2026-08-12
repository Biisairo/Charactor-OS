// 서버 로그에는 ANSI 색상 코드가 섞여 온다. 화면에서는 지운다.

export function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}

/** 턴당 비용은 $0.001 단위라 소수점 이하가 길다. 유효숫자를 살려 자른다. */
export function formatCost(usd: number | null | undefined): string {
  // null은 "단가표에 없는 모델"이고 0은 "정말 0"이다. 둘을 뭉개면
  // 비용이 0인 것처럼 보인다 (TASK-12가 구별한 것을 화면에서 되돌리지 않는다).
  if (usd == null) return "단가 미등록";
  if (usd === 0) return "$0";
  if (usd < 0.01) return `$${usd.toFixed(6)}`;
  return `$${usd.toFixed(4)}`;
}

export function formatTokens(n: number): string {
  return n.toLocaleString("en-US");
}
