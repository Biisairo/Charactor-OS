// 서버 로그에는 ANSI 색상 코드가 섞여 온다. 화면에서는 지운다.

export function stripAnsi(str: string): string {
  return str.replace(/\x1b\[[0-9;]*m/g, '');
}
