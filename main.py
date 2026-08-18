import argparse
from pathlib import Path

import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory

from src.call_log import from_config as call_logger_from_config
from src.character_os import CharacterOS
from src.trace import format_trace

# 상태 경로는 기본값을 두지 않는다. 생략하면 `CharacterOS`가
# `characters/<id>/state/` 로 파생해 캐릭터마다 분리된다 (TASK-17).
DEFAULT_CONFIG = {
    "character_dir": "characters/hong-gil-dong",
    "model_type": "api",
    "local_model": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "adapter_path": None,
}


def load_config(path: str) -> dict:
    config_file = Path(path)
    if config_file.exists():
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    parser = argparse.ArgumentParser(description="Character OS")
    parser.add_argument("--config", default="config.yaml", help="설정 파일 경로")
    parser.add_argument("--character", help="캐릭터 디렉토리 경로 (설정 파일 오버라이드)")
    parser.add_argument("--debug", action="store_true", help="디버그 모드")
    parser.add_argument(
        "--no-review", action="store_true", help="Reflection 검토 비활성화 (비용 절감)"
    )
    parser.add_argument("--trace", action="store_true", help="파이프라인 트레이싱 활성화")
    args = parser.parse_args()

    # 설정 로드: defaults → config.yaml → CLI 인자
    config = dict(DEFAULT_CONFIG)
    config.update(load_config(args.config))
    if args.character:
        config["character_dir"] = args.character

    debug = args.debug

    print("=== Character OS ===\n")

    if debug:
        print("[DEBUG 모드 활성화]\n")
        print(f"[설정] character: {config['character_dir']}")
        print()
        print()

    # 캐릭터 OS 초기화
    cos = CharacterOS(
        character_dir=config["character_dir"],
        memory_db_path=config.get("memory_db_path"),
        emotion_save_path=config.get("emotion_save_path"),
        history_save_path=config.get("history_save_path"),
        working_memory_path=config.get("working_memory_path"),
        debug=debug,
        output=print,
        debug_output=print,
        model_type=config["model_type"],
        local_model=config["local_model"],
        adapter_path=config["adapter_path"],
        no_review=args.no_review,
        trace=args.trace,
        call_logger=call_logger_from_config(config),
    )

    print("캐릭터가 준비되었습니다. 대화를 시작하세요!")
    print("(종료하려면 'quit' 또는 'exit'를 입력하세요)\n")

    # prompt_toolkit 세션 생성 (히스토리 자동 저장)
    session = PromptSession(
        history=FileHistory(".chat_history"),
        auto_suggest=AutoSuggestFromHistory(),
    )

    while True:
        try:
            user_input = session.prompt("사용자: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("대화를 종료합니다.")
                break

            print()
            cos.chat(user_input)
            if args.trace:
                print()
                print(format_trace(cos._last_trace))
            print()

        except KeyboardInterrupt:
            print("\n대화를 종료합니다.")
            break
        except EOFError:
            print("\n대화를 종료합니다.")
            break
        except Exception as e:
            print(f"\n오류 발생: {e}\n")
            if debug:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
