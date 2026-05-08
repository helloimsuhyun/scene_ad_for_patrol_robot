#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
관리자용 이벤트 초기화 스크립트

사용 예시:

1) 전체 이벤트 초기화
python3 reset_events_admin.py --all

2) 비전 이벤트만 초기화
python3 reset_events_admin.py --vision

3) 오디오 + 인증 이벤트만 초기화
python3 reset_events_admin.py --audio --auth

4) YOLO/person 이벤트만 초기화
python3 reset_events_admin.py --yolo

5) DB만 삭제하고 파일은 유지
python3 reset_events_admin.py --all --db_only

6) 서버 주소 직접 지정
python3 reset_events_admin.py --server_url http://127.0.0.1:8000 --all
"""

import argparse
import json
import sys
from typing import Any, Dict

import requests


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


def print_json(data: Dict[str, Any]):
    print(json.dumps(data, indent=2, ensure_ascii=False))


def reset_events(
    server_url: str,
    vision: bool,
    audio: bool,
    auth: bool,
    yolo: bool,
    delete_files: bool,
):
    url = server_url.rstrip("/") + "/admin/reset_events"

    payload = {
        "vision": vision,
        "audio": audio,
        "auth": auth,
        "yolo": yolo,
        "delete_files": delete_files,
    }

    print("========================================")
    print("[ADMIN RESET EVENTS]")
    print("========================================")
    print(f"server_url   : {server_url}")
    print(f"endpoint     : {url}")
    print(f"vision       : {vision}")
    print(f"audio        : {audio}")
    print(f"auth         : {auth}")
    print(f"yolo         : {yolo}")
    print(f"delete_files : {delete_files}")
    print("========================================")

    try:
        resp = requests.post(url, json=payload, timeout=30)

    except requests.exceptions.ConnectionError:
        print()
        print("[ERROR] 서버에 연결할 수 없습니다.")
        print("서버가 실행 중인지 확인하세요.")
        print()
        print("예:")
        print("cd ~/scene_ad_for_patrol_robot")
        print("./run_servers.sh")
        sys.exit(1)

    except requests.exceptions.Timeout:
        print()
        print("[ERROR] 요청 시간이 초과되었습니다.")
        sys.exit(1)

    print()
    print(f"[HTTP STATUS] {resp.status_code}")

    try:
        data = resp.json()
    except Exception:
        print("[ERROR] JSON 응답 파싱 실패")
        print(resp.text)
        sys.exit(1)

    print()
    print("[SERVER RESPONSE]")
    print_json(data)

    if resp.status_code != 200 or not data.get("ok", False):
        print()
        print("[ERROR] 이벤트 초기화 실패")
        sys.exit(1)

    print()
    print("[DONE] 이벤트 초기화 완료")


def main():
    parser = argparse.ArgumentParser(
        description="Admin tool for resetting event DB rows and event files."
    )

    parser.add_argument(
        "--server_url",
        default=DEFAULT_SERVER_URL,
        help=f"FastAPI server URL. Default: {DEFAULT_SERVER_URL}",
    )

    parser.add_argument(
        "--vision",
        action="store_true",
        help="Reset vision events. Files: delete only recv/<place_id>/query contents.",
    )

    parser.add_argument(
        "--audio",
        action="store_true",
        help="Reset audio events.",
    )

    parser.add_argument(
        "--auth",
        action="store_true",
        help="Reset auth events.",
    )

    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Reset YOLO/person events. Files: delete recv_person contents.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Reset vision, audio, auth, and YOLO/person events.",
    )

    parser.add_argument(
        "--core",
        action="store_true",
        help="Reset vision, audio, and auth events only. YOLO/person events are kept.",
    )

    parser.add_argument(
        "--db_only",
        action="store_true",
        help="Delete DB rows only. Keep files.",
    )

    args = parser.parse_args()

    if args.all:
        vision = True
        audio = True
        auth = True
        yolo = True

    elif args.core:
        vision = True
        audio = True
        auth = True
        yolo = False

    else:
        vision = args.vision
        audio = args.audio
        auth = args.auth
        yolo = args.yolo

    if not any([vision, audio, auth, yolo]):
        print("[ERROR] 삭제할 이벤트 종류를 지정하세요.")
        print()
        print("예:")
        print("python3 reset_events_admin.py --all")
        print("python3 reset_events_admin.py --core")
        print("python3 reset_events_admin.py --vision")
        print("python3 reset_events_admin.py --audio --auth")
        print("python3 reset_events_admin.py --yolo")
        sys.exit(1)

    reset_events(
        server_url=args.server_url,
        vision=vision,
        audio=audio,
        auth=auth,
        yolo=yolo,
        delete_files=not args.db_only,
    )


if __name__ == "__main__":
    main()