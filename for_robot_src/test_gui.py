#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
from pprint import pprint

SERVER_URL = "http://127.0.0.1:8000"      # vision server
BRIDGE_URL = "http://192.168.0.88"   # ROS2 patrol_http_bridge device IP
TIMEOUT = 5

MODE_CYCLE = ["idle", "bank", "th_calib", "query"]
VALID_ROBOT_COMMANDS = ["idle", "start", "pause", "resume", "stop", "teach", "reload_waypoints"]


# =========================
# Vision server API
# =========================
def get_places():
    resp = requests.get(f"{SERVER_URL}/places", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_place(place_id: str):
    resp = requests.get(f"{SERVER_URL}/places/{place_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_calibration_status():
    resp = requests.get(f"{SERVER_URL}/calibration_status", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def recalibrate_all():
    resp = requests.post(f"{SERVER_URL}/places/recalibrate_all", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def set_mode(place_id: str, mode: str):
    resp = requests.post(
        f"{SERVER_URL}/places/{place_id}/config",
        data={"mode": mode},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def delete_place(place_id: str):
    resp = requests.delete(f"{SERVER_URL}/places/{place_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def delete_all_places():
    resp = requests.delete(f"{SERVER_URL}/places", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def delete_threshold(place_id: str):
    resp = requests.delete(f"{SERVER_URL}/places/{place_id}/threshold", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_robot_command():
    resp = requests.get(f"{SERVER_URL}/robot/command", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def set_robot_command(command: str):
    command = normalize_robot_command(command)
    resp = requests.post(
        f"{SERVER_URL}/robot/command",
        json={"command": command},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def get_robot_pose():
    resp = requests.get(f"{SERVER_URL}/robot/pose", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def get_robot_goal():
    resp = requests.get(f"{SERVER_URL}/robot/goal", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# =========================
# ROS2 Patrol bridge API
# =========================
def set_patrol_place(place_id: str):
    resp = requests.post(
        f"{BRIDGE_URL}/patrol/place",
        json={"place_id": place_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def set_query_gt(label: str):
    label = normalize_label(label)

    resp = requests.post(
        f"{BRIDGE_URL}/patrol/query_gt",
        json={"label": label},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def trigger_capture():
    resp = requests.post(
        f"{BRIDGE_URL}/patrol/capture",
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def place_and_capture(place_id: str):
    resp = requests.post(
        f"{BRIDGE_URL}/patrol/place_and_capture",
        json={"place_id": place_id},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# =========================
# print helpers
# =========================
def print_help():
    print("\n=== GUI API Test ===")
    print("single-key commands:")
    print("  p          -> 다음 place")
    print("  o          -> 이전 place")
    print("  m          -> mode toggle (idle -> bank -> th_calib -> query)")
    print("  z          -> query label toggle (normal <-> abnormal)")
    print("  c          -> 현재 place로 capture")
    print("  v          -> 현재 place 설정 후 capture")
    print("  l          -> 현재 place 상세 조회")
    print("  g          -> 현재 robot pose / goal / next_place_id 조회")
    print("  s          -> calibration status 조회")
    print("  w          -> calibration status watch")
    print("  r          -> 전체 place recalibration 시작")
    print("")
    print("multi-key commands:")
    print("  places                         -> 전체 place 조회")
    print("  place <place_id>               -> 현재 선택 place 설정/조회")
    print("  mode <place_id> <mode>         -> 특정 place mode 변경")
    print("     valid mode: idle bank th_calib query")
    print("  cmdget                         -> 현재 robot command 조회")
    print("  cmd <command>                  -> robot command 설정")
    print("     valid command: idle start pause resume stop teach reload_waypoints")
    print("  delplace <place_id>            -> 특정 place 삭제")
    print("  delall                         -> 전체 place 삭제")
    print("  delth <place_id>               -> 특정 place threshold 삭제")
    print("  help                           -> 도움말")
    print("  exit                           -> 종료")
    print("")
    print("note:")
    print("  - capture 시 mode는 서버의 place.mode 기준으로 결정됨")
    print("  - z 는 query일 때 사용할 GT 라벨(normal/abnormal)만 토글")
    print("  - bank / th_calib 에서는 서버가 저장만 하고, query면 서버가 추론함")


def print_calibration_status(data: dict):
    print("\n=== CALIBRATION STATUS ===")
    print("ok:", data.get("ok"))
    print("global_calibrating:", data.get("global_calibrating"))

    prog = data.get("calib_progress", {})
    total = prog.get("total", 0)
    done = prog.get("done", 0)
    current = prog.get("current_place_id")
    pct = (done / total * 100.0) if total else 0.0

    print(f"calib_progress: {done}/{total} ({pct:.1f}%)")
    print("current_place_id:", current)


def print_robot_command(data: dict):
    print("\n=== ROBOT COMMAND ===")
    print("ok:", data.get("ok"))
    print("command:", data.get("command"))
    print("timestamp:", data.get("timestamp"))


def print_robot_state():
    pose_data = get_robot_pose()
    goal_data = get_robot_goal()

    pose = pose_data.get("pose", {}) or {}
    goal = goal_data.get("goal", {}) or {}

    print("\n=== ROBOT STATE ===")
    print(
        f"robot_pose: x={pose.get('x')}, y={pose.get('y')}, yaw={pose.get('yaw')}"
    )
    print(f"robot_status: {pose.get('status')}")
    print(f"pose_timestamp: {pose.get('timestamp')}")
    print(
        f"robot_goal: x={goal.get('x')}, y={goal.get('y')}, yaw={goal.get('yaw')}"
    )
    print(f"next_place_id: {goal.get('next_place_id')}")
    print(f"goal_timestamp: {goal.get('timestamp')}")


def print_one_place(p: dict):
    place_id = p.get("place_id")
    mode = p.get("mode")
    bank_count = p.get("bank_count")
    th_calib_count = p.get("th_calib_count")
    threshold_ready = p.get("threshold_ready")
    need_calibration = p.get("need_calibration")
    ready_for_calibration = p.get("ready_for_calibration")
    bank_target = p.get("bank_target")
    th_calib_target = p.get("th_calib_target")
    updated_at = p.get("updated_at")

    print(f"place_id: {place_id}")
    print(f"mode: {mode}")
    print(f"bank: {bank_count}/{bank_target}")
    print(f"th_calib: {th_calib_count}/{th_calib_target}")
    print(f"threshold_ready: {threshold_ready}")
    print(f"need_calibration: {need_calibration}")
    print(f"ready_for_calibration: {ready_for_calibration}")
    print(f"updated_at: {updated_at}")


def print_places_summary(data: dict, current_place: str | None = None):
    print("\n=== PLACES ===")
    print("ok:", data.get("ok"))
    print("global_calibrating:", data.get("global_calibrating"))

    prog = data.get("calib_progress")
    if prog is not None:
        total = prog.get("total", 0)
        done = prog.get("done", 0)
        current = prog.get("current_place_id")
        pct = (done / total * 100.0) if total else 0.0
        print(f"calib_progress: {done}/{total} ({pct:.1f}%)")
        print("current_place_id:", current)

    places = data.get("places", [])
    print(f"n_places: {len(places)}")
    for p in places:
        place_id = p.get("place_id")
        mode = p.get("mode")
        bank_count = p.get("bank_count")
        th_calib_count = p.get("th_calib_count")
        threshold_ready = p.get("threshold_ready")
        need_calibration = p.get("need_calibration")
        ready_for_calibration = p.get("ready_for_calibration")

        marker = "<-- current" if str(place_id) == str(current_place) else ""

        print(
            f"- place_id={place_id} | mode={mode} | "
            f"bank={bank_count} | th_calib={th_calib_count} | "
            f"threshold_ready={threshold_ready} | "
            f"need_calibration={need_calibration} | "
            f"ready_for_calibration={ready_for_calibration} {marker}"
        )


def watch_calibration_status(interval_sec: float = 1.0):
    print("\n[watch] calibration status polling 시작 (Ctrl+C 로 종료)\n")
    try:
        while True:
            data = get_calibration_status()
            prog = data.get("calib_progress", {})
            total = prog.get("total", 0)
            done = prog.get("done", 0)
            current = prog.get("current_place_id")
            running = data.get("global_calibrating", False)
            pct = (done / total * 100.0) if total else 0.0

            print(
                f"[status] running={running} | "
                f"{done}/{total} ({pct:.1f}%) | current_place_id={current}"
            )

            if not running:
                print("[watch] calibration finished")
                break

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("\n[watch] stopped by user")


# =========================
# helpers
# =========================
def normalize_place_id(value: str) -> str:
    s = str(value).strip()

    if not s.isdigit():
        raise ValueError("place_id must be numeric, e.g. 00, 01, 02")

    n = int(s)
    if n < 0:
        raise ValueError("place_id must be >= 0")

    return f"{n:02d}"


def normalize_label(value: str) -> str:
    s = str(value).strip().lower()

    alias = {
        "n": "normal",
        "normal": "normal",
        "a": "abnormal",
        "ab": "abnormal",
        "abnormal": "abnormal",
        "unnormal": "abnormal",
    }

    if s not in alias:
        raise ValueError("label must be one of: normal, abnormal")

    return alias[s]


def normalize_robot_command(value: str) -> str:
    s = str(value).strip().lower()
    if s not in VALID_ROBOT_COMMANDS:
        raise ValueError(f"command must be one of: {', '.join(VALID_ROBOT_COMMANDS)}")
    return s


def next_place_id(current_place: str | None) -> str:
    if current_place is None:
        return "00"

    n = int(current_place)
    return f"{n + 1:02d}"


def prev_place_id(current_place: str | None) -> str:
    if current_place is None:
        return "00"

    n = int(current_place)
    n = max(0, n - 1)
    return f"{n:02d}"


def ensure_current_place(current_place: str | None) -> str:
    if current_place is None:
        return "00"
    return normalize_place_id(current_place)


def get_mode_of_place(place_id: str) -> str:
    data = get_place(place_id)
    mode = data.get("place", {}).get("mode", "idle")
    return str(mode)


def next_mode(mode: str) -> str:
    mode = str(mode)
    if mode not in MODE_CYCLE:
        return "idle"
    idx = MODE_CYCLE.index(mode)
    return MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]


def toggle_label(label: str) -> str:
    return "abnormal" if label == "normal" else "normal"


def apply_mode_to_current(current_place: str, mode: str):
    data = set_mode(current_place, mode)
    print(f"[mode] place_id={current_place} -> {mode}")
    place = data.get("place", {})
    if place:
        print_one_place(place)
    else:
        pprint(data)
    return data


def show_current_place(current_place: str):
    data = get_place(current_place)
    print("\n=== PLACE ===")
    print("ok:", data.get("ok"))
    print("global_calibrating:", data.get("global_calibrating"))
    place = data.get("place", {})
    print_one_place(place)
    print(f"current_selected_place: {current_place}")
    return data


def main():
    current_place = "00"
    current_label = "normal"

    print_help()
    print(f"\ncurrent_selected_place: {current_place}")
    print(f"current_query_label: {current_label}")

    while True:
        raw = input("\ncmd> ").strip()
        if not raw:
            continue

        cmd = raw.split()
        op = cmd[0].lower()

        try:
            if op == "exit":
                break

            elif op == "help":
                print_help()

            elif op == "places":
                data = get_places()
                print_places_summary(data, current_place=current_place)
                print("current_selected_place:", current_place)
                print("current_query_label:", current_label)

            elif op == "place":
                if len(cmd) != 2:
                    print("usage: place <place_id>")
                    continue

                current_place = normalize_place_id(cmd[1])
                print(f"[place] current_selected_place: {current_place}")

                try:
                    show_current_place(current_place)
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status == 404:
                        print(f"[place] {current_place} is not created on server yet")
                    else:
                        raise

            elif op == "p":
                current_place = next_place_id(current_place)
                print(f"[place] current_selected_place: {current_place}")

            elif op == "o":
                current_place = prev_place_id(current_place)
                print(f"[place] current_selected_place: {current_place}")

            elif op == "m":
                current_place = ensure_current_place(current_place)
                current_mode = get_mode_of_place(current_place)
                new_mode = next_mode(current_mode)
                apply_mode_to_current(current_place, new_mode)
                print(f"[mode cycle] {current_mode} -> {new_mode}")

            elif op == "z":
                current_label = toggle_label(current_label)
                data = set_query_gt(current_label)
                print(f"[label toggle] query_label -> {current_label}")
                pprint(data)

            elif op == "c":
                current_place = ensure_current_place(current_place)

                set_patrol_place(current_place)
                set_query_gt(current_label)

                data = trigger_capture()
                print(f"[capture] place_id={current_place} gt={current_label} (server decides mode)")
                pprint(data)

            elif op == "v":
                current_place = ensure_current_place(current_place)

                set_query_gt(current_label)

                data = place_and_capture(current_place)
                print(f"[place_and_capture] place_id={current_place} gt={current_label} (server decides mode)")
                pprint(data)

            elif op == "l":
                current_place = ensure_current_place(current_place)
                show_current_place(current_place)
                print(f"current_query_label: {current_label}")

            elif op == "g":
                print_robot_state()

            elif op == "s":
                data = get_calibration_status()
                print_calibration_status(data)

            elif op == "r":
                data = recalibrate_all()
                pprint(data)

            elif op == "w":
                watch_calibration_status()

            elif op == "cmdget":
                data = get_robot_command()
                print_robot_command(data)

            elif op == "cmd":
                if len(cmd) != 2:
                    print("usage: cmd <idle|start|pause|resume|stop|teach|reload_waypoints>")
                    continue

                data = set_robot_command(cmd[1])
                print_robot_command({
                    "ok": data.get("ok"),
                    "command": data.get("command", {}).get("command"),
                    "timestamp": data.get("command", {}).get("timestamp"),
                })

            elif op == "mode":
                if len(cmd) != 3:
                    print("usage: mode <place_id> <idle|bank|th_calib|query>")
                    continue

                place_id = normalize_place_id(cmd[1])
                mode = cmd[2]
                data = set_mode(place_id, mode)

                print("\n=== SET MODE RESULT ===")
                print("ok:", data.get("ok"))
                place = data.get("place", {})
                if place:
                    print_one_place(place)
                else:
                    pprint(data)

            elif op == "delplace":
                if len(cmd) != 2:
                    print("usage: delplace <place_id>")
                    continue
                data = delete_place(normalize_place_id(cmd[1]))
                pprint(data)

            elif op == "delall":
                data = delete_all_places()
                pprint(data)

            elif op == "delth":
                if len(cmd) != 2:
                    print("usage: delth <place_id>")
                    continue
                data = delete_threshold(normalize_place_id(cmd[1]))
                print("\n=== DELETE THRESHOLD RESULT ===")
                print("ok:", data.get("ok"))
                print("status:", data.get("status"))
                place = data.get("place", {})
                if place:
                    print_one_place(place)
                else:
                    pprint(data)

            else:
                print("unknown command. type 'help'")

        except requests.HTTPError as e:
            print("[HTTP ERROR]")
            try:
                print("status:", e.response.status_code)
                print("body:", e.response.text)
            except Exception:
                print(e)

        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    main()