#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
from pprint import pprint

SERVER_URL = "http://127.0.0.1:8000"
TIMEOUT = 5


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


def print_help():
    print("=== GUI API Test ===")
    print("commands:")
    print("  places                         -> 전체 place 조회")
    print("  place <place_id>               -> 특정 place 조회")
    print("  mode <place_id> <mode>         -> place mode 변경")
    print("     valid mode: idle bank th_calib query")
    print("  delplace <place_id>            -> 특정 place 삭제")
    print("  delall                         -> 전체 place 삭제")
    print("  delth <place_id>               -> 특정 place threshold 삭제")
    print("  status                         -> calibration status 조회")
    print("  recaliball                     -> 전체 place recalibration 시작")
    print("  watch                          -> calibration status 1초 polling")
    print("  help                           -> 도움말")
    print("  exit                           -> 종료")


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


def print_places_summary(data: dict):
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

        print(
            f"- place_id={place_id} | mode={mode} | "
            f"bank={bank_count} | th_calib={th_calib_count} | "
            f"threshold_ready={threshold_ready} | "
            f"need_calibration={need_calibration} | "
            f"ready_for_calibration={ready_for_calibration}"
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


def main():
    print_help()

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
                print_places_summary(data)

            elif op == "place":
                if len(cmd) != 2:
                    print("usage: place <place_id>")
                    continue
                data = get_place(cmd[1])
                print("\n=== PLACE ===")
                print("ok:", data.get("ok"))
                print("global_calibrating:", data.get("global_calibrating"))
                place = data.get("place", {})
                print_one_place(place)

            elif op == "mode":
                if len(cmd) != 3:
                    print("usage: mode <place_id> <idle|bank|th_calib|query>")
                    continue
                place_id = cmd[1]
                mode = cmd[2]
                data = set_mode(place_id, mode)
                print("\n=== SET MODE RESULT ===")
                print("ok:", data.get("ok"))
                place = data.get("place", {})
                print_one_place(place)

            elif op == "delplace":
                if len(cmd) != 2:
                    print("usage: delplace <place_id>")
                    continue
                data = delete_place(cmd[1])
                pprint(data)

            elif op == "delall":
                data = delete_all_places()
                pprint(data)

            elif op == "delth":
                if len(cmd) != 2:
                    print("usage: delth <place_id>")
                    continue
                data = delete_threshold(cmd[1])
                print("\n=== DELETE THRESHOLD RESULT ===")
                print("ok:", data.get("ok"))
                print("status:", data.get("status"))
                place = data.get("place", {})
                print_one_place(place)

            elif op == "status":
                data = get_calibration_status()
                print_calibration_status(data)

            elif op == "recaliball":
                data = recalibrate_all()
                pprint(data)

            elif op == "watch":
                watch_calibration_status()

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