#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def recalibrate_place(place_id: str):
    resp = requests.post(f"{SERVER_URL}/places/{place_id}/recalibrate", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def recalibrate_all():
    resp = requests.post(f"{SERVER_URL}/places/recalibrate_all", timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def print_help():
    print("=== GUI API Test ===")
    print("commands:")
    print("  places                       -> 전체 place 조회")
    print("  show <place_id>              -> 특정 place 조회")
    print("  bank <place_id>              -> mode=bank")
    print("  calib <place_id>             -> mode=th_calib")
    print("  query <place_id>             -> mode=query")
    print("  del <place_id>               -> 특정 place 삭제")
    print("  delall                       -> 전체 place 삭제")
    print("  delth <place_id>             -> 특정 place threshold 삭제")
    print("  recalib <place_id>           -> 특정 place recalibration")
    print("  recaliball                   -> 전체 place recalibration")
    print("  help                         -> 도움말")
    print("  exit                         -> 종료")


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
                pprint(get_places())

            elif op == "show":
                if len(cmd) < 2:
                    print("usage: show <place_id>")
                    continue
                pprint(get_place(cmd[1]))

            elif op == "bank":
                if len(cmd) < 2:
                    print("usage: bank <place_id>")
                    continue
                pprint(set_mode(cmd[1], "bank"))

            elif op == "calib":
                if len(cmd) < 2:
                    print("usage: calib <place_id>")
                    continue
                pprint(set_mode(cmd[1], "th_calib"))

            elif op == "query":
                if len(cmd) < 2:
                    print("usage: query <place_id>")
                    continue
                pprint(set_mode(cmd[1], "query"))

            elif op == "del":
                if len(cmd) < 2:
                    print("usage: del <place_id>")
                    continue
                pprint(delete_place(cmd[1]))

            elif op == "delall":
                pprint(delete_all_places())

            elif op == "delth":
                if len(cmd) < 2:
                    print("usage: delth <place_id>")
                    continue
                pprint(delete_threshold(cmd[1]))

            elif op == "recalib":
                if len(cmd) < 2:
                    print("usage: recalib <place_id>")
                    continue
                pprint(recalibrate_place(cmd[1]))

            elif op == "recaliball":
                pprint(recalibrate_all())

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