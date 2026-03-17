# sqlite_dl

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict, Any

ISO8601 = str

#random한 uuid 문자열 생성 - 고유 id
def _uuid() -> str:
    # 예: "550e8400-e29b-41d4-a716-446655440000"
    return str(uuid.uuid4())

#db열어서 connection객체 생성
def connect_db(db_path: Union[str, Path]) -> sqlite3.Connection:
    # check_same_thread = False > 멀티스레드 가능 (동시에 write하면 충돌 주의)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row

    # 중요: FK 활성화 안 하면 ON DELETE CASCADE가 "동작 안 함"
    con.execute("PRAGMA foreign_keys = ON;")

    # 운영 권장 옵션
    con.execute("PRAGMA journal_mode = WAL;")
    con.execute("PRAGMA synchronous = NORMAL;")
    con.execute("PRAGMA busy_timeout = 5000;")

    return con


# db생성, 이미 있으면 무시하고 없으면 만들기
def init_db(db: sqlite3.Connection) -> None:
    cur = db.cursor()
    cur.executescript(
        """
        -- =========================
        -- events: 1행 = 이벤트 1건
        -- =========================
        CREATE TABLE IF NOT EXISTS events (
          event_id        TEXT    PRIMARY KEY,          -- UUID 문자열
          place_id        TEXT    NOT NULL,
          captured_at     TEXT    NOT NULL,
          anomaly_flag    INTEGER NOT NULL CHECK (anomaly_flag IN (0,1)),

          anomaly_score   REAL,
          threshold_used  REAL,
          ref_bank_id     TEXT,
          ref_topk_json   TEXT,
          summary_text    TEXT,
          manual_label TEXT,

          created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        -- =========================
        -- frames: 1행 = 이미지 1장
        -- =========================
        CREATE TABLE IF NOT EXISTS frames (
          -- 필수
          frame_id      TEXT    PRIMARY KEY,            -- UUID 문자열
          event_id      TEXT    NOT NULL,               -- FK
          idx           INTEGER NOT NULL,
          image_path    TEXT    NOT NULL,

          -- 옵션
          frame_score   REAL,
          capture_time  TEXT,

          FOREIGN KEY (event_id) REFERENCES events(event_id)
            ON DELETE CASCADE
        );

        -- =========================
        -- places: 장소 상태 관리
        -- =========================
        CREATE TABLE IF NOT EXISTS places (
            place_id           TEXT PRIMARY KEY,

            display_name       TEXT,
            x                  REAL,
            y                  REAL,
            yaw                REAL,

            patrol_enabled     INTEGER NOT NULL DEFAULT 1 CHECK (patrol_enabled IN (0,1)),
            patrol_order       INTEGER NOT NULL DEFAULT 0,
            is_active          INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),

            mode               TEXT NOT NULL DEFAULT 'idle'
                                CHECK (mode IN ('idle', 'bank', 'th_calib', 'query')),
            need_calibration   INTEGER NOT NULL DEFAULT 1 CHECK (need_calibration IN (0,1)),
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- =========================
        -- 인덱스 (GUI/조회 성능)
        -- =========================
        CREATE INDEX IF NOT EXISTS idx_events_place_time
        ON events(place_id, captured_at);

        CREATE INDEX IF NOT EXISTS idx_events_flag_time
        ON events(anomaly_flag, captured_at);

        CREATE INDEX IF NOT EXISTS idx_frames_event_idx
        ON frames(event_id, idx);

        CREATE INDEX IF NOT EXISTS idx_places_mode
        ON places(mode);

        CREATE INDEX IF NOT EXISTS idx_places_need_calibration
        ON places(need_calibration);

        CREATE INDEX IF NOT EXISTS idx_places_active
        ON places(is_active);

        CREATE INDEX IF NOT EXISTS idx_places_patrol_enabled
        ON places(patrol_enabled);

        CREATE INDEX IF NOT EXISTS idx_places_patrol_order
        ON places(patrol_order);
        """
    )
    db.commit()


# -------------------------
# Insert (새로운 행 만들기)

def insert_event(
    db: sqlite3.Connection,
    place_id: str,
    captured_at: ISO8601,
    anomaly_flag: int = 0,  
    event_id: Optional[str] = None,
    anomaly_score: Optional[float] = None,
    threshold_used: Optional[float] = None,
    ref_bank_id: Optional[str] = None,
    ref_topk_json: Optional[str] = None,
    summary_text: Optional[str] = None,
) -> str:
    
    # events 1행 추가. event_id(UUID TEXT)를 return 처음 evnet 생긴 경우에는 해당 id를 바로 프레임 insert에 넣어줘야 !
    
    eid = event_id or _uuid()
    af = int(anomaly_flag)
    if af not in (0, 1):
        raise ValueError("anomaly_flag must be 0 or 1")

    cur = db.cursor()
    cur.execute(
        """
        INSERT INTO events
        (event_id, place_id, captured_at, anomaly_flag,
         anomaly_score, threshold_used, ref_bank_id, ref_topk_json, summary_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            eid,
            place_id,
            captured_at,
            af,
            anomaly_score,
            threshold_used,
            ref_bank_id,
            ref_topk_json,
            summary_text,
        ),
    )
    db.commit()
    return eid


def insert_frames(
    db: sqlite3.Connection,
    event_id: str,
    image_paths: List[str],
    frame_scores: Optional[Sequence[Optional[float]]] = None,
    capture_times: Optional[Union[ISO8601, Sequence[Optional[ISO8601]]]] = None,
    frame_ids: Optional[Sequence[str]] = None,
) -> List[str]:
    
    # frames N행 추가. 생성된 frame_id 리스트 반환.

    n = len(image_paths)

    if frame_scores is not None and len(frame_scores) != n:
        raise ValueError("frame_scores length must match image_paths length")

    if capture_times is None:
        ct_list = [None] * n
    elif isinstance(capture_times, str):
        ct_list = [capture_times] * n
    else:
        if len(capture_times) != n:
            raise ValueError("capture_times length must match image_paths length")
        ct_list = list(capture_times)

    if frame_ids is None: #안들어오면 _uuid로 만들어줌
        fids = [_uuid() for _ in range(n)]
    else:
        if len(frame_ids) != n:
            raise ValueError("frame_ids length must match image_paths length")
        fids = list(frame_ids)

    rows = []
    for idx, path in enumerate(image_paths):
        score = None if frame_scores is None else frame_scores[idx]
        rows.append((fids[idx], event_id, idx, path, score, ct_list[idx]))

    cur = db.cursor()
    cur.executemany(
        """
        INSERT INTO frames
        (frame_id, event_id, idx, image_path, frame_score, capture_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    db.commit()
    return fids


# -------------------------
# Update (모델 추론 결과 반영)

#frame 업데이트
def update_frame_scores(
    db: sqlite3.Connection,
    event_id: str,
    frame_scores: Sequence[float],
) -> None:
    """
    event_id에 속한 프레임들의 frame_score 업데이트
    받아오는 scores는 idx 순서와 (초기 받아온 json순서)같아야함

    """
    cur = db.cursor()
    cur.executemany(
        """
        UPDATE frames
        SET frame_score = ?
        WHERE event_id = ? AND idx = ?
        """,
        [(float(s), event_id, i) for i, s in enumerate(frame_scores)],
    )
    db.commit()

# 이벤트 업데이트
def update_event_result(
    db: sqlite3.Connection,
    event_id: str,
    anomaly_flag: int,
    anomaly_score: Optional[float] = None,
    threshold_used: Optional[float] = None,
    ref_bank_id: Optional[str] = None,
    ref_topk_json: Optional[str] = None,
    summary_text: Optional[str] = None,
) -> None:
    """
    events 행에 추론 결과 업데이트
    anomaly_flag는 반드시 0/1로 넣기.
    """
    af = int(anomaly_flag)
    if af not in (0, 1):
        raise ValueError("anomaly_flag must be 0 or 1")

    cur = db.cursor()
    cur.execute(
        """
        UPDATE events
        SET anomaly_flag = ?,
            anomaly_score = ?,
            threshold_used = ?,
            ref_bank_id = ?,
            ref_topk_json = ?,
            summary_text = ?
        WHERE event_id = ?
        """,
        (
            af,
            anomaly_score,
            threshold_used,
            ref_bank_id,
            ref_topk_json,
            summary_text,
            event_id,
        ),
    )
    db.commit()


# -------------------------
# (선택) 조회 유틸 - GUI 붙일 때 바로 유용
# -------------------------
def get_event(db: sqlite3.Connection, event_id: str) -> Optional[Dict[str, Any]]:
    cur = db.cursor()
    row = cur.execute("SELECT * FROM events WHERE event_id = ?", (event_id,)).fetchone()
    return dict(row) if row else None


def list_frames(db: sqlite3.Connection, event_id: str) -> List[Dict[str, Any]]:
    cur = db.cursor()
    rows = cur.execute(
        "SELECT * FROM frames WHERE event_id = ? ORDER BY idx ASC",
        (event_id,),
    ).fetchall()
    return [dict(r) for r in rows]


#-------------------- place 관련 함수

# 해당 place row가 없으면 기본 상태(idle)로 생성
def ensure_place(db, place_id: str, display_name: Optional[str] = None):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO places
        (place_id, display_name, patrol_enabled, is_active, mode, need_calibration, updated_at)
        VALUES (?, ?, 1, 1, 'idle', 1, ?)
    """, (str(place_id), display_name or str(place_id), now))
    db.commit()


# 특정 place 상태 조회 
def get_place(db, place_id: str):
    cur = db.cursor()
    cur.execute("""
        SELECT place_id, display_name, x, y, yaw,
            patrol_enabled, patrol_order, is_active,
            mode, need_calibration, updated_at
        FROM places
        WHERE place_id = ?
    """, (str(place_id),))
    row = cur.fetchone()
    return dict(row) if row else None


# 전체 place 상태 리스트 반환
def list_places(db, active_only: bool = True):
    cur = db.cursor()
    if active_only:
        cur.execute("""
            SELECT place_id, display_name, x, y, yaw,
                patrol_enabled, patrol_order, is_active,
                mode, need_calibration, updated_at
            FROM places
            WHERE is_active = 1
            ORDER BY patrol_order ASC, place_id ASC
        """)
    else:
        cur.execute("""
            SELECT place_id, display_name, x, y, yaw,
                patrol_enabled, patrol_order, is_active,
                mode, need_calibration, updated_at
            FROM places
            ORDER BY patrol_order ASC, place_id ASC
        """)
    rows = cur.fetchall()
    return [dict(r) for r in rows]


# place mode 변경 (idle / bank / th_calib / query)
def set_place_mode(db, place_id: str, mode: str):
    ensure_place(db, place_id)
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET mode = ?, updated_at = ?
        WHERE place_id = ?
    """, (mode, now, str(place_id)))
    db.commit()

#threshold 재계산 필요 여부 업데이트 함수
def set_place_need_calibration(db, place_id: str, need: bool):
    ensure_place(db, place_id)
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET need_calibration = ?, updated_at = ?
        WHERE place_id = ?
    """, (1 if need else 0, now, str(place_id)))
    db.commit()


# 특정 place 상태 row 삭제
def delete_place_row(db, place_id: str):
    cur = db.cursor()
    cur.execute("DELETE FROM places WHERE place_id = ?", (str(place_id),))
    db.commit()

# 모든 place 상태 row 삭제
def delete_all_place_rows(db):
    cur = db.cursor()
    cur.execute("DELETE FROM places")
    db.commit()

# ---------------------------------- 관리자 event 라벨링 관련 db 유틸

def set_event_manual_label(
    db: sqlite3.Connection,
    event_id: str,
    manual_label: Optional[str],
) -> None:
    cur = db.cursor()
    cur.execute(
        """
        UPDATE events
        SET manual_label = ?
        WHERE event_id = ?
        """,
        (manual_label, event_id),
    )
    db.commit()


# ---------------------------------- 교시 / patrol 관련 place db 유틸

# 새로 place가 들어오면 현재 마지막 active의 다음 순서로 초기 배정
def get_next_patrol_order(db: sqlite3.Connection) -> int:

    cur = db.cursor()
    row = cur.execute("""
        SELECT COALESCE(MAX(patrol_order), -1) + 1 AS next_order
        FROM places
        WHERE is_active = 1
    """).fetchone()
    return int(row["next_order"])


def set_place_patrol_order(db: sqlite3.Connection, place_id: str, patrol_order: int) -> None:
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET patrol_order = ?, updated_at = ?
        WHERE place_id = ?
    """, (int(patrol_order), now, str(place_id)))
    db.commit()


def reorder_patrol_places(db: sqlite3.Connection, ordered_place_ids: List[str]) -> None:
    """
    GUI에서 전달한 전체 place 순서대로 patrol_order를 0,1,2... 재부여
    주의: active place 전체 리스트를 넘긴다고 가정
    """
    now = datetime.now().isoformat()
    cur = db.cursor()

    existing_rows = cur.execute("""
        SELECT place_id
        FROM places
        WHERE is_active = 1
    """).fetchall()
    existing_ids = {row["place_id"] for row in existing_rows}

    req_ids = [str(pid) for pid in ordered_place_ids]
    missing = [pid for pid in req_ids if pid not in existing_ids]
    if missing:
        raise ValueError(f"unknown place_ids: {missing}")

    # active 전체를 보내는 정책 체크
    if set(req_ids) != existing_ids:
        raise ValueError("place_ids must contain all active places exactly once")

    if len(req_ids) != len(set(req_ids)):
        raise ValueError("duplicate place_id in reorder request")

    for idx, place_id in enumerate(req_ids):
        cur.execute("""
            UPDATE places
            SET patrol_order = ?, updated_at = ?
            WHERE place_id = ?
        """, (idx, now, place_id))

    db.commit()


def upsert_place_waypoint(
    db: sqlite3.Connection,
    place_id: str,
    x: float,
    y: float,
    yaw: float,
    display_name: Optional[str] = None,
    patrol_enabled: bool = True,
    patrol_order: int = 0,
) -> None:
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO places
        (place_id, display_name, x, y, yaw, patrol_enabled, patrol_order, is_active,
         mode, need_calibration, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'idle', 1, ?)
        ON CONFLICT(place_id) DO UPDATE SET
            display_name = COALESCE(excluded.display_name, places.display_name),
            x = excluded.x,
            y = excluded.y,
            yaw = excluded.yaw,
            patrol_enabled = excluded.patrol_enabled,
            patrol_order = excluded.patrol_order,
            is_active = 1,
            updated_at = excluded.updated_at
    """, (
        str(place_id),
        display_name if display_name is not None else str(place_id),
        float(x),
        float(y),
        float(yaw),
        1 if patrol_enabled else 0,
        int(patrol_order),
        now,
    ))
    db.commit()


def set_place_display_name(db, place_id: str, display_name: str):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET display_name = ?, updated_at = ?
        WHERE place_id = ?
    """, (display_name, now, str(place_id)))
    db.commit()


def set_place_patrol_enabled(db, place_id: str, enabled: bool):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET patrol_enabled = ?, updated_at = ?
        WHERE place_id = ?
    """, (1 if enabled else 0, now, str(place_id)))
    db.commit()


def deactivate_place(db, place_id: str):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET is_active = 0, updated_at = ?
        WHERE place_id = ?
    """, (now, str(place_id)))
    db.commit()


def reactivate_place(db, place_id: str):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        UPDATE places
        SET is_active = 1, updated_at = ?
        WHERE place_id = ?
    """, (now, str(place_id)))
    db.commit()


def list_patrol_places(db):
    """
    현재 순찰 waypoint로 지정된 place들을 patrol_order 기준으로 반환
    """
    cur = db.cursor()
    cur.execute("""
        SELECT place_id, display_name, x, y, yaw, patrol_order
        FROM places
        WHERE is_active = 1
          AND patrol_enabled = 1
          AND x IS NOT NULL
          AND y IS NOT NULL
          AND yaw IS NOT NULL
        ORDER BY patrol_order ASC, place_id ASC
    """)
    rows = cur.fetchall()
    return [dict(r) for r in rows]
