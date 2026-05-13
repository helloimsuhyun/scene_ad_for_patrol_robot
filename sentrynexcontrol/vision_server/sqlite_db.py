#sqlite_dl.py

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Union, Dict, Any

ISO8601 = str
DEMO_MODE = True

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
    
    # ----------------------------------------------------
    # 자동 마이그레이션 (schema update)
    # ----------------------------------------------------
    
    # 1. events 테이블 마이그레이션
    try:
        cur.execute("SELECT admin_checked FROM events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE events ADD COLUMN admin_checked INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE events ADD COLUMN admin_label TEXT")
        except:
            pass

    try:
        cur.execute("SELECT verified_change_image_path FROM events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE events ADD COLUMN verified_change_image_path TEXT")
        except:
            pass

    # 2. places 테이블 마이그레이션
    try:
        cur.execute("SELECT is_active FROM places LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE places ADD COLUMN display_name TEXT")
            cur.execute("ALTER TABLE places ADD COLUMN x REAL")
            cur.execute("ALTER TABLE places ADD COLUMN y REAL")
            cur.execute("ALTER TABLE places ADD COLUMN yaw REAL")
            cur.execute("ALTER TABLE places ADD COLUMN patrol_enabled INTEGER NOT NULL DEFAULT 1")
            cur.execute("ALTER TABLE places ADD COLUMN patrol_order INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE places ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
        except:
            pass
    
    try:
        cur.execute("SELECT place_type FROM places LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("""
                ALTER TABLE places
                ADD COLUMN place_type TEXT NOT NULL DEFAULT 'capture'
            """)
        except:
            pass
    # 3. yolo_events 테이블 마이그레이션
    try:
        cur.execute("SELECT event_type FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN event_type TEXT")
        except:
            pass

    try:
        cur.execute("SELECT dwell_time_sec FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN dwell_time_sec REAL")
        except:
            pass

    try:
        cur.execute("SELECT source_region_id FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN source_region_id INTEGER")
        except:
            pass

    try:
        cur.execute("SELECT source_region_name FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN source_region_name TEXT")
        except:
            pass

    try:
        cur.execute("SELECT admin_checked FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN admin_checked INTEGER NOT NULL DEFAULT 0")
        except:
            pass

    try:
        cur.execute("SELECT admin_label FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN admin_label TEXT")
        except:
            pass
    try:
        cur.execute("SELECT tracking_person_id FROM yolo_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE yolo_events ADD COLUMN tracking_person_id TEXT")
        except:
            pass
    
    # 4. employees / auth_events 테이블 마이그레이션
    try:
        cur.execute("SELECT employee_id FROM employees LIMIT 1")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("SELECT auth_event_id FROM auth_events LIMIT 1")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("SELECT x FROM auth_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE auth_events ADD COLUMN x REAL")
            cur.execute("ALTER TABLE auth_events ADD COLUMN y REAL")
            cur.execute("ALTER TABLE auth_events ADD COLUMN yaw REAL")
        except:
            pass
    try:
        cur.execute("SELECT admin_checked FROM auth_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE auth_events ADD COLUMN admin_checked INTEGER NOT NULL DEFAULT 0")
            cur.execute("ALTER TABLE auth_events ADD COLUMN admin_label TEXT")
        except:
            pass

    try:
        cur.execute("SELECT source_region_id FROM audio_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE audio_events ADD COLUMN source_region_id INTEGER")
        except sqlite3.OperationalError:
            pass

    try:
        cur.execute("SELECT source_region_name FROM audio_events LIMIT 1")
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE audio_events ADD COLUMN source_region_name TEXT")
        except sqlite3.OperationalError:
            pass

    cur.executescript(
        """
        -- =========================
        -- events: 1행 = 이벤트 1건
        -- =========================
        CREATE TABLE IF NOT EXISTS events (
        event_id        TEXT    PRIMARY KEY,
        place_id        TEXT    NOT NULL,
        captured_at     TEXT    NOT NULL,
        anomaly_flag    INTEGER NOT NULL CHECK (anomaly_flag IN (0,1)),

        anomaly_score   REAL,
        threshold_used  REAL,
        ref_bank_id     TEXT,
        ref_topk_json   TEXT,
        summary_text    TEXT,
        verified_change_image_path TEXT,

        admin_checked   INTEGER NOT NULL DEFAULT 0,
        admin_label     TEXT,

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

            place_type TEXT NOT NULL DEFAULT 'capture',

            mode               TEXT NOT NULL DEFAULT 'idle'
                                CHECK (mode IN ('idle', 'bank', 'th_calib', 'query')),
            need_calibration   INTEGER NOT NULL DEFAULT 1 CHECK (need_calibration IN (0,1)),
            updated_at         TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- =========================
        -- audio_events: 오디오 이벤트 관리
        -- =========================
        CREATE TABLE IF NOT EXISTS audio_events (
            audio_event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            audio_path TEXT NOT NULL,

            x REAL,
            y REAL,
            yaw REAL,

            doa REAL,
            model_label TEXT,

            source_region_id INTEGER,
            source_region_name TEXT,

            admin_checked INTEGER NOT NULL DEFAULT 0,
            admin_label TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- =========================
        -- yolo_events
        -- =========================
        CREATE TABLE IF NOT EXISTS yolo_events (
            yolo_event_id TEXT PRIMARY KEY,
            tracking_person_id TEXT,
            timestamp TEXT NOT NULL,
            image_path TEXT,

            x REAL,
            y REAL,
            yaw REAL,

            person_count INTEGER NOT NULL DEFAULT 0,
            event_type TEXT,

            source_region_id INTEGER,
            source_region_name TEXT,

            dwell_time_sec REAL,


            admin_checked INTEGER NOT NULL DEFAULT 0,
            admin_label TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- =========================
        -- yolo 구역
        -- =========================

        CREATE TABLE IF NOT EXISTS yolo_regions (
            region_id     INTEGER PRIMARY KEY AUTOINCREMENT,

            name          TEXT NOT NULL,

            x_min         REAL NOT NULL,
            x_max         REAL NOT NULL,
            y_min         REAL NOT NULL,
            y_max         REAL NOT NULL,

            is_enabled    INTEGER NOT NULL DEFAULT 1 CHECK (is_enabled IN (0,1)),

            updated_at    TEXT NOT NULL
        );


        -- =========================
        -- patrol_presets: 순찰 루트 프리셋
        -- =========================
        CREATE TABLE IF NOT EXISTS patrol_presets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            routes TEXT NOT NULL, -- JSON formatted list of place_ids
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- =========================
        -- patrol_schedules: 타임라인 자동 스케줄
        -- =========================
        CREATE TABLE IF NOT EXISTS patrol_schedules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preset_id INTEGER NOT NULL,
            time_str TEXT NOT NULL, -- "HH:MM"
            is_active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (preset_id) REFERENCES patrol_presets(id) ON DELETE CASCADE
        );

        -- =========================
        -- employees: RFID -> 직원 매칭
        -- =========================
        CREATE TABLE IF NOT EXISTS employees (
            employee_id      TEXT PRIMARY KEY,
            name             TEXT NOT NULL,
            rfid_uid         TEXT UNIQUE NOT NULL,
            is_active        INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
            created_at       TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- =========================
        -- auth_events: 인증 이벤트
        -- =========================
        CREATE TABLE IF NOT EXISTS auth_events (
            auth_event_id        TEXT PRIMARY KEY,
            tracking_person_id   TEXT,
            yolo_event_id        TEXT,
            employee_id          TEXT,
            timestamp            TEXT NOT NULL,
            status               TEXT NOT NULL,
            rfid_uid             TEXT,
            employee_name        TEXT,
            result_message       TEXT,
            image_path           TEXT,
            source_region_id     INTEGER,
            source_region_name   TEXT,
            x REAL,
            y REAL,
            yaw REAL,

            admin_checked        INTEGER NOT NULL DEFAULT 0,
            admin_label          TEXT,

            created_at           TEXT NOT NULL DEFAULT (datetime('now'))
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


        CREATE INDEX IF NOT EXISTS idx_audio_events_time
        ON audio_events(timestamp);

        CREATE INDEX IF NOT EXISTS idx_audio_events_admin_checked
        ON audio_events(admin_checked);

        CREATE INDEX IF NOT EXISTS idx_audio_events_model_label
        ON audio_events(model_label);

        CREATE INDEX IF NOT EXISTS idx_yolo_events_time
        ON yolo_events(timestamp);

        CREATE INDEX IF NOT EXISTS idx_yolo_regions_enabled
        ON yolo_regions(is_enabled);

        CREATE INDEX IF NOT EXISTS idx_employees_rfid_uid
        ON employees(rfid_uid);

        CREATE INDEX IF NOT EXISTS idx_auth_events_time
        ON auth_events(timestamp);

        CREATE INDEX IF NOT EXISTS idx_auth_events_status
        ON auth_events(status);

        CREATE INDEX IF NOT EXISTS idx_auth_events_tracking_person_id
        ON auth_events(tracking_person_id);

        CREATE INDEX IF NOT EXISTS idx_auth_events_admin_checked
        ON auth_events(admin_checked);
        """
    )
    db.commit()
    if DEMO_MODE:
        seed_demo_employees(db)


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

def update_event_verified_change_image(
    db: sqlite3.Connection,
    event_id: str,
    image_path: Optional[str],
) -> None:
    """
    events 행에 verified change overlay 이미지 경로 저장.
    image_path는 SAVE_ROOT 기준 상대 경로를 넣는 것을 권장.
    예: P004/query/_verified_change/xxx_verified_change_overlay.png
    """
    cur = db.cursor()
    cur.execute(
        """
        UPDATE events
        SET verified_change_image_path = ?
        WHERE event_id = ?
        """,
        (image_path, event_id),
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

# 새 장소 추가 (프론트 맵 클릭 생성용)
def create_place(db, place_id: str, display_name: str, x: float, y: float, yaw: float, patrol_enabled: int = 0):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute("""
        INSERT INTO places
        (place_id, display_name, x, y, yaw, patrol_enabled, is_active, mode, need_calibration, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, 1, 'idle', 1, ?)
    """, (str(place_id), display_name, x, y, yaw, patrol_enabled, now))
    db.commit()
    return get_place(db, place_id)


# 특정 place 상태 조회 
def get_place(db, place_id: str):
    cur = db.cursor()
    cur.execute("""
        SELECT place_id, display_name, x, y, yaw,
            patrol_enabled, patrol_order, is_active,
            place_type,
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
                place_type,
                mode, need_calibration, updated_at
            FROM places
            WHERE is_active = 1
            ORDER BY patrol_order ASC, place_id ASC
        """)
    else:
        cur.execute("""
            SELECT place_id, display_name, x, y, yaw,
                patrol_enabled, patrol_order, is_active,
                place_type,
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
        SET admin_checked = 1,
            admin_label = ?
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
    cur = db.cursor()
    cur.execute("""
        SELECT place_id, display_name, x, y, yaw, patrol_order, place_type
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

# ======================================= 오디오 관련 함수

# 오디오 이벤트 db에 넣는 함수
def insert_audio_event(
    db: sqlite3.Connection,
    timestamp: ISO8601,
    audio_path: str,
    x: Optional[float] = None,
    y: Optional[float] = None,
    yaw: Optional[float] = None,
    doa: Optional[float] = None,
    model_label: Optional[str] = None,
    audio_event_id: Optional[str] = None,
    source_region_id=None,
    source_region_name=None,
) -> str:
    aid = audio_event_id or _uuid()
    cur = db.cursor()
    cur.execute(
        """
        INSERT INTO audio_events
        (audio_event_id, timestamp, audio_path, x, y, yaw, doa, model_label,
        source_region_id, source_region_name)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aid, timestamp, audio_path,
            x, y, yaw, doa, model_label,
            source_region_id, source_region_name
        ),
    )
    db.commit()
    return aid

# 관리자 오디오 이벤트 라벨링
def set_audio_event_admin_label(
    db: sqlite3.Connection,
    audio_event_id: str,
    admin_label: Optional[str],
) -> None:
    cur = db.cursor()
    cur.execute(
        """
        UPDATE audio_events
        SET admin_checked = 1,
            admin_label = ?
        WHERE audio_event_id = ?
        """,
        (admin_label, audio_event_id),
    )
    db.commit()

#오디오 이벤트 조회 함수 (id로 조회)
def get_audio_event(db: sqlite3.Connection, audio_event_id: str) -> Optional[Dict[str, Any]]:
    cur = db.cursor()
    row = cur.execute(
        "SELECT * FROM audio_events WHERE audio_event_id = ?",
        (audio_event_id,),
    ).fetchone()
    return dict(row) if row else None

#오디오 이벤트 조회 (list 형태로 여러개 / 체크 안된것만 할것지 설정 가능)
def list_audio_events(db: sqlite3.Connection, unchecked_only: bool = False) -> List[Dict[str, Any]]:
    cur = db.cursor()
    if unchecked_only:
        rows = cur.execute(
            """
            SELECT * FROM audio_events
            WHERE admin_checked = 0
            ORDER BY timestamp DESC
            """
        ).fetchall()
    else:
        rows = cur.execute(
            """
            SELECT * FROM audio_events
            ORDER BY timestamp DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]

# ======================================= 순찰 프리셋 및 스케줄 관련 함수

def create_preset(db: sqlite3.Connection, name: str, routes_json: str) -> int:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO patrol_presets (name, routes) VALUES (?, ?)",
        (name, routes_json)
    )
    db.commit()
    return cur.lastrowid

def update_preset(db: sqlite3.Connection, preset_id: int, name: str, routes_json: str) -> None:
    cur = db.cursor()
    cur.execute(
        "UPDATE patrol_presets SET name = ?, routes = ? WHERE id = ?",
        (name, routes_json, preset_id)
    )
    db.commit()

def get_preset(db: sqlite3.Connection, preset_id: int) -> Optional[Dict[str, Any]]:
    cur = db.cursor()
    row = cur.execute("SELECT * FROM patrol_presets WHERE id = ?", (preset_id,)).fetchone()
    return dict(row) if row else None

def list_presets(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = db.cursor()
    rows = cur.execute("SELECT * FROM patrol_presets ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]

def delete_preset(db: sqlite3.Connection, preset_id: int) -> None:
    cur = db.cursor()
    cur.execute("DELETE FROM patrol_presets WHERE id = ?", (preset_id,))
    db.commit()

def create_schedule(db: sqlite3.Connection, preset_id: int, time_str: str, is_active: int = 1) -> int:
    cur = db.cursor()
    cur.execute(
        "INSERT INTO patrol_schedules (preset_id, time_str, is_active) VALUES (?, ?, ?)",
        (preset_id, time_str, is_active)
    )
    db.commit()
    return cur.lastrowid

def update_schedule(db: sqlite3.Connection, schedule_id: int, preset_id: int, time_str: str, is_active: int) -> None:
    cur = db.cursor()
    cur.execute(
        "UPDATE patrol_schedules SET preset_id = ?, time_str = ?, is_active = ? WHERE id = ?",
        (preset_id, time_str, is_active, schedule_id)
    )
    db.commit()

def get_schedule(db: sqlite3.Connection, schedule_id: int) -> Optional[Dict[str, Any]]:
    cur = db.cursor()
    row = cur.execute("SELECT * FROM patrol_schedules WHERE id = ?", (schedule_id,)).fetchone()
    return dict(row) if row else None

def list_schedules(db: sqlite3.Connection) -> List[Dict[str, Any]]:
    cur = db.cursor()
    rows = cur.execute("SELECT * FROM patrol_schedules ORDER BY time_str ASC").fetchall()
    return [dict(r) for r in rows]

def delete_schedule(db: sqlite3.Connection, schedule_id: int) -> None:
    cur = db.cursor()
    cur.execute("DELETE FROM patrol_schedules WHERE id = ?", (schedule_id,))
    db.commit()


# ================= YOLO EVENT =====================================================================

def insert_yolo_event(
    db,
    timestamp,
    image_path=None,
    tracking_person_id=None,
    x=None,
    y=None,
    yaw=None,
    person_count=0,
    event_type="person_present",
    source_region_id=None,
    source_region_name=None,
    dwell_time_sec=None,
    yolo_event_id=None,
):
    yeid = yolo_event_id or _uuid()

    cur = db.cursor()
    cur.execute(
        """
        INSERT INTO yolo_events
        (yolo_event_id, tracking_person_id, timestamp, image_path,
        x, y, yaw,
        person_count, event_type,
        source_region_id, source_region_name, dwell_time_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            yeid,
            tracking_person_id,
            timestamp,
            image_path,
            x,
            y,
            yaw,
            int(person_count),
            event_type,          
            source_region_id,
            source_region_name,
            dwell_time_sec,      
        ),
    )
    db.commit()
    return yeid


def get_yolo_event(db, yolo_event_id):
    cur = db.cursor()
    row = cur.execute(
        "SELECT * FROM yolo_events WHERE yolo_event_id=?",
        (yolo_event_id,),
    ).fetchone()
    return dict(row) if row else None


def list_yolo_events(db, since=None, limit=50, unchecked_only=False):
    cur = db.cursor()

    if since is None:
        if unchecked_only:
            rows = cur.execute(
                "SELECT * FROM yolo_events WHERE admin_checked=0 ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM yolo_events ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
    else:
        if unchecked_only:
            rows = cur.execute(
                "SELECT * FROM yolo_events WHERE timestamp>? AND admin_checked=0 ORDER BY timestamp DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        else:
            rows = cur.execute(
                "SELECT * FROM yolo_events WHERE timestamp>? ORDER BY timestamp DESC LIMIT ?",
                (since, limit),
            ).fetchall()

    return [dict(r) for r in rows]


def set_yolo_event_admin_label(db, yolo_event_id, admin_label):
    cur = db.cursor()
    cur.execute(
        """
        UPDATE yolo_events
        SET admin_checked=1, admin_label=?
        WHERE yolo_event_id=?
        """,
        (admin_label, yolo_event_id),
    )
    db.commit()


# ============ YOLO 구역관리 , 제어  ======================================================



# yolo 구역 전체 조회 (enabled_only=True면 활성 구역만 반환)
def list_yolo_regions(db, enabled_only: bool = False):
    cur = db.cursor()
    if enabled_only:
        cur.execute(
            """
            SELECT * FROM yolo_regions
            WHERE is_enabled = 1
            ORDER BY region_id ASC
            """
        )
    else:
        cur.execute(
            """
            SELECT * FROM yolo_regions
            ORDER BY region_id ASC
            """
        )
    return cur.fetchall()


# 한개 구역 생성 (bbox + 이름 + on/off 상태 저장)
def insert_yolo_region(db, name, x_min, x_max, y_min, y_max, is_enabled=True):
    name = name.strip()
    if not name:
        raise ValueError("name empty")
    now = datetime.now().isoformat()

    x1 = min(x_min, x_max)
    x2 = max(x_min, x_max)
    y1 = min(y_min, y_max)
    y2 = max(y_min, y_max)

    cur = db.cursor()
    cur.execute("""
        INSERT INTO yolo_regions (
            name, x_min, x_max, y_min, y_max, is_enabled, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        name,
        x1, x2, y1, y2,   
        int(is_enabled),
        now,
    ))
    db.commit()
    return cur.lastrowid


# 한개 구역 조회 (region_id 기준)
def get_yolo_region(db, region_id: int):
    cur = db.cursor()
    cur.execute(
        "SELECT * FROM yolo_regions WHERE region_id = ?",
        (region_id,),
    )
    return cur.fetchone()


# 한개 구역 수정 (이름 + bbox 좌표 업데이트)
def update_yolo_region(db, region_id, name, x_min, x_max, y_min, y_max):
    name = name.strip()
    if not name:
        raise ValueError("name empty")
    now = datetime.now().isoformat()

    x1 = min(x_min, x_max)
    x2 = max(x_min, x_max)
    y1 = min(y_min, y_max)
    y2 = max(y_min, y_max)

    cur = db.cursor()
    cur.execute(
        """
        UPDATE yolo_regions
        SET name=?, x_min=?, x_max=?, y_min=?, y_max=?, updated_at=?
        WHERE region_id=?
        """,
        (name, x1, x2, y1, y2, now, region_id),
    )
    db.commit()
    return cur.rowcount


# 개별 구역 on/off (특정 region 활성화 상태 변경)
def set_yolo_region_enabled(db, region_id, is_enabled):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute(
        """
        UPDATE yolo_regions
        SET is_enabled=?, updated_at=?
        WHERE region_id=?
        """,
        (int(is_enabled), now, region_id),
    )
    db.commit()
    return cur.rowcount


# 전체 구역 on/off (모든 region 활성화 상태 일괄 변경)
def set_all_yolo_regions_enabled(db, is_enabled):
    now = datetime.now().isoformat()
    cur = db.cursor()
    cur.execute(
        """
        UPDATE yolo_regions
        SET is_enabled=?, updated_at=?
        """,
        (int(is_enabled), now),
    )
    db.commit()
    return cur.rowcount


# 한개 구역 삭제 (region_id 기준)
def delete_yolo_region(db, region_id):
    cur = db.cursor()
    cur.execute(
        "DELETE FROM yolo_regions WHERE region_id=?",
        (region_id,),
    )
    db.commit()
    return cur.rowcount


# 전체 구역 삭제 (모든 region 제거)
def delete_all_yolo_regions(db):
    cur = db.cursor()
    cur.execute("DELETE FROM yolo_regions")
    db.commit()
    return cur.rowcount


# 활성 구역 존재 여부 확인 (region mode에서 실행 판단용)
def has_enabled_yolo_region(db):
    cur = db.cursor()
    cur.execute(
        "SELECT 1 FROM yolo_regions WHERE is_enabled=1 LIMIT 1"
    )
    return cur.fetchone() is not None


# 활성 구역 개수 반환 (GUI/로봇 상태 표시용)
def count_enabled_yolo_regions(db):
    cur = db.cursor()
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM yolo_regions WHERE is_enabled=1"
    )
    row = cur.fetchone()
    return int(row["cnt"]) if row else 0

# ======================================================== 2차 인증 관련 DB 함수

# 직원 DB는 고정으로 하드코딩 (데모용)
def seed_demo_employees(db):
    cur = db.cursor()

    # 기존 값 제거
    cur.execute("DELETE FROM employees")

    # 등록 카드: 1,2,3만 사용
    rows = [
        ("E001", "Kim",  "49B73204", 1),
        ("E002", "Lee",  "AC9DE33D", 1),
        ("E003", "Park", "E9D9E33D", 1),
    ]

    cur.executemany(
        """
        INSERT INTO employees
        (employee_id, name, rfid_uid, is_active)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )

    db.commit()

def get_employee_by_rfid(db, rfid_uid):
    cur = db.cursor()
    row = cur.execute(
        """
        SELECT *
        FROM employees
        WHERE rfid_uid = ?
          AND is_active = 1
        """,
        (str(rfid_uid).strip().upper(),),
    ).fetchone()

    return dict(row) if row else None


def insert_auth_event(
    db,
    timestamp,
    tracking_person_id=None,
    yolo_event_id=None,
    status="waiting_rfid",
    source_region_id=None,
    source_region_name=None,
    x=None,
    y=None,
    yaw=None,
    auth_event_id=None,
):
    aeid = auth_event_id or _uuid()

    cur = db.cursor()
    cur.execute(
        """
        INSERT INTO auth_events
        (auth_event_id, tracking_person_id, yolo_event_id,
        timestamp, status,
        source_region_id, source_region_name,
        x, y, yaw)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            aeid,
            tracking_person_id,
            yolo_event_id,
            timestamp,
            status,
            source_region_id,
            source_region_name,
            x,
            y,
            yaw,
        ),
    )
    db.commit()
    return aeid

def update_auth_event_result(
    db,
    auth_event_id,
    status,
    employee_id=None,
    rfid_uid=None,
    employee_name=None,
    result_message=None,
    image_path=None,
):
    cur = db.cursor()
    cur.execute(
        """
        UPDATE auth_events
        SET employee_id=?,
            status=?,
            rfid_uid=?,
            employee_name=?,
            result_message=?,
            image_path = COALESCE(?, image_path)
        WHERE auth_event_id=?
        """,
        (
            employee_id,
            status,
            rfid_uid,
            employee_name,
            result_message,
            image_path,
            auth_event_id,
        ),
    )
    db.commit()

def set_auth_event_timeout(db, auth_event_id, image_path=None):
    cur = db.cursor()

    if image_path is None:
        cur.execute(
            """
            UPDATE auth_events
            SET status = 'timeout'
            WHERE auth_event_id = ?
            """,
            (auth_event_id,),
        )
    else:
        cur.execute(
            """
            UPDATE auth_events
            SET status = 'timeout',
                image_path = ?
            WHERE auth_event_id = ?
            """,
            (image_path, auth_event_id),
        )

    db.commit()

def set_auth_event_admin_label(db, auth_event_id, admin_label):
    cur = db.cursor()
    cur.execute(
        """
        UPDATE auth_events
        SET admin_checked = 1,
            admin_label = ?
        WHERE auth_event_id = ?
        """,
        (admin_label, auth_event_id),
    )
    db.commit()

def get_auth_event(db, auth_event_id):
    cur = db.cursor()
    row = cur.execute(
        """
        SELECT *
        FROM auth_events
        WHERE auth_event_id=?
        """,
        (auth_event_id,),
    ).fetchone()

    return dict(row) if row else None

def list_auth_events(
    db,
    since: Optional[str] = None,
    limit: int = 50,
    status: Optional[str] = None,
):
    cur = db.cursor()

    if since is None and status is None:
        rows = cur.execute(
            """
            SELECT *
            FROM auth_events
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    elif since is not None and status is None:
        rows = cur.execute(
            """
            SELECT *
            FROM auth_events
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (since, limit),
        ).fetchall()

    elif since is None and status is not None:
        rows = cur.execute(
            """
            SELECT *
            FROM auth_events
            WHERE status = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (status, limit),
        ).fetchall()

    else:
        rows = cur.execute(
            """
            SELECT *
            FROM auth_events
            WHERE timestamp > ?
              AND status = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (since, status, limit),
        ).fetchall()

    return [dict(r) for r in rows]

def get_latest_yolo_event(db, tracking_person_id=None):
    cur = db.cursor()

    if tracking_person_id is not None:
        row = cur.execute(
            """
            SELECT *
            FROM yolo_events
            WHERE tracking_person_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (str(tracking_person_id),),
        ).fetchone()

        if row:
            return dict(row)

    row = cur.execute(
        """
        SELECT *
        FROM yolo_events
        ORDER BY timestamp DESC
        LIMIT 1
        """
    ).fetchone()

    return dict(row) if row else None

# ================================================================ 경유점 추가 

def generate_unique_waypoint_id(db, prefix: str = "W") -> str:
    cur = db.cursor()

    rows = cur.execute(
        """
        SELECT place_id
        FROM places
        WHERE place_id LIKE ?
        """,
        (f"{prefix}%",),
    ).fetchall()

    max_num = 0

    for r in rows:
        pid = r["place_id"]
        tail = pid[len(prefix):]

        if tail.isdigit():
            num = int(tail)
            if num > max_num:
                max_num = num

    return f"{prefix}{max_num + 1:03d}"

# 사진을 찍지 않고 경우만 하는 place 위한 db 함수
def insert_gui_waypoint(
    db,
    place_id: str,
    x: float,
    y: float,
    yaw: float,
    display_name: Optional[str] = None,
    patrol_enabled: bool = True,
    patrol_order: Optional[int] = None,
):
    now = datetime.now().isoformat()

    if patrol_order is None:
        patrol_order = get_next_patrol_order(db)

    cur = db.cursor()
    cur.execute("""
        INSERT INTO places
        (place_id, display_name, x, y, yaw,
         patrol_enabled, patrol_order, is_active,
         place_type, mode, need_calibration, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 1,
                'waypoint', 'idle', 0, ?)
    """, (
        str(place_id),
        display_name or str(place_id),
        float(x),
        float(y),
        float(yaw),
        1 if patrol_enabled else 0,
        int(patrol_order),
        now,
    ))
    db.commit()