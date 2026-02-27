# db.py (운영형: UUID + CHECK + CASCADE + 인덱스)
import sqlite3
import uuid
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
        -- 인덱스 (GUI/조회 성능)
        -- =========================
        CREATE INDEX IF NOT EXISTS idx_events_place_time
        ON events(place_id, captured_at);

        CREATE INDEX IF NOT EXISTS idx_events_flag_time
        ON events(anomaly_flag, captured_at);

        CREATE INDEX IF NOT EXISTS idx_frames_event_idx
        ON frames(event_id, idx);
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