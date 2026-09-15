"""SQLite-backed short- and long-term memory for the local companion app."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY_DB = PROJECT_ROOT / "backend" / "data" / "emo_agent_memory.sqlite3"
ALLOWED_ROLES = {"user", "assistant"}
ALLOWED_MEMORY_KINDS = {"emotion_event", "trigger", "preference", "summary"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _evidence_text(value: Any) -> str:
    if isinstance(value, str):
        return _clip(value, 240)
    if isinstance(value, dict):
        for key in ("keyword", "text", "reason", "value"):
            if value.get(key):
                return _clip(value[key], 240)
    return ""


class MemoryService:
    """Keep session context and durable, structured emotion memories in SQLite."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            if self.database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()

    @classmethod
    def from_env(cls) -> "MemoryService":
        configured = os.environ.get("EMO_AGENT_MEMORY_DB")
        return cls(Path(configured) if configured else DEFAULT_MEMORY_DB)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT '新会话',
                current_emotion TEXT,
                current_confidence REAL,
                latest_analysis_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions(user_id, updated_at DESC);

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                request_id TEXT,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                emotion TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_request_role
                ON messages(session_id, request_id, role) WHERE request_id IS NOT NULL;

            CREATE TABLE IF NOT EXISTS tool_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
                request_id TEXT,
                name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_tools_session_id
                ON tool_calls(session_id, id);

            CREATE TABLE IF NOT EXISTS emotion_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT REFERENCES sessions(session_id) ON DELETE SET NULL,
                source TEXT NOT NULL,
                emotion TEXT,
                confidence REAL,
                summary TEXT NOT NULL,
                triggers_json TEXT NOT NULL DEFAULT '[]',
                dedupe_key TEXT UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_user_created
                ON emotion_events(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                source TEXT NOT NULL,
                confirmed INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, preference_key)
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                session_id TEXT PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
                user_id TEXT NOT NULL,
                summary_json TEXT NOT NULL DEFAULT '{}',
                summarized_through_message_id INTEGER NOT NULL DEFAULT 0,
                source_message_count INTEGER NOT NULL DEFAULT 0,
                summary_version INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL DEFAULT '',
                cooldown_until TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_session_summaries_user_updated
                ON session_summaries(user_id, updated_at DESC);
            """
        )
        self._connection.commit()

    @staticmethod
    def _identity(value: Any, default: str) -> str:
        clean = _clip(value, 128)
        return clean or default

    def ensure_session(
        self,
        user_id: str = "local-user",
        session_id: str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        user_id = self._identity(user_id, "local-user")
        session_id = self._identity(session_id, uuid.uuid4().hex)
        now = _now()
        with self._lock:
            existing = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing and existing["user_id"] != user_id:
                raise ValueError("session does not belong to this user")
            if not existing:
                self._connection.execute(
                    """INSERT INTO sessions
                       (session_id, user_id, title, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, user_id, _clip(title, 80) or "新会话", now, now),
                )
                self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return self._session_dict(row)

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
                (self._identity(session_id, ""), self._identity(user_id, "local-user")),
            ).fetchone()
        return self._session_dict(row) if row else None

    def list_sessions(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        user_id = self._identity(user_id, "local-user")
        limit = max(1, min(int(limit), 100))
        with self._lock:
            rows = self._connection.execute(
                """SELECT s.*,
                          (SELECT content FROM messages m
                           WHERE m.session_id = s.session_id
                           ORDER BY m.id DESC LIMIT 1) AS preview,
                          (SELECT COUNT(*) FROM messages m
                           WHERE m.session_id = s.session_id) AS message_count
                   FROM sessions s
                   WHERE s.user_id = ?
                   ORDER BY s.updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = self._session_dict(row)
            item["preview"] = _clip(row["preview"], 120)
            item["message_count"] = int(row["message_count"] or 0)
            result.append(item)
        return result

    def delete_session(self, user_id: str, session_id: str) -> bool:
        with self._lock:
            self._connection.execute(
                "DELETE FROM emotion_events WHERE session_id = ? AND user_id = ?",
                (self._identity(session_id, ""), self._identity(user_id, "local-user")),
            )
            cursor = self._connection.execute(
                "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
                (self._identity(session_id, ""), self._identity(user_id, "local-user")),
            )
            self._connection.commit()
        return cursor.rowcount > 0

    def delete_user_memory(self, user_id: str) -> dict[str, int]:
        user_id = self._identity(user_id, "local-user")
        with self._lock:
            event_count = self._connection.execute(
                "DELETE FROM emotion_events WHERE user_id = ?", (user_id,)
            ).rowcount
            preference_count = self._connection.execute(
                "DELETE FROM preferences WHERE user_id = ?", (user_id,)
            ).rowcount
            session_count = self._connection.execute(
                "DELETE FROM sessions WHERE user_id = ?", (user_id,)
            ).rowcount
            self._connection.commit()
        return {
            "sessions": session_count,
            "events": event_count,
            "preferences": preference_count,
        }

    def append_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        *,
        emotion: str | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any] | None:
        if role not in ALLOWED_ROLES:
            raise ValueError("role must be user or assistant")
        content = _clip(content, 12000)
        if not content:
            return None
        session = self.ensure_session(user_id, session_id)
        request_id = _clip(request_id, 128) or None
        emotion = _clip(emotion, 40) or None
        now = _now()
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """INSERT INTO messages
                       (session_id, request_id, role, content, emotion, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session["session_id"], request_id, role, content, emotion, now),
                )
            except sqlite3.IntegrityError:
                return None
            if role == "user" and session["title"] == "新会话":
                self._connection.execute(
                    "UPDATE sessions SET title = ? WHERE session_id = ?",
                    (_clip(content.replace("\n", " "), 28), session["session_id"]),
                )
            self._connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session["session_id"]),
            )
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._message_dict(row)

    def seed_history(
        self,
        user_id: str,
        session_id: str,
        history: Iterable[dict[str, str]],
    ) -> None:
        if self.get_messages(user_id, session_id, limit=1):
            return
        for item in list(history)[-20:]:
            role = item.get("role")
            if role in ALLOWED_ROLES:
                self.append_message(user_id, session_id, role, item.get("content") or "")

    def get_messages(self, user_id: str, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        session = self.get_session(user_id, session_id)
        if not session:
            return []
        limit = max(1, min(int(limit), 200))
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM (
                       SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?
                   ) ORDER BY id""",
                (session_id, limit),
            ).fetchall()
        return [self._message_dict(row) for row in rows]

    def get_history(self, user_id: str, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        return [
            {"role": item["role"], "content": item["content"]}
            for item in self.get_messages(user_id, session_id, limit=limit)
        ]

    def get_messages_after(
        self,
        user_id: str,
        session_id: str,
        after_message_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        if not self.get_session(user_id, session_id):
            return []
        limit = max(1, min(int(limit), 2000))
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM messages
                   WHERE session_id = ? AND id > ?
                   ORDER BY id LIMIT ?""",
                (session_id, max(0, int(after_message_id)), limit),
            ).fetchall()
        return [self._message_dict(row) for row in rows]

    def get_session_summary(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        if not self.get_session(user_id, session_id):
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM session_summaries WHERE session_id = ? AND user_id = ?",
                (session_id, self._identity(user_id, "local-user")),
            ).fetchone()
        if not row:
            return None
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "summary": _loads(row["summary_json"], {}),
            "summarized_through_message_id": row["summarized_through_message_id"],
            "source_message_count": row["source_message_count"],
            "summary_version": row["summary_version"],
            "model": row["model"],
            "cooldown_until": row["cooldown_until"],
            "updated_at": row["updated_at"],
        }

    def save_session_summary(
        self,
        user_id: str,
        session_id: str,
        summary: dict[str, Any],
        summarized_through_message_id: int,
        source_message_count: int,
        model: str,
        expected_version: int,
    ) -> bool:
        user_id = self._identity(user_id, "local-user")
        if not self.get_session(user_id, session_id):
            return False
        now = _now()
        with self._lock:
            row = self._connection.execute(
                "SELECT summary_version FROM session_summaries WHERE session_id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            current_version = int(row["summary_version"]) if row else 0
            if current_version != int(expected_version):
                return False
            self._connection.execute(
                """INSERT INTO session_summaries
                   (session_id, user_id, summary_json, summarized_through_message_id,
                    source_message_count, summary_version, model, cooldown_until, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       summary_json = excluded.summary_json,
                       summarized_through_message_id = excluded.summarized_through_message_id,
                       source_message_count = excluded.source_message_count,
                       summary_version = excluded.summary_version,
                       model = excluded.model,
                       cooldown_until = NULL,
                       updated_at = excluded.updated_at
                   WHERE session_summaries.user_id = excluded.user_id""",
                (
                    session_id,
                    user_id,
                    _json(summary),
                    int(summarized_through_message_id),
                    int(source_message_count),
                    current_version + 1,
                    _clip(model, 120),
                    now,
                    now,
                ),
            )
            self._connection.commit()
        return True

    def set_summary_cooldown(self, user_id: str, session_id: str, cooldown_until: str) -> None:
        user_id = self._identity(user_id, "local-user")
        self.ensure_session(user_id, session_id)
        now = _now()
        with self._lock:
            self._connection.execute(
                """INSERT INTO session_summaries
                   (session_id, user_id, cooldown_until, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                       cooldown_until = excluded.cooldown_until,
                       updated_at = excluded.updated_at
                   WHERE session_summaries.user_id = excluded.user_id""",
                (session_id, user_id, cooldown_until, now, now),
            )
            self._connection.commit()

    def update_session_analysis(
        self,
        user_id: str,
        session_id: str,
        analysis: dict[str, Any] | None,
        *,
        source: str,
        dedupe_key: str | None = None,
        record_event: bool = True,
    ) -> dict[str, Any] | None:
        analysis = analysis or {}
        emotion = _clip(analysis.get("fusion_prediction"), 40)
        if not emotion:
            return None
        confidence = analysis.get("fusion_confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        keywords = [text for text in (_evidence_text(item) for item in analysis.get("keywords") or []) if text][:12]
        reasons = [text for text in (_evidence_text(item) for item in analysis.get("possible_reasons") or []) if text][:8]
        transcript = _clip(analysis.get("asr_text"), 1000)
        summary = transcript or (reasons[0] if reasons else f"一次真实语音分析的主要情绪为 {emotion}")
        session = self.ensure_session(user_id, session_id)
        user_id = session["user_id"]
        session_id = session["session_id"]
        now = _now()
        safe_analysis = {
            "asr_text": transcript,
            "fusion_prediction": emotion,
            "fusion_confidence": confidence,
            "keywords": keywords,
            "possible_reasons": reasons,
            "emotion_curve": (analysis.get("emotion_curve") or [])[-40:],
            "emotion_change_points": (analysis.get("emotion_change_points") or [])[-20:],
        }
        with self._lock:
            self._connection.execute(
                """UPDATE sessions
                   SET current_emotion = ?, current_confidence = ?,
                       latest_analysis_json = ?, updated_at = ?
                   WHERE session_id = ? AND user_id = ?""",
                (emotion, confidence, _json(safe_analysis), now, session_id, user_id),
            )
            if not record_event:
                self._connection.commit()
                return None
            try:
                cursor = self._connection.execute(
                    """INSERT INTO emotion_events
                       (user_id, session_id, source, emotion, confidence, summary,
                        triggers_json, dedupe_key, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        session_id,
                        _clip(source, 60) or "audio_analysis",
                        emotion,
                        confidence,
                        summary,
                        _json(keywords),
                        _clip(dedupe_key, 180) or None,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                self._connection.commit()
                return None
            self._connection.commit()
            row = self._connection.execute(
                "SELECT * FROM emotion_events WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        return self._event_dict(row)

    def record_audio_turn(
        self,
        user_id: str,
        session_id: str | None,
        analysis: dict[str, Any],
        assistant_response: str,
        *,
        request_id: str | None = None,
    ) -> str:
        session = self.ensure_session(user_id, session_id)
        session_id = session["session_id"]
        request_id = _clip(request_id, 128) or uuid.uuid4().hex
        emotion = _clip(analysis.get("fusion_prediction"), 40) or None
        transcript = _clip(analysis.get("asr_text"), 12000) or "我上传了一段语音。"
        self.append_message(user_id, session_id, "user", transcript, emotion=emotion, request_id=f"{request_id}:audio")
        self.append_message(
            user_id,
            session_id,
            "assistant",
            assistant_response,
            emotion=emotion,
            request_id=f"{request_id}:audio",
        )
        self.update_session_analysis(
            user_id,
            session_id,
            analysis,
            source="audio_analysis",
            dedupe_key=f"audio:{session_id}:{request_id}",
        )
        return session_id

    def record_tool_calls(
        self,
        user_id: str,
        session_id: str,
        request_id: str,
        calls: Iterable[dict[str, Any]],
    ) -> None:
        self.ensure_session(user_id, session_id)
        now = _now()
        with self._lock:
            for call in calls:
                self._connection.execute(
                    """INSERT INTO tool_calls
                       (session_id, request_id, name, arguments_json, result_json, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        _clip(request_id, 128) or None,
                        _clip(call.get("name"), 80) or "unknown",
                        _json(call.get("arguments") or {}),
                        _json(call.get("result") or {}),
                        now,
                    ),
                )
            self._connection.commit()

    def write_memory(
        self,
        user_id: str,
        session_id: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        user_id = self._identity(user_id, "local-user")
        kind = _clip(payload.get("kind"), 40)
        if kind not in ALLOWED_MEMORY_KINDS:
            return {"ok": False, "available": True, "error": "invalid_memory_kind"}
        if session_id:
            self.ensure_session(user_id, session_id)
        if kind == "preference":
            if payload.get("confirmed") is not True:
                return {"ok": False, "available": True, "error": "preference_requires_confirmation"}
            key = _clip(payload.get("key"), 80)
            value = _clip(payload.get("value") or payload.get("content"), 500)
            if not key or not value:
                return {"ok": False, "available": True, "error": "preference_key_and_value_required"}
            now = _now()
            with self._lock:
                self._connection.execute(
                    """INSERT INTO preferences
                       (user_id, preference_key, preference_value, source, confirmed, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 1, ?, ?)
                       ON CONFLICT(user_id, preference_key) DO UPDATE SET
                           preference_value = excluded.preference_value,
                           source = excluded.source,
                           confirmed = 1,
                           updated_at = excluded.updated_at""",
                    (user_id, key, value, _clip(payload.get("source"), 60) or "user_confirmed", now, now),
                )
                self._connection.commit()
            return {"ok": True, "available": True, "kind": kind, "key": key, "value": value}

        content = _clip(payload.get("content") or payload.get("summary"), 1000)
        if not content:
            return {"ok": False, "available": True, "error": "memory_content_required"}
        triggers = payload.get("triggers") or ([content] if kind == "trigger" else [])
        triggers = [_clip(item, 240) for item in triggers if _clip(item, 240)][:12]
        confidence = payload.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None
        now = _now()
        with self._lock:
            cursor = self._connection.execute(
                """INSERT INTO emotion_events
                   (user_id, session_id, source, emotion, confidence, summary, triggers_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    session_id,
                    _clip(payload.get("source"), 60) or "memory_write_tool",
                    _clip(payload.get("emotion"), 40) or None,
                    confidence,
                    content,
                    _json(triggers),
                    now,
                ),
            )
            self._connection.commit()
        return {"ok": True, "available": True, "kind": kind, "event_id": cursor.lastrowid}

    def search(self, user_id: str, query: str, limit: int = 8) -> dict[str, Any]:
        user_id = self._identity(user_id, "local-user")
        query = _clip(query, 200)
        limit = max(1, min(int(limit), 20))
        pattern = f"%{query}%"
        with self._lock:
            if query:
                event_rows = self._connection.execute(
                    """SELECT * FROM emotion_events
                       WHERE user_id = ? AND (summary LIKE ? OR triggers_json LIKE ? OR emotion LIKE ?)
                       ORDER BY created_at DESC LIMIT ?""",
                    (user_id, pattern, pattern, pattern, limit),
                ).fetchall()
                preference_rows = self._connection.execute(
                    """SELECT * FROM preferences
                       WHERE user_id = ? AND confirmed = 1
                         AND (preference_key LIKE ? OR preference_value LIKE ?)
                       ORDER BY updated_at DESC LIMIT ?""",
                    (user_id, pattern, pattern, limit),
                ).fetchall()
            else:
                event_rows = self._connection.execute(
                    "SELECT * FROM emotion_events WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
                preference_rows = self._connection.execute(
                    "SELECT * FROM preferences WHERE user_id = ? AND confirmed = 1 ORDER BY updated_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
        events = [self._event_dict(row) for row in event_rows]
        preferences = [self._preference_dict(row) for row in preference_rows]
        return {
            "available": True,
            "query": query,
            "events": events,
            "preferences": preferences,
            "result_count": len(events) + len(preferences),
        }

    def emotion_trend(self, user_id: str, range_value: Any = "month") -> dict[str, Any]:
        user_id = self._identity(user_id, "local-user")
        days = self._range_days(range_value)
        threshold = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM emotion_events
                   WHERE user_id = ? AND created_at >= ? AND emotion IS NOT NULL
                   ORDER BY created_at""",
                (user_id, threshold),
            ).fetchall()
        events = [self._event_dict(row) for row in rows]
        emotion_counts = Counter(item["emotion"] for item in events if item.get("emotion"))
        trigger_counts = Counter(
            trigger for item in events for trigger in item.get("triggers", []) if trigger
        )
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in events:
            buckets[item["created_at"][:10]].append(item)
        points = []
        for day, items in sorted(buckets.items()):
            counts = Counter(item["emotion"] for item in items if item.get("emotion"))
            confidences = [item["confidence"] for item in items if isinstance(item.get("confidence"), (int, float))]
            points.append({
                "date": day,
                "dominant_emotion": counts.most_common(1)[0][0] if counts else None,
                "count": len(items),
                "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
            })
        return {
            "available": bool(events),
            "scope": "memory_trend",
            "range_days": days,
            "event_count": len(events),
            "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else None,
            "emotion_counts": dict(emotion_counts),
            "common_triggers": [
                {"text": text, "count": count} for text, count in trigger_counts.most_common(8)
            ],
            "points": points,
            "note": "长期趋势来自用户本地保存的结构化事件，不用于医疗或心理诊断。",
        }

    def summary(self, user_id: str, session_id: str | None = None) -> dict[str, Any]:
        user_id = self._identity(user_id, "local-user")
        session = self.get_session(user_id, session_id) if session_id else None
        rolling_summary = self.get_session_summary(user_id, session_id) if session_id else None
        preferences = self.search(user_id, "", limit=20)["preferences"]
        return {
            "user_id": user_id,
            "short_term": {
                "session": session,
                "recent_messages": self.get_messages(user_id, session_id, limit=20) if session_id else [],
                "rolling_summary": rolling_summary,
            },
            "long_term": {
                "trend": self.emotion_trend(user_id, "month"),
                "preferences": preferences,
            },
            "safety_note": "记忆用于本地个性化陪伴，不用于医疗或心理诊断。",
        }

    @staticmethod
    def _range_days(value: Any) -> int:
        mapping = {"week": 7, "month": 30, "quarter": 90, "year": 365}
        if str(value) in mapping:
            return mapping[str(value)]
        try:
            return max(1, min(int(value), 3650))
        except (TypeError, ValueError):
            return 30

    @staticmethod
    def _session_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "title": row["title"],
            "current_emotion": row["current_emotion"],
            "current_confidence": row["current_confidence"],
            "latest_analysis": _loads(row["latest_analysis_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _message_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "emotion": row["emotion"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _event_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "session_id": row["session_id"],
            "source": row["source"],
            "emotion": row["emotion"],
            "confidence": row["confidence"],
            "summary": row["summary"],
            "triggers": _loads(row["triggers_json"], []),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _preference_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "key": row["preference_key"],
            "value": row["preference_value"],
            "source": row["source"],
            "confirmed": bool(row["confirmed"]),
            "updated_at": row["updated_at"],
        }
