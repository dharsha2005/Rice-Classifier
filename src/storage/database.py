from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "results" / "rice_quality.db"


class PredictionStore:
    """Small SQLite repository for durable prediction and review records."""

    def __init__(self, database_path: Path = DATABASE_PATH) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30.0, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    needs_review INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    image_name TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,
                    corrected_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    normal_count INTEGER NOT NULL,
                    defect_count INTEGER NOT NULL,
                    defect_rate REAL NOT NULL,
                    review_required_count INTEGER NOT NULL,
                    recommended_action TEXT,
                    operator_action TEXT,
                    operator_notes TEXT
                );
                """
            )
            self._migrate_predictions(connection)

    def _migrate_predictions(self, connection: sqlite3.Connection) -> None:
        """Add recommendation/operator columns without deleting existing rows."""
        existing = {row[1] for row in connection.execute("PRAGMA table_info(predictions)").fetchall()}
        additions = {
            "recommended_action": "TEXT",
            "operator_action": "TEXT",
            "operator_notes": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {definition}")

    def save_prediction(self, record: Dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions
                (timestamp, image_name, source, predicted_class, confidence, needs_review,
                 recommended_action, operator_action, operator_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["image_name"],
                    record["source"],
                    record["predicted_class"],
                    float(record["confidence"]),
                    int(bool(record.get("needs_review", record.get("review_required", False)))),
                    record.get("recommended_action"),
                    record.get("operator_action"),
                    record.get("operator_notes"),
                ),
            )
            return int(cursor.lastrowid)

    def save_operator_decision(self, prediction_id: int, operator_action: str, operator_notes: str = "") -> None:
        """Store a human decision without overwriting the original ML prediction."""
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE predictions
                SET operator_action = ?, operator_notes = ?
                WHERE id = ?
                """,
                (operator_action, operator_notes, int(prediction_id)),
            )
            if cursor.rowcount == 0:
                raise ValueError(f"No prediction record found for id={prediction_id}")

    def save_batch_review(self, record: Dict[str, Any]) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO batch_reviews
                (timestamp, total_samples, normal_count, defect_count, defect_rate,
                 review_required_count, recommended_action, operator_action, operator_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    int(record["total_samples"]),
                    int(record["normal_count"]),
                    int(record["defect_count"]),
                    float(record["defect_rate"]),
                    int(record["review_required_count"]),
                    record.get("recommended_action"),
                    record.get("operator_action"),
                    record.get("operator_notes"),
                ),
            )
            return int(cursor.lastrowid)

    def save_correction(self, record: Dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_corrections
                (timestamp, image_name, original_prediction, corrected_label, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    record["image_name"],
                    record["original_prediction"],
                    record["corrected_label"],
                    float(record["confidence"]),
                    record["source"],
                ),
            )

    def recent_predictions(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, timestamp, image_name, source, predicted_class, confidence, needs_review,
                       recommended_action, operator_action, operator_notes
                FROM predictions ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        records = []
        for row in rows:
            item = dict(row)
            item["review_required"] = bool(item.get("needs_review"))
            item["model_probability"] = float(item.get("confidence") or 0.0)
            records.append(item)
        return records

    def recent_corrections(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp, image_name, original_prediction, corrected_label, confidence, source "
                "FROM review_corrections ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_prediction(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, timestamp, image_name, source, predicted_class, confidence, needs_review,
                       recommended_action, operator_action, operator_notes
                FROM predictions WHERE id = ?
                """,
                (int(prediction_id),),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["review_required"] = bool(item.get("needs_review"))
        item["model_probability"] = float(item.get("confidence") or 0.0)
        return item

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM predictions")
            connection.execute("DELETE FROM review_corrections")
            connection.execute("DELETE FROM batch_reviews")
