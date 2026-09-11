"""
attendance_manager.py
SQLite-backed Attendance Manager
- Marks attendance (one entry per person per day)
- Queries attendance by date range
- Exports to CSV
"""

import sqlite3
import csv
import os
import io
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "attendance.db")


class AttendanceManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    # ─────────────────────────────────────────────
    # DB setup
    # ─────────────────────────────────────────────

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      TEXT    NOT NULL,
                    date      TEXT    NOT NULL,
                    time      TEXT    NOT NULL,
                    UNIQUE(name, date)
                )
            """)

    def _connect(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    # ─────────────────────────────────────────────
    # Write operations
    # ─────────────────────────────────────────────

    def mark_attendance(self, name: str) -> dict:
        """
        Mark a person as present today.
        Returns {"success": bool, "already_marked": bool, "message": str}
        """
        today = date.today().isoformat()
        now = datetime.now().strftime("%H:%M:%S")
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO attendance (name, date, time) VALUES (?, ?, ?)",
                    (name, today, now)
                )
                changes = conn.execute("SELECT changes()").fetchone()[0]
            if changes:
                return {"success": True, "already_marked": False,
                        "message": f"{name} marked present at {now}"}
            else:
                return {"success": True, "already_marked": True,
                        "message": f"{name} already marked today"}
        except Exception as e:
            return {"success": False, "already_marked": False, "message": str(e)}

    def clear_date(self, target_date: str):
        """Delete all records for a given date (YYYY-MM-DD)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM attendance WHERE date = ?", (target_date,))

    # ─────────────────────────────────────────────
    # Read operations
    # ─────────────────────────────────────────────

    def get_attendance(self, target_date: str | None = None) -> list[dict]:
        """
        Fetch attendance records.
        If target_date is None, returns today's records.
        """
        if target_date is None:
            target_date = date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, date, time FROM attendance WHERE date = ? ORDER BY time",
                (target_date,)
            ).fetchall()
        return [{"name": r[0], "date": r[1], "time": r[2]} for r in rows]

    def get_all_attendance(self) -> list[dict]:
        """Return all attendance records ordered by date desc, time asc."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, date, time FROM attendance ORDER BY date DESC, time ASC"
            ).fetchall()
        return [{"name": r[0], "date": r[1], "time": r[2]} for r in rows]

    def get_summary_by_date(self) -> list[dict]:
        """Returns count of attendees per date for charting."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, COUNT(*) as count FROM attendance GROUP BY date ORDER BY date DESC LIMIT 30"
            ).fetchall()
        return [{"date": r[0], "count": r[1]} for r in rows]

    def get_available_dates(self) -> list[str]:
        """All distinct dates that have any attendance records."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT date FROM attendance ORDER BY date DESC"
            ).fetchall()
        return [r[0] for r in rows]

    def get_stats(self, target_date: str | None = None) -> dict:
        """Return quick stats: total registered (passed in), present, absent."""
        records = self.get_attendance(target_date)
        return {
            "date": target_date or date.today().isoformat(),
            "present_count": len(records),
            "records": records
        }

    # ─────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────

    def export_csv(self, target_date: str | None = None) -> str:
        """Return attendance as a CSV string. All records if date is None."""
        records = self.get_all_attendance() if target_date is None else self.get_attendance(target_date)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=["name", "date", "time"])
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

