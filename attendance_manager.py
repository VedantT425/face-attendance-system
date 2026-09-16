"""
attendance_manager.py
SQLite-backed Attendance Manager
- Marks check-in & check-out attendance (with duration calculation)
- Queries attendance by specific date or date range
- Calculates attendance percentage and defaulters list (< 75%)
- Exports detailed logs to CSV
"""

import sqlite3
import csv
import os
import io
from datetime import date, datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "attendance.db")


def calculate_duration(check_in: str | None, check_out: str | None) -> str:
    """Calculate formatted duration (e.g. '2h 15m') between check_in and check_out times."""
    if not check_in or not check_out or check_out == "--":
        return "--"
    try:
        t1 = datetime.strptime(check_in, "%H:%M:%S")
        t2 = datetime.strptime(check_out, "%H:%M:%S")
        diff_seconds = int((t2 - t1).total_seconds())
        if diff_seconds < 0:
            diff_seconds += 24 * 3600
        hrs = diff_seconds // 3600
        mins = (diff_seconds % 3600) // 60
        secs = diff_seconds % 60
        if hrs > 0:
            return f"{hrs}h {mins}m"
        elif mins > 0:
            return f"{mins}m {secs}s"
        else:
            return f"{secs}s"
    except Exception:
        return "--"


class AttendanceManager:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._init_db()

    # ─────────────────────────────────────────────
    # DB setup & Migrations
    # ─────────────────────────────────────────────

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    name      TEXT    NOT NULL,
                    date      TEXT    NOT NULL,
                    time      TEXT    NOT NULL,
                    check_out TEXT,
                    UNIQUE(name, date)
                )
            """)
            # Check if check_out column exists (migration for existing database)
            cursor = conn.execute("PRAGMA table_info(attendance)")
            cols = [row[1] for row in cursor.fetchall()]
            if "check_out" not in cols:
                conn.execute("ALTER TABLE attendance ADD COLUMN check_out TEXT")

    def _connect(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    # ─────────────────────────────────────────────
    # Write operations
    # ─────────────────────────────────────────────

    def mark_attendance(self, name: str, punch_type: str = "auto") -> dict:
        """
        Mark a person's attendance.
        - If first time today: logs check-in.
        - If already checked in and punch_type is 'out' or after a short delay: updates check-out.
        Returns {"success": bool, "punch_type": "in"|"out"|"cooldown", "already_marked": bool, "message": str}
        """
        today = date.today().isoformat()
        now = datetime.now().strftime("%H:%M:%S")
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT time, check_out FROM attendance WHERE name = ? AND date = ?",
                    (name, today)
                ).fetchone()

                if not row:
                    # First punch of the day: Check-in
                    conn.execute(
                        "INSERT INTO attendance (name, date, time, check_out) VALUES (?, ?, ?, NULL)",
                        (name, today, now)
                    )
                    return {
                        "success": True,
                        "punch_type": "in",
                        "already_marked": False,
                        "time": now,
                        "message": f"{name} checked in at {now}"
                    }

                check_in_time, check_out_time = row[0], row[1]

                # Check time difference in seconds since check_in
                try:
                    t_in = datetime.strptime(check_in_time, "%H:%M:%S")
                    t_now = datetime.strptime(now, "%H:%M:%S")
                    diff_sec = (t_now - t_in).total_seconds()
                except Exception:
                    diff_sec = 100

                # If requested punch_type is explicitly 'out' or auto with >= 60 seconds diff
                if punch_type == "out" or (punch_type == "auto" and diff_sec >= 60):
                    conn.execute(
                        "UPDATE attendance SET check_out = ? WHERE name = ? AND date = ?",
                        (now, name, today)
                    )
                    duration = calculate_duration(check_in_time, now)
                    return {
                        "success": True,
                        "punch_type": "out",
                        "already_marked": True,
                        "time": now,
                        "check_in": check_in_time,
                        "check_out": now,
                        "duration": duration,
                        "message": f"{name} checked out at {now} (Duration: {duration})"
                    }
                else:
                    return {
                        "success": True,
                        "punch_type": "cooldown",
                        "already_marked": True,
                        "time": check_in_time,
                        "check_out": check_out_time or "--",
                        "message": f"{name} already checked in at {check_in_time}"
                    }
        except Exception as e:
            return {"success": False, "punch_type": "error", "already_marked": False, "message": str(e)}

    def mark_manual_entry(self, name: str, target_date: str, check_in: str, check_out: str | None = None) -> dict:
        """Admin override: manually record or update attendance for any student and date."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO attendance (name, date, time, check_out)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(name, date) DO UPDATE SET
                        time = excluded.time,
                        check_out = excluded.check_out
                    """,
                    (name, target_date, check_in, check_out)
                )
            return {"success": True, "message": f"Manual record updated for {name} on {target_date}"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def clear_date(self, target_date: str):
        """Delete all records for a given date (YYYY-MM-DD)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM attendance WHERE date = ?", (target_date,))

    # ─────────────────────────────────────────────
    # Read operations
    # ─────────────────────────────────────────────

    def get_attendance(self, target_date: str | None = None) -> list[dict]:
        """Fetch attendance records for a given date."""
        if target_date is None:
            target_date = date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, date, time, check_out FROM attendance WHERE date = ? ORDER BY time",
                (target_date,)
            ).fetchall()
        return [
            {
                "name": r[0],
                "date": r[1],
                "time": r[2],
                "check_in": r[2],
                "check_out": r[3] if r[3] else "--",
                "duration": calculate_duration(r[2], r[3])
            }
            for r in rows
        ]

    def get_attendance_range(self, start_date: str, end_date: str) -> list[dict]:
        """Fetch attendance records within a date range [start_date, end_date]."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, date, time, check_out FROM attendance WHERE date >= ? AND date <= ? ORDER BY date DESC, time ASC",
                (start_date, end_date)
            ).fetchall()
        return [
            {
                "name": r[0],
                "date": r[1],
                "time": r[2],
                "check_in": r[2],
                "check_out": r[3] if r[3] else "--",
                "duration": calculate_duration(r[2], r[3])
            }
            for r in rows
        ]

    def get_all_attendance(self) -> list[dict]:
        """Return all attendance records ordered by date desc, time asc."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name, date, time, check_out FROM attendance ORDER BY date DESC, time ASC"
            ).fetchall()
        return [
            {
                "name": r[0],
                "date": r[1],
                "time": r[2],
                "check_in": r[2],
                "check_out": r[3] if r[3] else "--",
                "duration": calculate_duration(r[2], r[3])
            }
            for r in rows
        ]

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

    def get_defaulters(self, registered_names: list[str], threshold: float = 75.0) -> list[dict]:
        """
        Calculate attendance percentages and return all students who fall below the threshold (e.g. 75%).
        """
        available_dates = self.get_available_dates()
        total_days = max(1, len(available_dates))

        result = []
        with self._connect() as conn:
            for name in registered_names:
                count = conn.execute(
                    "SELECT COUNT(DISTINCT date) FROM attendance WHERE name = ?",
                    (name,)
                ).fetchone()[0]
                pct = round((count / total_days) * 100, 1)
                result.append({
                    "name": name,
                    "attended_days": count,
                    "total_days": total_days,
                    "percentage": pct,
                    "is_defaulter": pct < threshold
                })

        # Sort: lowest attendance percentage first
        result.sort(key=lambda x: x["percentage"])
        return result

    def get_stats(self, target_date: str | None = None) -> dict:
        """Return quick stats: date, present_count, records."""
        records = self.get_attendance(target_date)
        return {
            "date": target_date or date.today().isoformat(),
            "present_count": len(records),
            "records": records
        }

    # ─────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────

    def export_csv(self, target_date: str | None = None, start_date: str | None = None, end_date: str | None = None) -> str:
        """Return attendance as a CSV string with check_in, check_out, and duration."""
        if start_date and end_date:
            records = self.get_attendance_range(start_date, end_date)
        elif target_date:
            records = self.get_attendance(target_date)
        else:
            records = self.get_all_attendance()

        output = io.StringIO()
        fieldnames = ["name", "date", "check_in", "check_out", "duration"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()
