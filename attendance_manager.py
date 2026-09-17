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
from datetime import date, datetime, time

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "attendance.db")
LATE_AFTER = time(9, 0)


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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS registration_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    reviewed_at TEXT
                )
            """)

    def _connect(self):
        return sqlite3.connect(DB_PATH, check_same_thread=False)

    def create_registration_request(self, name: str, image_path: str) -> dict:
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO registration_requests
                   (name, image_path, status, created_at)
                   VALUES (?, ?, 'pending', ?)""",
                (name, image_path, datetime.now().isoformat(timespec="seconds"))
            )
            return {"id": cursor.lastrowid, "name": name, "status": "pending"}

    def get_registration_requests(self, status: str | None = None) -> list[dict]:
        query = "SELECT id, name, image_path, status, created_at FROM registration_requests"
        params = ()
        if status:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"id": r[0], "name": r[1], "image_path": r[2], "status": r[3], "created_at": r[4]}
            for r in rows
        ]

    def review_registration_request(self, request_id: int, status: str) -> dict | None:
        if status not in {"approved", "rejected"}:
            raise ValueError("Invalid registration review status.")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, image_path, status FROM registration_requests WHERE id = ?",
                (request_id,)
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE registration_requests SET status = ?, reviewed_at = ? WHERE id = ?",
                (status, datetime.now().isoformat(timespec="seconds"), request_id)
            )
        return {"id": row[0], "name": row[1], "image_path": row[2], "status": status}

    @staticmethod
    def _is_late(check_in: str | None) -> bool:
        if not check_in:
            return False
        try:
            return datetime.strptime(check_in, "%H:%M:%S").time() > LATE_AFTER
        except ValueError:
            return False

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
                        "is_late": self._is_late(now),
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
                        "is_late": self._is_late(check_in_time),
                        "message": f"{name} checked out at {now} (Duration: {duration})"
                    }
                else:
                    return {
                        "success": True,
                        "punch_type": "cooldown",
                        "already_marked": True,
                        "time": check_in_time,
                        "check_out": check_out_time or "--",
                        "is_late": self._is_late(check_in_time),
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
                "duration": calculate_duration(r[2], r[3]),
                "is_late": self._is_late(r[2])
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
                "duration": calculate_duration(r[2], r[3]),
                "is_late": self._is_late(r[2])
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
                "duration": calculate_duration(r[2], r[3]),
                "is_late": self._is_late(r[2])
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

    def get_person_summary(self, registered_names: list[str] | None = None) -> list[dict]:
        """Return each person's attendance percentage over the currently tracked date window."""
        if registered_names is None:
            registered_names = []

        available_dates = self.get_available_dates()
        total_days = max(1, len(available_dates))

        results = []
        with self._connect() as conn:
            for name in sorted(registered_names):
                count = conn.execute(
                    "SELECT COUNT(DISTINCT date) FROM attendance WHERE name = ?",
                    (name,)
                ).fetchone()[0] or 0
                pct = round((count / total_days) * 100, 1)
                results.append({
                    "name": name,
                    "attended_days": count,
                    "total_days": total_days,
                    "percentage": pct,
                })

        results.sort(key=lambda x: (-x["percentage"], x["name"]))
        return results

    def get_defaulters(self, registered_names: list[str], threshold: float = 75.0) -> list[dict]:
        """
        Calculate attendance percentages and return all students who fall below the threshold (e.g. 75%).
        """
        result = self.get_person_summary(registered_names)
        for item in result:
            item["is_defaulter"] = item["percentage"] < threshold

        result.sort(key=lambda x: (x["percentage"], x["name"]))
        return [item for item in result if item["is_defaulter"]]

    def get_attendance_champions(self, registered_names: list[str], limit: int = 5) -> list[dict]:
        """Return top performers with their consecutive attendance streak."""
        summaries = self.get_person_summary(registered_names)
        available_dates = self.get_available_dates()
        champions = []

        with self._connect() as conn:
            for item in summaries:
                rows = conn.execute(
                    "SELECT DISTINCT date FROM attendance WHERE name = ? ORDER BY date DESC",
                    (item["name"],)
                ).fetchall()
                attended_dates = {row[0] for row in rows}
                streak = 0
                for recorded_date in available_dates:
                    if recorded_date in attended_dates:
                        streak += 1
                    else:
                        break
                champions.append({
                    **item,
                    "streak": streak,
                    "badge": "Perfect Attendance" if item["percentage"] >= 100 else "Attendance Champion"
                })

        champions.sort(key=lambda x: (-x["percentage"], -x["streak"], -x["attended_days"], x["name"]))
        return champions[:limit]

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
