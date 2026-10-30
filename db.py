"""
db.py
SQLite storage: users (operator/admin) and the inspection log.
Default accounts are created on first run:  admin / admin123   operator / operator123
CHANGE THESE before any real use.
"""
import sqlite3
from datetime import datetime, timedelta

from werkzeug.security import check_password_hash, generate_password_hash

from config import DB_PATH


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin','operator')))""")
        c.execute("""CREATE TABLE IF NOT EXISTS inspections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT, image_name TEXT, source TEXT,
            verdict TEXT, defect_type TEXT, confidence REAL,
            reason TEXT, gradcam TEXT, inspected_at TEXT, operator TEXT)""")
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            for u, p, r in (("admin", "admin123", "admin"), ("operator", "operator123", "operator")):
                c.execute("INSERT INTO users(username,password_hash,role) VALUES(?,?,?)",
                          (u, generate_password_hash(p), r))


def verify_user(username: str, password: str):
    with conn() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    if row and check_password_hash(row["password_hash"], password):
        return dict(row)
    return None


def add_inspection(rec: dict):
    with conn() as c:
        c.execute("""INSERT INTO inspections(result_id,image_name,source,verdict,defect_type,
                     confidence,reason,gradcam,inspected_at,operator)
                     VALUES(:result_id,:image_name,:source,:verdict,:defect_type,
                     :confidence,:reason,:gradcam,:inspected_at,:operator)""", rec)


def get_inspections(verdict=None, source=None, limit=None):
    q, args = "SELECT * FROM inspections WHERE 1=1", []
    if verdict:
        q += " AND verdict=?"; args.append(verdict)
    if source:
        q += " AND source=?"; args.append(source)
    q += " ORDER BY id DESC"
    if limit:
        q += " LIMIT ?"; args.append(limit)
    with conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def stats():
    with conn() as c:
        total = c.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
        by_verdict = {r["verdict"]: r["n"] for r in c.execute(
            "SELECT verdict, COUNT(*) n FROM inspections GROUP BY verdict")}
        by_type = [dict(r) for r in c.execute(
            "SELECT defect_type, COUNT(*) n FROM inspections WHERE verdict='DEFECTIVE' "
            "GROUP BY defect_type ORDER BY n DESC")]
        since = (datetime.now() - timedelta(days=13)).strftime("%Y-%m-%d")
        per_day = [dict(r) for r in c.execute(
            "SELECT substr(inspected_at,1,10) day, COUNT(*) n, "
            "SUM(verdict='DEFECTIVE') defective FROM inspections "
            "WHERE substr(inspected_at,1,10)>=? GROUP BY day ORDER BY day", (since,))]
        avg_conf = c.execute("SELECT AVG(confidence) FROM inspections").fetchone()[0] or 0
    defective = by_verdict.get("DEFECTIVE", 0)
    return {
        "total": total,
        "good": by_verdict.get("GOOD", 0),
        "defective": defective,
        "review": by_verdict.get("NEEDS REVIEW", 0),
        "defect_rate": round(100 * defective / total, 1) if total else 0,
        "avg_conf": round(avg_conf, 1),
        "by_type": by_type,
        "per_day": per_day,
    }
