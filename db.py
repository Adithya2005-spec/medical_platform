import sqlite3
import pandas as pd
from datetime import date, datetime
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS Patient (
            PatientID INTEGER PRIMARY KEY AUTOINCREMENT,
            Name      TEXT    NOT NULL,
            Age       INTEGER NOT NULL,
            Gender    TEXT    NOT NULL,
            Contact   TEXT    DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS Diabetes_Record (
            Record_ID   INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID   INTEGER NOT NULL,
            Date        TEXT    NOT NULL,
            Glucose     REAL, BMI REAL, Insulin REAL,
            BP          REAL, SkinThickness REAL,
            Pregnancies INTEGER, Pedigree REAL, Age INTEGER,
            FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
        );
        CREATE TABLE IF NOT EXISTS Diabetes_Pred (
            Pred_ID    INTEGER PRIMARY KEY AUTOINCREMENT,
            Record_ID  INTEGER NOT NULL,
            Model_Used TEXT    NOT NULL,
            Risk_Level TEXT    NOT NULL,
            Confidence REAL    NOT NULL,
            FOREIGN KEY (Record_ID) REFERENCES Diabetes_Record(Record_ID)
        );
        CREATE TABLE IF NOT EXISTS Mood_Entry (
            Entry_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID       INTEGER NOT NULL,
            Date            TEXT    NOT NULL,
            Mood_Score      INTEGER NOT NULL,
            Journal_Text    TEXT,
            Sentiment_Label TEXT,
            Sentiment_Score REAL,
            FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
        );
        CREATE TABLE IF NOT EXISTS Image_Record (
            Image_ID        INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID       INTEGER NOT NULL,
            Date            TEXT    NOT NULL,
            Image_Path      TEXT,
            Image_Type      TEXT,
            Predicted_Class TEXT,
            Confidence      REAL,
            FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
        );
        CREATE TABLE IF NOT EXISTS Drug_Check (
            Check_ID  INTEGER PRIMARY KEY AUTOINCREMENT,
            PatientID INTEGER NOT NULL,
            Date      TEXT    NOT NULL,
            Drugs_Input TEXT  NOT NULL,
            Severity  TEXT,
            Details   TEXT,
            FOREIGN KEY (PatientID) REFERENCES Patient(PatientID)
        );
    """)
    conn.commit()

    # Seed demo patients if empty
    cur = conn.cursor()
    if cur.execute("SELECT COUNT(*) FROM Patient").fetchone()[0] == 0:
        cur.execute("INSERT INTO Patient (Name,Age,Gender,Contact) VALUES (?,?,?,?)",
                    ("Demo Patient A", 45, "Female", "demo@hospital.com"))
        cur.execute("INSERT INTO Patient (Name,Age,Gender,Contact) VALUES (?,?,?,?)",
                    ("Demo Patient B", 62, "Male", "demo2@hospital.com"))
        conn.commit()
    conn.close()


# ── PATIENTS ──────────────────────────────────────────────────────────────────
def add_patient(name, age, gender, contact=""):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO Patient (Name,Age,Gender,Contact) VALUES (?,?,?,?)",
        (name, int(age), gender, contact)
    )
    pid = cur.lastrowid
    conn.commit(); conn.close()
    return pid


def get_patients():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM Patient ORDER BY PatientID DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_patient(pid):
    conn = get_conn()
    row = conn.execute("SELECT * FROM Patient WHERE PatientID=?", (pid,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_patient(pid):
    conn = get_conn()
    conn.execute("DELETE FROM Patient WHERE PatientID=?", (pid,))
    conn.commit(); conn.close()


def search_patients(q):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM Patient WHERE Name LIKE ? OR CAST(PatientID AS TEXT)=?",
        (f"%{q}%", q)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── DIABETES ──────────────────────────────────────────────────────────────────
def add_diabetes_record(pid, glucose, bmi, insulin, bp, skin, preg, pedigree, age):
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO Diabetes_Record (PatientID,Date,Glucose,BMI,Insulin,BP,SkinThickness,Pregnancies,Pedigree,Age) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, date.today().isoformat(), glucose, bmi, insulin, bp, skin, preg, pedigree, age)
    )
    rid = cur.lastrowid
    conn.commit(); conn.close()
    return rid


def add_diabetes_pred(record_id, model_used, risk, confidence):
    conn = get_conn()
    conn.execute(
        "INSERT INTO Diabetes_Pred (Record_ID,Model_Used,Risk_Level,Confidence) VALUES (?,?,?,?)",
        (record_id, model_used, risk, round(confidence, 4))
    )
    conn.commit(); conn.close()


def get_diabetes_history(pid):
    conn = get_conn()
    rows = conn.execute("""
        SELECT r.Record_ID, r.Date, r.Glucose, r.BMI, r.Insulin,
               p.Model_Used, p.Risk_Level, ROUND(p.Confidence*100,2) AS Conf_Pct
        FROM Diabetes_Record r
        LEFT JOIN Diabetes_Pred p ON r.Record_ID=p.Record_ID
        WHERE r.PatientID=? ORDER BY r.Record_ID DESC
    """, (pid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_diabetes_preds():
    conn = get_conn()
    rows = conn.execute("SELECT Risk_Level FROM Diabetes_Pred").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── MOOD ──────────────────────────────────────────────────────────────────────
def add_mood_entry(pid, score, journal, label, polarity):
    conn = get_conn()
    conn.execute(
        "INSERT INTO Mood_Entry (PatientID,Date,Mood_Score,Journal_Text,Sentiment_Label,Sentiment_Score) VALUES (?,?,?,?,?,?)",
        (pid, date.today().isoformat(), score, journal, label, round(polarity, 4))
    )
    conn.commit(); conn.close()


def get_mood_history(pid, limit=30):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM Mood_Entry WHERE PatientID=? ORDER BY Entry_ID DESC LIMIT ?",
        (pid, limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_mood():
    conn = get_conn()
    rows = conn.execute(
        "SELECT Date,Mood_Score,Sentiment_Label FROM Mood_Entry ORDER BY Entry_ID DESC LIMIT 30"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── IMAGE ──────────────────────────────────────────────────────────────────────
def add_image_record(pid, path, img_type, pred_class, conf):
    conn = get_conn()
    conn.execute(
        "INSERT INTO Image_Record (PatientID,Date,Image_Path,Image_Type,Predicted_Class,Confidence) VALUES (?,?,?,?,?,?)",
        (pid, date.today().isoformat(), path, img_type, pred_class, round(conf, 4))
    )
    conn.commit(); conn.close()


def get_image_history(pid):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM Image_Record WHERE PatientID=? ORDER BY Image_ID DESC", (pid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── DRUGS ──────────────────────────────────────────────────────────────────────
def add_drug_check(pid, drugs_str, severity, details):
    conn = get_conn()
    conn.execute(
        "INSERT INTO Drug_Check (PatientID,Date,Drugs_Input,Severity,Details) VALUES (?,?,?,?,?)",
        (pid, date.today().isoformat(), drugs_str, severity, details)
    )
    conn.commit(); conn.close()


def get_drug_history(pid):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM Drug_Check WHERE PatientID=? ORDER BY Check_ID DESC", (pid,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── DASHBOARD ─────────────────────────────────────────────────────────────────
def get_counts():
    conn = get_conn()
    cur = conn.cursor()
    out = {
        "patients": cur.execute("SELECT COUNT(*) FROM Patient").fetchone()[0],
        "diabetes": cur.execute("SELECT COUNT(*) FROM Diabetes_Record").fetchone()[0],
        "mood":     cur.execute("SELECT COUNT(*) FROM Mood_Entry").fetchone()[0],
        "images":   cur.execute("SELECT COUNT(*) FROM Image_Record").fetchone()[0],
        "drugs":    cur.execute("SELECT COUNT(*) FROM Drug_Check").fetchone()[0],
    }
    conn.close()
    return out


def get_recent_activity():
    conn = get_conn()
    results = []
    queries = [
        """SELECT p.Name, 'Diabetes' AS Module, d.Date, dp.Risk_Level AS Result
           FROM Diabetes_Record d JOIN Patient p ON d.PatientID=p.PatientID
           LEFT JOIN Diabetes_Pred dp ON d.Record_ID=dp.Record_ID
           ORDER BY d.Record_ID DESC LIMIT 3""",
        """SELECT p.Name, 'Mood' AS Module, m.Date, m.Sentiment_Label AS Result
           FROM Mood_Entry m JOIN Patient p ON m.PatientID=p.PatientID
           ORDER BY m.Entry_ID DESC LIMIT 3""",
        """SELECT p.Name, 'Image' AS Module, i.Date, i.Predicted_Class AS Result
           FROM Image_Record i JOIN Patient p ON i.PatientID=p.PatientID
           ORDER BY i.Image_ID DESC LIMIT 3""",
        """SELECT p.Name, 'Drug Check' AS Module, d.Date, d.Severity AS Result
           FROM Drug_Check d JOIN Patient p ON d.PatientID=p.PatientID
           ORDER BY d.Check_ID DESC LIMIT 2""",
    ]
    for q in queries:
        try:
            rows = conn.execute(q).fetchall()
            results.extend([dict(r) for r in rows])
        except Exception:
            pass
    conn.close()
    results.sort(key=lambda x: x.get("Date",""), reverse=True)
    return results[:8]
