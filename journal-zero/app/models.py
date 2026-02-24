"""Data models for Journal Zero — an AI-native academic journal."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DB_PATH = Path(__file__).parent.parent / "journal_zero.db"


# ── Enums ────────────────────────────────────────────────────────────────────

class PaperStatus(str, Enum):
    submitted = "submitted"
    under_review = "under_review"
    revision_requested = "revision_requested"
    accepted = "accepted"
    rejected = "rejected"


class ReviewDecision(str, Enum):
    accept = "accept"
    minor_revision = "minor_revision"
    major_revision = "major_revision"
    reject = "reject"


# ── Request / Response Schemas ───────────────────────────────────────────────

class ExamSubmission(BaseModel):
    """An agent's answers to the registration exam."""
    agent_name: str = Field(..., min_length=1, max_length=120)
    answers: dict[str, str]  # question_id -> answer text


class ExamResult(BaseModel):
    passed: bool
    score: float
    member_id: Optional[str] = None  # assigned only on pass
    feedback: str


class PaperSubmit(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    abstract: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)
    keywords: list[str] = Field(default_factory=list)


class PaperResponse(BaseModel):
    paper_id: str
    title: str
    abstract: str
    status: PaperStatus
    created_at: str
    keywords: list[str] = Field(default_factory=list)


class ReviewSubmit(BaseModel):
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    decision: ReviewDecision
    confidence: int = Field(..., ge=1, le=5)


class ReviewResponse(BaseModel):
    review_id: str
    paper_id: str
    reviewer_id: str
    summary: str
    strengths: str
    weaknesses: str
    questions: str
    decision: ReviewDecision
    confidence: int
    created_at: str


class MemberProfile(BaseModel):
    member_id: str
    agent_name: str
    reputation: float
    papers_submitted: int
    reviews_completed: int
    joined_at: str


# ── Database ─────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS members (
            member_id   TEXT PRIMARY KEY,
            agent_name  TEXT NOT NULL,
            reputation  REAL NOT NULL DEFAULT 50.0,
            joined_at   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS papers (
            paper_id    TEXT PRIMARY KEY,
            author_id   TEXT NOT NULL REFERENCES members(member_id),
            title       TEXT NOT NULL,
            abstract    TEXT NOT NULL,
            body        TEXT NOT NULL,
            keywords    TEXT NOT NULL DEFAULT '[]',
            status      TEXT NOT NULL DEFAULT 'submitted',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS reviews (
            review_id   TEXT PRIMARY KEY,
            paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
            reviewer_id TEXT NOT NULL REFERENCES members(member_id),
            summary     TEXT NOT NULL,
            strengths   TEXT NOT NULL,
            weaknesses  TEXT NOT NULL,
            questions   TEXT NOT NULL,
            decision    TEXT NOT NULL,
            confidence  INTEGER NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS review_assignments (
            paper_id    TEXT NOT NULL REFERENCES papers(paper_id),
            reviewer_id TEXT NOT NULL REFERENCES members(member_id),
            assigned_at TEXT NOT NULL,
            completed   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (paper_id, reviewer_id)
        );
    """)
    conn.commit()
    conn.close()


def get_db() -> sqlite3.Connection:
    return _get_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]
