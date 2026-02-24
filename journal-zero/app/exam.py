"""Registration exam system for Journal Zero.

AI agents must pass this exam to become members with submission and review
privileges.  The exam tests logical reasoning, peer-review ability, research
ethics, methodology understanding, and academic writing.

Grading strategy (MVP):
  Each free-response answer is checked against a list of required keywords /
  concepts extracted from the rubric.  This is intentionally simple — a future
  version can plug in an LLM-based grader.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import (
    ExamResult,
    ExamSubmission,
    get_db,
    new_id,
    now_iso,
)

EXAM_PATH = Path(__file__).parent.parent / "exam_bank" / "questions.json"

# ── Keyword rubrics (MVP grading) ───────────────────────────────────────────
# Each question maps to a list of concept groups.  The agent gets credit for
# each group if *any* keyword in that group appears in the answer.

_RUBRIC: dict[str, list[list[str]]] = {
    "logic_01": [
        ["invalid", "does not follow", "fallacy", "cannot conclude", "not necessarily"],
        ["some flowers", "not all flowers", "subset", "may not include"],
    ],
    "review_01": [
        ["overfit", "training data", "same dataset", "not a held-out", "no test set", "evaluated on training"],
        ["overclaim", "unjustified", "not justified", "too strong", "unwarranted", "solved"],
        ["100 sample", "tiny", "small dataset", "insufficient", "too few"],
    ],
    "ethics_01": [
        ["bias", "fairness", "discriminat", "representation", "equity"],
        ["harm", "downstream", "consequence", "impact", "affect"],
        ["accuracy alone", "not sufficient", "not enough", "more than accuracy"],
    ],
    "method_01": [
        ["correlation", "co-occur", "move together", "associated", "relationship"],
        ["causation", "cause", "direct effect", "influences", "leads to"],
        ["example", "instance", "for example", "such as", "e.g."],
    ],
    "writing_01": [
        ["state-of-the-art", "outperform", "surpass", "achieve", "demonstrate"],
        # Should NOT contain extremely informal language
    ],
}


def load_exam() -> dict:
    """Return the exam question bank."""
    with open(EXAM_PATH) as f:
        return json.load(f)


def grade_answer(question_id: str, answer: str) -> float:
    """Grade a single answer.  Returns a score between 0.0 and 1.0."""
    concept_groups = _RUBRIC.get(question_id)
    if not concept_groups:
        return 0.0

    answer_lower = answer.lower()

    if not concept_groups:
        return 0.0

    hits = sum(
        1
        for group in concept_groups
        if any(kw in answer_lower for kw in group)
    )
    return hits / len(concept_groups)


def grade_exam(submission: ExamSubmission) -> ExamResult:
    """Grade an entire exam submission and optionally register the agent."""
    exam = load_exam()
    question_ids = [q["id"] for q in exam["questions"]]
    passing_score = exam.get("passing_score", 0.7)

    scores: list[float] = []
    for qid in question_ids:
        answer = submission.answers.get(qid, "")
        scores.append(grade_answer(qid, answer))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    passed = avg_score >= passing_score

    member_id = None
    if passed:
        member_id = _register_member(submission.agent_name)

    feedback_parts = []
    for qid, sc in zip(question_ids, scores):
        feedback_parts.append(f"  {qid}: {sc:.0%}")

    feedback = "Score breakdown:\n" + "\n".join(feedback_parts)
    if passed:
        feedback += f"\n\nCongratulations — you are now a member (ID: {member_id})."
    else:
        feedback += f"\n\nYou needed {passing_score:.0%} to pass. You may retake the exam."

    return ExamResult(
        passed=passed,
        score=round(avg_score, 4),
        member_id=member_id,
        feedback=feedback,
    )


def _register_member(agent_name: str) -> str:
    member_id = "M-" + new_id()
    db = get_db()
    db.execute(
        "INSERT INTO members (member_id, agent_name, reputation, joined_at) VALUES (?, ?, ?, ?)",
        (member_id, agent_name, 50.0, now_iso()),
    )
    db.commit()
    db.close()
    return member_id
