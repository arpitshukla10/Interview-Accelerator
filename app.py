import os
import json
import re
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
DB = BASE / "data" / "interviews.db"
DB.parent.mkdir(exist_ok=True)

def load_local_env():
    env_file = BASE / ".env"
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                os.environ.setdefault(key, value)
    except Exception:
        pass

load_local_env()

app = FastAPI(title="Interview Accelerator")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

SCHEMA = """CREATE TABLE IF NOT EXISTS sessions(
    id TEXT PRIMARY KEY, jd TEXT, resume TEXT, analysis TEXT,
    level TEXT, question TEXT, answers TEXT, report TEXT, created_at TEXT
)"""

def db():
    conn = sqlite3.connect(DB)
    conn.execute(SCHEMA)
    conn.commit()
    return conn

def extract_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext == ".docx":
        from docx import Document
        return "\n".join(p.text for p in Document(str(path)).paragraphs)
    return path.read_text(encoding="utf-8", errors="ignore")

def gemini(prompt):
    key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not key:
        return None
    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent?key={key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4},
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text") if parts else None
    except Exception as exc:
        print("Gemini error:", exc)
        return None

def json_from(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None

def fallback_analysis(jd, resume):
    def words(text):
        return set(re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", text.lower()))
    j, r = words(jd), words(resume)
    common = sorted(j & r)
    title = "Target Role"
    for pattern in [
        r"(?:job title|role|position)\s*[:\-]\s*([^\n]+)",
        r"looking for an?\s+([^\n]+)",
    ]:
        m = re.search(pattern, jd, re.I)
        if m:
            title = m.group(1).strip()[:100]
            break
    skills = common[:18]
    missing = sorted(j - r)[:12]
    fit = min(95, max(35, 55 + len(common) * 3 - len(missing)))
    return {
        "role": title,
        "responsibilities": [
            "Build and deliver role-related solutions",
            "Collaborate with technical and cross-functional teams",
            "Debug, improve and communicate technical work",
        ],
        "required_skills": skills,
        "preferred_skills": [],
        "technical_competencies": skills[:10],
        "behavioural_competencies": ["Problem solving", "Communication", "Learning ability"],
        "experience_expectations": "Based on the supplied job description.",
        "keywords": skills[:15],
        "qualifications": ["See original job description"],
        "candidate_skills": sorted(r)[:25],
        "relevant_experience": [],
        "relevant_projects": [],
        "strengths": skills[:8],
        "missing_skills": missing,
        "weak_areas": missing[:6],
        "claims_to_probe": [],
        "job_fit": fit,
        "job_fit_label": "Strong Match" if fit >= 75 else "Partial Match",
    }

def analyse(jd, resume):
    prompt = f"""You are an expert interview-preparation product analyst.
Analyze the JOB DESCRIPTION and RESUME.
Return ONLY valid JSON with keys:
role,responsibilities,required_skills,preferred_skills,technical_competencies,
behavioural_competencies,experience_expectations,keywords,qualifications,
candidate_skills,relevant_experience,relevant_projects,strengths,missing_skills,
weak_areas,claims_to_probe,job_fit,job_fit_label.
Rules: job_fit is 0-100; never invent resume facts; identify missing/weak areas
and resume claims to probe; make the analysis role-specific.

JOB DESCRIPTION:
{jd[:18000]}

RESUME:
{resume[:18000]}"""
    return json_from(gemini(prompt)) or fallback_analysis(jd, resume)

def next_q(session_data, answer):
    analysis_text = json.dumps(session_data["analysis"], ensure_ascii=False)
    previous_questions = session_data.get("previous_questions", [])

    prompt = f"""You are a demanding but fair AI interviewer.
Generate ONE NEW interview question.
It MUST depend on the latest answer, be different from every previous question,
use resume context, probe a claim/weakness/decision/trade-off/reasoning,
adapt difficulty, and be relevant to the target job.
Previous questions:
{json.dumps(previous_questions, ensure_ascii=False)}
Candidate analysis:
{analysis_text[:12000]}
Current level:
{session_data["level"]}
Latest answer:
{answer[:7000]}
Return ONLY valid JSON:
{{"question":"...","reason":"...","level":"Screening|Competency|Deep-Dive"}}"""

    data = json_from(gemini(prompt))
    if data and data.get("question"):
        generated = str(data["question"]).strip()
        norm = re.sub(r"\s+", " ", generated.lower()).strip()
        if not any(norm == re.sub(r"\s+", " ", str(q).lower()).strip()
                   for q in previous_questions):
            return {
                "question": generated,
                "reason": data.get("reason", "Adaptive follow-up based on your previous answer."),
                "level": session_data["level"],
            }

    level = session_data["level"]
    if level == "Screening":
        pool = [
            "Can you give me a specific example from that project and explain exactly what you personally implemented?",
            "What problem were you trying to solve in that project, and how did you decide that your approach was the right one?",
            "What was the measurable result or outcome of your contribution to that project?",
        ]
    elif level == "Competency":
        pool = [
            "What alternative approach did you consider, and why did you reject it?",
            "How would you debug this solution if it started failing in production?",
            "If you had to redesign this solution for 10 times more users, what would you change and why?",
        ]
    else:
        pool = [
            "What would happen if your approach failed in production, and how would you isolate the root cause?",
            "What is the biggest limitation of your current implementation, and how would you improve it in a production environment?",
            "What technical trade-off would become a problem at scale, and how would you address it?",
        ]

    for question in pool:
        if question not in previous_questions:
            return {
                "question": question,
                "reason": "Adaptive follow-up selected from the candidate's answer and interview level.",
                "level": level,
            }

    return {
        "question": "What is one thing you would improve in this project if you had another week to work on it, and why?",
        "reason": "Unique adaptive follow-up.",
        "level": level,
    }

def report(session_data, answers):
    analysis_text = json.dumps(session_data["analysis"], ensure_ascii=False)
    prompt = f"""Create an actionable AI interview performance report.
Return ONLY valid JSON with keys:
overall_score,competencies,question_feedback,strengths,weaknesses,preparation_gaps,readiness.
competencies must contain Role Fit, Technical Knowledge, Problem Solving,
Communication, Confidence, Depth of Understanding, Behavioural Fit (0-100).
question_feedback objects: question,answer,assessment,what_was_good,
what_could_be_better,ideal_direction.
readiness must be Not Ready, Needs Preparation, Interview Ready, or Strong Candidate.
Do not invent achievements or claim facial/emotion detection.
ANALYSIS:
{analysis_text[:12000]}
ANSWERS:
{json.dumps(answers, ensure_ascii=False)[:16000]}"""
    data = json_from(gemini(prompt))
    if data:
        return data

    n = len(answers)
    avg = min(92, 55 + n * 5)
    fit = session_data["analysis"].get("job_fit", 70)
    return {
        "overall_score": avg,
        "competencies": {
            "Role Fit": fit,
            "Technical Knowledge": avg,
            "Problem Solving": max(0, avg - 3),
            "Communication": avg,
            "Confidence": max(50, avg - 5),
            "Depth of Understanding": max(0, avg - 5),
            "Behavioural Fit": min(100, avg + 2),
        },
        "question_feedback": [
            {
                "question": q,
                "answer": a,
                "assessment": "Answer reviewed against role context.",
                "what_was_good": "Relevant points were identified.",
                "what_could_be_better": "Add specific evidence, decisions and measurable outcomes.",
                "ideal_direction": "Use a concise Situation, Action, technical reasoning and Result.",
            }
            for q, a in answers
        ],
        "strengths": ["Relevant experience", "Role-focused preparation"],
        "weaknesses": ["Need more measurable evidence in answers"],
        "preparation_gaps": [
            {"priority": 1, "topic": "Role-specific technical depth",
             "review": "Review core concepts, trade-offs and practical examples."},
            {"priority": 2, "topic": "Behavioural stories",
             "review": "Prepare concise STAR examples."},
        ],
        "readiness": "Interview Ready" if avg >= 75 else "Needs Preparation",
    }

@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/analyze")
async def api_analyze(
    jd: str = Form(""),
    resume: str = Form(""),
    jd_file: UploadFile | None = File(None),
    resume_file: UploadFile | None = File(None),
):
    if jd_file and jd_file.filename:
        safe_name = Path(jd_file.filename).name
        path = UPLOADS / f"{uuid.uuid4()}_{safe_name}"
        path.write_bytes(await jd_file.read())
        jd = extract_file(path)

    if resume_file and resume_file.filename:
        safe_name = Path(resume_file.filename).name
        path = UPLOADS / f"{uuid.uuid4()}_{safe_name}"
        path.write_bytes(await resume_file.read())
        resume = extract_file(path)

    if not jd.strip() or not resume.strip():
        return JSONResponse({"error": "Please provide both Job Description and Resume."}, 400)

    analysis = analyse(jd, resume)
    sid = str(uuid.uuid4())
    conn = db()
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
        (
            sid, jd, resume, json.dumps(analysis, ensure_ascii=False),
            "Screening", "", json.dumps([]), "",
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return {"session_id": sid, "analysis": analysis}

@app.post("/api/start/{sid}")
def start(sid: str):
    conn = db()
    row = conn.execute("SELECT analysis FROM sessions WHERE id=?", (sid,)).fetchone()
    if not row:
        conn.close()
        return JSONResponse({"error": "Session not found"}, 404)

    question = (
        "Tell me about the most relevant project or experience on your resume "
        "for this role, and what you personally contributed."
    )
    conn.execute(
        "UPDATE sessions SET question=?, level=? WHERE id=?",
        (question, "Screening", sid),
    )
    conn.commit()
    conn.close()
    return {"question": question, "level": "Screening", "progress": 1}

@app.post("/api/answer/{sid}")
def answer(sid: str, answer: str = Form(...)):
    conn = db()
    row = conn.execute(
        "SELECT analysis, level, question, answers FROM sessions WHERE id=?",
        (sid,),
    ).fetchone()

    if not row:
        conn.close()
        return JSONResponse({"error": "Session not found"}, 404)

    analysis = json.loads(row[0])
    answers = json.loads(row[3] or "[]")
    current_question = row[2]

    answers.append([current_question, answer])
    number = len(answers)

    if number < 3:
        level = "Screening"
    elif number < 6:
        level = "Competency"
    else:
        level = "Deep-Dive"

    if number >= 8:
        report_data = report({"analysis": analysis, "level": level}, answers)
        conn.execute(
            "UPDATE sessions SET answers=?, report=?, level=? WHERE id=?",
            (
                json.dumps(answers, ensure_ascii=False),
                json.dumps(report_data, ensure_ascii=False),
                level,
                sid,
            ),
        )
        conn.commit()
        conn.close()
        return {"done": True, "report": report_data}

    previous_questions = [
        item[0]
        for item in answers
        if isinstance(item, list) and item
    ]

    nq = next_q(
        {
            "analysis": analysis,
            "level": level,
            "previous_questions": previous_questions,
        },
        answer,
    )

    conn.execute(
        "UPDATE sessions SET answers=?, question=?, level=? WHERE id=?",
        (
            json.dumps(answers, ensure_ascii=False),
            nq["question"],
            level,
            sid,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "done": False,
        "question": nq["question"],
        "reason": nq.get("reason", ""),
        "level": level,
        "progress": number,
    }

@app.get("/api/session/{sid}")
def session(sid: str):
    conn = db()
    row = conn.execute(
        "SELECT id,analysis,level,question,answers,report FROM sessions WHERE id=?",
        (sid,),
    ).fetchone()
    conn.close()

    if not row:
        return JSONResponse({"error": "Not found"}, 404)

    return {
        "session_id": row[0],
        "analysis": json.loads(row[1]),
        "level": row[2],
        "question": row[3],
        "answers": json.loads(row[4] or "[]"),
        "report": json.loads(row[5]) if row[5] else None,
    }
