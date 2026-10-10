# Interview Accelerator

AI-powered interview preparation prototype for students and job candidates. The product follows the problem framing of StudentCredibility: turn career signals into actionable, explainable preparation rather than a generic ATS checker.

## Flow
JD + Resume → Role Analysis → Candidate Analysis → Job Fit → Screening → Competency → Deep-Dive → Voice Interaction → Performance Report → Preparation Gaps → Readiness

## Stack
- Python + FastAPI
- Gemini API (optional; set `GEMINI_API_KEY` in `.env`)
- SQLite
- HTML/CSS/JavaScript frontend
- Browser SpeechRecognition for speech-to-text
- Browser SpeechSynthesis for interviewer text-to-speech
- pypdf/python-docx for document uploads

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Put GEMINI_API_KEY in .env for full AI generation
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

Without a Gemini key, the prototype still runs with deterministic fallback analysis/questions for local UI testing. For the assignment demo, use a valid Gemini API key.

## Voice
Use Chrome/Edge. The microphone button uses the browser's SpeechRecognition implementation; the interviewer uses SpeechSynthesis. The transcript is sent to the backend, where the next question is generated from the JD, resume analysis, current level and previous answer.

## Architecture
The backend stores each session in SQLite. The analysis prompt produces structured role/candidate data. Interview turns are stored and passed back into the adaptive question prompt. After eight turns the report prompt generates competency scores, question-level feedback, strengths, weaknesses, preparation gaps and readiness.

## Design principle
Never claim resume facts that were not provided. Feedback focuses on relevance, clarity, evidence, technical depth and reasoning rather than facial/emotion inference.
