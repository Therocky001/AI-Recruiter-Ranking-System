# AI Recruiter Ranking System

## The Problem
Recruiters miss great candidates every day — not because talent isn't there, but because keyword-based filters can't see what actually matters. A candidate who built a production recommendation system gets filtered out because their resume doesn't say "RAG." A candidate with every buzzword but zero real depth sails through. We built a system that ranks candidates the way a sharp recruiter actually would — by understanding career trajectory, not string-matching skills.

## What We Built
A fully offline, CPU-only hybrid ranking pipeline that processes 100,000 candidate profiles against a job description and returns a validated, explainable top-100 shortlist — in ~33 minutes, zero API calls, zero GPU.

## Architecture
- **JD → Structured Requirements**
- **Stream 100K candidates** (JSONL, batched, memory-safe)
- **CPU embeddings** (sentence-transformers, MiniLM) — role history + project descriptions, not raw skill lists
- **23 behavioral/platform signal scoring** (response rate, activity recency, GitHub, verification, etc.)
- **9-stage rule-based penalty chain**
- **Composite score → Ranked, validated CSV output**

## Scoring Logic
`final_score = (embedding_similarity × 0.6 + signal_score × 0.4) × Π(penalty_multipliers)`

Multiplicative, not additive — one serious red flag (6 months inactive, title inflation, pure-academic career) can override an otherwise strong skill match. That's how real recruiters actually reason.

## The 9 Penalty Filters
Behavioral inactivity, WITCH-only consulting careers, pure-academic backgrounds, job-hopping, title inflation, low experience bands, hedging language in self-description, and missing production evidence (no vector DB / eval-metric mentions). On the full dataset, this filtered out 86,236 candidates for missing core technical evidence, 16,120 for insufficient experience, 10,439 for behavioral inactivity — proving the system separates real fits from keyword-matchers, not just re-sorts the same noise.

## Accuracy / Validation
- Manually audited top-ranked candidates against the JD's explicit disqualifiers (pure research, <12mo LangChain-only "AI experience," senior titles with no recent code) — false positives caught and fixed iteratively.
- Every reasoning string is template-generated from real candidate fields — zero LLM hallucination risk in the final ranking output.
- Output validated against the official `validate_submission.py` — 100/100 rows, correct schema, correct tie-breaking. Passes clean.

## Explainability
Every ranked candidate ships with a human-readable trace: title, years of experience, matched skills, and key behavioral signals — e.g. "Senior ML Engineer, 6.1yrs, Machine Learning match; active 64 days ago, 28% response rate." No black box.

## Tech Stack
Python · sentence-transformers (MiniLM) · Pydantic · scikit-learn · pandas/numpy · Streamlit (interactive demo dashboard) · Gemini (JD structuring, demo layer only — not in the offline submission path).

## Compliance
Fully satisfies the challenge's no-network / no-GPU constraint for the final ranking script — verified in `submission_metadata.yaml`.

## Ambition
This isn't a keyword filter with an AI label. It's a system that reasons about fit the way a great recruiter does — reading between the lines of a career, weighing whether someone is actually reachable and active, and being honest about who doesn't belong on the list, even if their resume looks perfect on paper.
