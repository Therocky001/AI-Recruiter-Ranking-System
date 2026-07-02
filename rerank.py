import asyncio
import json
from google import genai
from google.genai import types
from schema import JDRequirements, CandidateProfile
from pydantic import BaseModel, Field
from typing import List, Optional

class CandidateScore(BaseModel):
    candidate_id: str
    score: float = Field(description="Score from 0 to 100")
    key_strengths: List[str]
    gaps: List[str]
    reasoning: str = Field(description="Max 2 sentences of reasoning")
    confidence: str = Field(description="high, medium, or low")

async def score_candidate(client: genai.Client, jd: JDRequirements, candidate: CandidateProfile, model_id: str) -> CandidateScore:
    prompt = f"""
    Score this candidate's fit for the role 0-100.
    Consider skill match, experience relevance, domain alignment, and behavioral/platform signals.
    
    Job Requirements:
    Role: {jd.role_title}
    Domain: {jd.domain}
    Experience: {jd.min_years_exp}+ years
    Must Have Skills: {', '.join(jd.must_have_skills)}
    Nice To Have: {', '.join(jd.nice_to_have_skills)}
    Soft Signals: {', '.join(jd.soft_signals)}
    
    Candidate Profile:
    Name: {candidate.name}
    Total Experience: {candidate.total_years_exp} years
    Domains: {', '.join(candidate.domains)}
    Skills: {', '.join(candidate.core_skills)}
    Platform Signals: {json.dumps(candidate.platform_signals)}
    """
    
    try:
        response = await client.aio.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CandidateScore,
                temperature=0.2
            )
        )
        data = json.loads(response.text)
        # Ensure candidate_id matches what was passed in
        data["candidate_id"] = candidate.candidate_id
        return CandidateScore(**data)
    except Exception as e:
        print(f"Error scoring candidate {candidate.candidate_id}: {e}")
        return CandidateScore(
            candidate_id=candidate.candidate_id,
            score=0.0,
            key_strengths=[],
            gaps=["Error during LLM evaluation"],
            reasoning=f"Failed to score: {e}",
            confidence="low"
        )

async def async_rerank_candidates(jd: JDRequirements, candidates: List[CandidateProfile]) -> List[CandidateScore]:
    """Score multiple candidates concurrently using Gemini."""
    client = genai.Client()
    model_id = "gemini-2.5-flash"
    
    tasks = [score_candidate(client, jd, cand, model_id) for cand in candidates]
    scores = await asyncio.gather(*tasks)
    
    # Sort descending by score
    scores.sort(key=lambda x: x.score, reverse=True)
    return scores
