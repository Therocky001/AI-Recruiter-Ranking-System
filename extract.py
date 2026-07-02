import os
import json
from google import genai
from google.genai import types
from schema import JDRequirements, CandidateProfile, Role
from pydantic import ValidationError

def extract_jd(jd_text: str) -> JDRequirements:
    """Extract structured JDRequirements from raw JD text using Gemini."""
    client = genai.Client()
    
    # We use gemini-2.5-flash as the default model for structured extraction
    model_id = "gemini-2.5-flash"
    
    prompt = f"""
    Extract the key requirements from the following Job Description.
    Map it strictly to the requested JSON schema.
    
    Job Description:
    {jd_text}
    """
    
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JDRequirements,
            temperature=0.1
        )
    )
    
    try:
        data = json.loads(response.text)
        return JDRequirements(**data)
    except Exception as e:
        print(f"Error parsing JD: {e}")
        # Return a fallback empty schema
        return JDRequirements(
            role_title="Unknown Role",
            must_have_skills=[],
            nice_to_have_skills=[],
            min_years_exp=0.0,
            domain="Unknown",
            seniority_level="Unknown",
            soft_signals=[]
        )

def extract_candidate(raw_cand: dict) -> CandidateProfile:
    """Deterministically map raw candidate JSON dict to CandidateProfile."""
    
    # Safely get nested fields
    profile = raw_cand.get("profile", {})
    candidate_id = raw_cand.get("candidate_id", "Unknown")
    name = profile.get("anonymized_name", "Unknown Candidate")
    total_years_exp = profile.get("years_of_experience", 0.0)
    
    skills = raw_cand.get("skills", [])
    core_skills = [s.get("name") for s in skills if isinstance(s, dict) and s.get("name")]
    
    # Domains from current and past roles
    domains = set()
    current_ind = profile.get("current_industry")
    if current_ind:
        domains.add(current_ind)
        
    past_roles = []
    career_hist = raw_cand.get("career_history", [])
    for job in career_hist:
        if not isinstance(job, dict):
            continue
        company = job.get("company", "Unknown")
        title = job.get("title", "Unknown")
        desc = job.get("description", "")
        past_roles.append(Role(title=title, company=company, description=desc))
        
        ind = job.get("industry")
        if ind:
            domains.add(ind)
            
    # Notable projects can be inferred from summary or role descriptions, but since we are doing 
    # deterministic mapping, we will just include a generic placeholder or omit them.
    notable_projects = []
    
    redrob_signals = raw_cand.get("redrob_signals", {})
    platform_signals = {}
    if isinstance(redrob_signals, dict):
        for k, v in redrob_signals.items():
            platform_signals[k] = v
                
    education = []
    ed_hist = raw_cand.get("education", [])
    for ed in ed_hist:
        if not isinstance(ed, dict):
            continue
        degree = ed.get("degree", "")
        field = ed.get("field_of_study", "")
        inst = ed.get("institution", "")
        ed_str = f"{degree} in {field} from {inst}".strip()
        if ed_str:
            education.append(ed_str)

    return CandidateProfile(
        candidate_id=candidate_id,
        name=name,
        total_years_exp=total_years_exp,
        core_skills=core_skills,
        domains=list(domains),
        past_roles=past_roles,
        notable_projects=notable_projects,
        platform_signals=platform_signals,
        education=education
    )
