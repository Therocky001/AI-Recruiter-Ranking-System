from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class JDRequirements(BaseModel):
    role_title: str
    must_have_skills: List[str]
    nice_to_have_skills: List[str]
    min_years_exp: float
    domain: str
    seniority_level: str
    soft_signals: List[str] = Field(description="e.g. leadership, ownership, communication")

class Role(BaseModel):
    title: str
    company: str
    description: str

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    total_years_exp: float
    core_skills: List[str]
    domains: List[str]
    past_roles: List[Role]
    notable_projects: List[str]
    platform_signals: Dict[str, Any]
    education: List[str]
