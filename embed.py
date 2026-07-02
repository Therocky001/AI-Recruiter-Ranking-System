import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from schema import JDRequirements, CandidateProfile
from typing import List, Tuple

def get_jd_summary(jd: JDRequirements) -> str:
    """Create a textual summary of the JD for embedding."""
    skills = ", ".join(jd.must_have_skills + jd.nice_to_have_skills)
    soft = ", ".join(jd.soft_signals)
    return (
        f"Role: {jd.role_title}. Domain: {jd.domain}. "
        f"Seniority: {jd.seniority_level} with {jd.min_years_exp}+ years of experience. "
        f"Required Skills: {skills}. "
        f"Soft Skills: {soft}."
    )

def get_candidate_summary(cand: CandidateProfile) -> str:
    """Create a textual summary prioritizing role history, projects, and seniority."""
    skills_subset = ", ".join(cand.core_skills[:5]) # Don't just stuff all skills
    roles = " | ".join([f"{r.title} at {r.company}: {r.description}" for r in cand.past_roles])
    return (
        f"Candidate: {cand.name}. Experience: {cand.total_years_exp} years. "
        f"Key Skills: {skills_subset}. "
        f"Role History & Projects: {roles} "
    )

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
        
def get_top_candidates(jd_emb: np.ndarray, cand_embs: np.ndarray, cand_ids: List[str], top_k: int = 25) -> List[Tuple[str, float]]:
    """Returns list of (candidate_id, similarity_score) sorted by score desc."""
    jd_emb = jd_emb.reshape(1, -1)
    
    # Compute cosine similarity
    sims = cosine_similarity(jd_emb, cand_embs)[0]
    
    # Get top K indices
    top_indices = np.argsort(sims)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append((cand_ids[idx], float(sims[idx])))
        
    return results
