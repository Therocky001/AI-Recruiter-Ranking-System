import argparse
import json
import csv
import time
import numpy as np
from pathlib import Path
from schema import CandidateProfile
from extract import extract_candidate
from embed import EmbeddingService, get_candidate_summary, get_jd_summary, JDRequirements
from datetime import datetime

def compute_offline_score(sim_score: float, cand: CandidateProfile, jd: JDRequirements) -> float:
    signals = cand.platform_signals
    
    # 1. Skill Assessment Match (0.2)
    skill_score = 0.0
    assessments = signals.get('skill_assessment_scores', {})
    if assessments and isinstance(assessments, dict):
        match_scores = []
        for req_skill in jd.must_have_skills + jd.nice_to_have_skills:
            for ast_skill, ast_score in assessments.items():
                if req_skill.lower() in ast_skill.lower() or ast_skill.lower() in req_skill.lower():
                    match_scores.append(ast_score)
        if match_scores:
            skill_score = sum(match_scores) / len(match_scores) / 100.0
            
    # 2. Engagement Signals (0.2)
    engagement = 0.0
    eng_components = []
    
    resp_rate = signals.get('recruiter_response_rate', -1)
    if resp_rate >= 0: eng_components.append(resp_rate)
        
    interview = signals.get('interview_completion_rate', -1)
    if interview >= 0: eng_components.append(interview)
        
    github = signals.get('github_activity_score', -1)
    if github >= 0: eng_components.append(github / 100.0)
        
    offer = signals.get('offer_acceptance_rate', -1)
    if offer >= 0: eng_components.append(offer)
        
    last_active = signals.get('last_active_date')
    active_days_ago = 0
    if last_active:
        try:
            d = datetime.strptime(last_active, "%Y-%m-%d").date()
            active_days_ago = (datetime.today().date() - d).days
            recency_score = max(-1.0, 1.0 - (active_days_ago / 90.0))
            eng_components.append(recency_score)
        except:
            pass
            
    if eng_components:
        engagement = sum(eng_components) / len(eng_components)
        engagement = max(0.0, min(1.0, engagement))

    # 3. Availability/Trust Signals (0.1)
    trust = 0.0
    trust_components = []
    
    open_to_work = signals.get('open_to_work_flag')
    if open_to_work is True: trust_components.append(1.0)
    elif open_to_work is False: trust_components.append(0.0)
        
    verified = sum([
        1 if signals.get('verified_email') else 0,
        1 if signals.get('verified_phone') else 0,
        1 if signals.get('linkedin_connected') else 0
    ])
    trust_components.append(verified / 3.0)
    
    notice = signals.get('notice_period_days', -1)
    if notice >= 0:
        notice_score = max(0.0, 1.0 - (notice / 60.0))
        trust_components.append(notice_score)
        
    if trust_components:
        trust = sum(trust_components) / len(trust_components)
        
    # --- PENALTIES & DISQUALIFIERS ---
    behavioral_penalty = 1.0
    applied_penalties = []
    
    if active_days_ago > 180 and 0 <= resp_rate < 0.3:
        behavioral_penalty *= 0.5 
        applied_penalties.append("behavioral_downweight")
        
    witch_companies = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]
    is_witch = all(any(w in r.company.lower() for w in witch_companies) for r in cand.past_roles) if cand.past_roles else False
    if is_witch:
        behavioral_penalty *= 0.8
        applied_penalties.append("witch_only")
        
    academic_keywords = ["research", "student", "phd", "academic", "university", "professor"]
    pure_academic = all(any(a in r.title.lower() or a in r.company.lower() for a in academic_keywords) for r in cand.past_roles) if cand.past_roles else False
    if pure_academic:
        behavioral_penalty *= 0.5
        applied_penalties.append("pure_academic")
        
    if cand.past_roles and (cand.total_years_exp / len(cand.past_roles)) < 1.5 and len(cand.past_roles) > 2:
        behavioral_penalty *= 0.85
        applied_penalties.append("job_hopper")
        
    claimed_senior = any("senior" in r.title.lower() or "lead" in r.title.lower() or "manager" in r.title.lower() for r in cand.past_roles)
    if claimed_senior and cand.total_years_exp < 4.0:
        behavioral_penalty *= 0.7
        applied_penalties.append("title_inflation")
        
    # NOTE: final_score computed AFTER all hard filters below
    base_score = (sim_score * 0.5) + (skill_score * 0.2) + (engagement * 0.2) + (trust * 0.1)
    
    cand._tmp_active_days = active_days_ago
    cand._tmp_resp_rate = resp_rate
    cand._tmp_github = github
    cand._tmp_penalty = behavioral_penalty
    cand._tmp_applied_penalties = applied_penalties
    
    # Store data for reasoning
    cand._tmp_title = cand.past_roles[0].title if cand.past_roles else "Candidate"
    
    # Calculate skill overlap string
    overlap_skills = [s for s in cand.core_skills if any(req.lower() in s.lower() for req in jd.must_have_skills + jd.nice_to_have_skills)]
    cand._tmp_overlap = ", ".join(overlap_skills[:2]) if overlap_skills else "general skills"
    
    # --- HARD FILTERS (applied BEFORE final score computation) ---
    # These are multiplicative and ordered from least to most severe.
    
    # 1a. Experience band: JD requires 5-9 yrs; 3-4 yrs = mild flex (0.6x)
    if 3.0 <= cand.total_years_exp < 4.0:
        behavioral_penalty *= 0.6
        applied_penalties.append("exp_low_3_4yr")
        
    # 1b. Experience band: <3 yrs is hard below threshold (0.3x)
    if cand.total_years_exp < 3.0:
        behavioral_penalty *= 0.3
        applied_penalties.append("exp_below_3yr")
    
    # 2. Hedging phrases in any role title or description — soft skill gap signal
    HEDGE_PHRASES = ["still building", "growing into", "learning", "enthusiast"]
    all_role_text = " ".join(
        f"{r.title} {r.description}".lower() for r in cand.past_roles
    )
    if any(phrase in all_role_text for phrase in HEDGE_PHRASES):
        behavioral_penalty *= 0.85
        applied_penalties.append("hedging_language")
    
    # 3. Require vector DB or eval metric mention in career history
    # If neither present → candidate has no evidence of production IR/search work
    VECTOR_DBS = ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch"]
    EVAL_METRICS = ["ndcg", "mrr", "map", "a-b test", "a/b test", "ab test"]
    
    has_vector_db = any(term in all_role_text for term in VECTOR_DBS)
    has_eval_metric = any(term in all_role_text for term in EVAL_METRICS)
    
    # Also check skills list for vector DB names
    all_skills_text = " ".join(cand.core_skills).lower()
    has_vector_db = has_vector_db or any(term in all_skills_text for term in VECTOR_DBS)
    
    if not has_vector_db and not has_eval_metric:
        behavioral_penalty *= 0.75
        applied_penalties.append("no_vector_db_or_eval")
    
    # Apply ALL accumulated penalties to the base score
    final_score = base_score * behavioral_penalty
    
    return round(final_score, 4)

def generate_reasoning(cand: CandidateProfile, score: float) -> str:
    title = getattr(cand, '_tmp_title', "Candidate")
    exp = f"{cand.total_years_exp:.1f}yrs"
    overlap = getattr(cand, '_tmp_overlap', "skills")
    
    active_str = f"active {cand._tmp_active_days} days ago" if getattr(cand, '_tmp_active_days', 0) > 0 else "recently active"
    resp_rate = getattr(cand, '_tmp_resp_rate', -1)
    resp_str = f"{int(resp_rate*100)}% response rate" if resp_rate >= 0 else "unknown response rate"
    
    penalty_list = getattr(cand, '_tmp_applied_penalties', [])
    pen_str = f" [Penalties: {','.join(penalty_list)}]" if penalty_list else ""
    
    return f"{title}, {exp}, {overlap} match; {active_str}, {resp_str}{pen_str}."

def process_batch(batch_lines, embed_service, jd_emb, default_jd):
    candidates = []
    for line in batch_lines:
        if line.strip():
            candidates.append(extract_candidate(json.loads(line)))
            
    if not candidates:
        return [], {}
        
    cand_summaries = [get_candidate_summary(c) for c in candidates]
    cand_embs = embed_service.model.encode(cand_summaries, convert_to_numpy=True, batch_size=256)
    
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(jd_emb.reshape(1, -1), cand_embs)[0]
    
    results = []
    batch_penalties = {}
    
    for i, cand in enumerate(candidates):
        base_sim = float(sims[i])
        final_score = compute_offline_score(base_sim, cand, default_jd)
        reasoning = generate_reasoning(cand, final_score)
        results.append({
            "candidate_id": cand.candidate_id,
            "score": final_score,
            "reasoning": reasoning
        })
        
        for p in getattr(cand, '_tmp_applied_penalties', []):
            batch_penalties[p] = batch_penalties.get(p, 0) + 1
            
    return results, batch_penalties

def main():
    parser = argparse.ArgumentParser(description="Offline Candidate Ranking")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Path to output CSV file")
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Initializing offline embedding service...")
    embed_service = EmbeddingService()
    
    default_jd = JDRequirements(
        role_title="AI Engineer",
        must_have_skills=["Python", "Machine Learning", "Data Science"],
        nice_to_have_skills=["PyTorch", "NLP", "Spark"],
        min_years_exp=3.0,
        domain="Technology",
        seniority_level="Mid-Level",
        soft_signals=["Leadership", "Communication"]
    )
    jd_summary = get_jd_summary(default_jd)
    jd_emb = embed_service.model.encode([jd_summary], convert_to_numpy=True)[0]
    
    print(f"Streaming candidates from {args.candidates}...")
    all_scored_candidates = []
    batch_lines = []
    batch_size = 2000
    processed_count = 0
    global_penalties = {}
    
    with open(args.candidates, 'r', encoding='utf-8') as f:
        for line in f:
            batch_lines.append(line)
            if len(batch_lines) >= batch_size:
                results, penalties = process_batch(batch_lines, embed_service, jd_emb, default_jd)
                all_scored_candidates.extend(results)
                for p, count in penalties.items():
                    global_penalties[p] = global_penalties.get(p, 0) + count
                processed_count += len(batch_lines)
                print(f"Processed {processed_count} candidates... (Elapsed: {time.time()-start_time:.1f}s)")
                batch_lines = []
                
        # Process remaining
        if batch_lines:
            results, penalties = process_batch(batch_lines, embed_service, jd_emb, default_jd)
            all_scored_candidates.extend(results)
            for p, count in penalties.items():
                global_penalties[p] = global_penalties.get(p, 0) + count
            processed_count += len(batch_lines)
            
    print(f"Total candidates processed: {processed_count}")
    print("--- Penalty Triggers Report ---")
    for p, count in global_penalties.items():
        print(f"  {p}: {count} candidates penalized")
    print("-------------------------------")
    
    print("Sorting all candidates...")
    all_scored_candidates.sort(key=lambda x: (-x['score'], x['candidate_id']))
    
    top_100 = all_scored_candidates[:100]
    
    print(f"Writing top {len(top_100)} candidates to {args.out}...")
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, cand in enumerate(top_100, start=1):
            writer.writerow([
                cand["candidate_id"],
                rank,
                f"{cand['score']:.4f}",
                cand["reasoning"]
            ])
            
    end_time = time.time()
    print(f"Done! Total Runtime: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
