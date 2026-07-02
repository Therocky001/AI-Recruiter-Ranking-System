import json
import os
import numpy as np
from extract import extract_candidate
from embed import get_candidate_summary, EmbeddingService

DATA_PATH = r"..\Data\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_candidates.json"
CACHE_JSON = "cached_candidates.json"
CACHE_NPY = "embeddings.npy"
MAX_CANDIDATES = 500

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find dataset at {DATA_PATH}")
        return
        
    print("Loading raw candidates...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        raw_candidates = json.load(f)
        
    print(f"Loaded {len(raw_candidates)} total candidates. Limiting to {MAX_CANDIDATES} for demo.")
    raw_candidates = raw_candidates[:MAX_CANDIDATES]
    
    print("Parsing candidates into structured schema...")
    structured_candidates = []
    for raw in raw_candidates:
        cand = extract_candidate(raw)
        structured_candidates.append(cand)
        
    print("Saving structured candidates to cache...")
    with open(CACHE_JSON, 'w', encoding='utf-8') as f:
        # Pydantic v2 uses model_dump(), v1 uses dict(). We will use model_dump()
        json.dump([c.model_dump() for c in structured_candidates], f, indent=2)
        
    print("Generating summaries for embedding...")
    summaries = [get_candidate_summary(c) for c in structured_candidates]
    
    print("Initializing embedding service (this may download the model)...")
    service = EmbeddingService()
    
    print("Computing embeddings...")
    embeddings = service.embed_texts(summaries)
    
    print("Saving embeddings to cache...")
    np.save(CACHE_NPY, embeddings)
    
    print("Preprocessing complete!")

if __name__ == "__main__":
    main()
