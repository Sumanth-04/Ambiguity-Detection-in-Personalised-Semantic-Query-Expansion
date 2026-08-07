from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ambiguity.baseScore import base_score
from ambiguity.userProfileMatching import final_ambiguity_score
from ambiguity.valuation import get_wordnet_ambiguity

app = FastAPI(title="PSemQE Ambiguity Detection API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load profile embeddings once at startup
profile_embeddings = np.load(
    "embeddings/profile_embeddings.npy",
    allow_pickle=True
).item()

available_topics = list(profile_embeddings.keys())


# Request model
class QueryRequest(BaseModel):
    query: str
    topic_id: int


# Response model
class QueryResponse(BaseModel):
    query: str
    topic_id: int
    base_score: float
    final_score: float
    is_ambiguous: bool
    ambiguity_level: str
    wordnet_label: str
    wordnet_avg_synsets: float
    wordnet_is_ambiguous: bool


def get_ambiguity_level(score: float) -> str:
    if score >= 0.6:
        return "HIGH"
    elif score >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"


@app.get("/")
def root():
    return {
        "message": "PSemQE Ambiguity Detection API",
        "available_topics": available_topics
    }


@app.get("/topics")
def get_topics():
    return {
        "available_topic_ids": available_topics,
        "total": len(available_topics)
    }

@app.post("/detect", response_model=QueryResponse)
def detect_ambiguity(request: QueryRequest):
    if request.topic_id not in profile_embeddings:
        raise HTTPException(status_code=404, detail=f"Topic ID {request.topic_id} not found.")
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    b_score   = base_score(request.query)
    f_score   = final_ambiguity_score(b_score, request.query, request.topic_id)
    wordnet   = get_wordnet_ambiguity(request.query)

    return QueryResponse(
        query = request.query,
        topic_id = request.topic_id,
        base_score = b_score,
        final_score = f_score,
        is_ambiguous = f_score >= 0.2,
        ambiguity_level = get_ambiguity_level(f_score),
        wordnet_label = wordnet["wordnet_label"],
        wordnet_avg_synsets = wordnet["avg_synsets"],
        wordnet_is_ambiguous = wordnet["is_ambiguous"],
    )