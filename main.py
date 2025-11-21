import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import math

app = FastAPI(title="ClamSense API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response

# ----- Core APIs for prototype -----

# PSS-10 scoring
REVERSE_SCORED = {3, 4, 6, 7}  # 0-indexed, per PSS-10 convention

class PSS10Request(BaseModel):
    answers: List[int] = Field(..., description="10 answers, each 0..4", min_items=10, max_items=10)

class PSS10Response(BaseModel):
    score: int
    band: str
    explanation: str

@app.post("/survey/pss10/score", response_model=PSS10Response)
def pss10_score(req: PSS10Request):
    if any(a not in [0,1,2,3,4] for a in req.answers):
        # FastAPI will turn this into a 422 if we raise ValueError; simpler: clamp defensively
        answers = [min(4, max(0, int(a))) for a in req.answers]
    else:
        answers = req.answers

    score = 0
    for i, a in enumerate(answers):
        score += (4 - a) if i in REVERSE_SCORED else a

    # Rough banding (0-40 scale)
    band = "low" if score <= 13 else "moderate" if score <= 26 else "high"
    explanation = (
        "Low perceived stress" if band == "low" else 
        "Moderate perceived stress—consider regular check-ins" if band == "moderate" else 
        "High perceived stress—try immediate coping tools and consider professional support"
    )
    return PSS10Response(score=score, band=band, explanation=explanation)

# Simple prediction (placeholder without external model file)
class PredictRequest(BaseModel):
    heart_rate: float = Field(..., gt=0, description="Current or recent heart rate in bpm")
    sleep_hours: float = Field(..., ge=0, le=14, description="Sleep duration in last night")
    steps: int = Field(..., ge=0, description="Steps so far today")
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon .. 6=Sun")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (24h)")
    mood_score: float = Field(..., ge=0, le=1, description="Self-reported mood 0..1 (1 best)")
    pss10_score: Optional[int] = Field(None, ge=0, le=40, description="Optional baseline from PSS-10")

class PredictResponse(BaseModel):
    predicted_level: float
    risk_band: str
    factors: List[str]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Heuristic model for demo: combine indicators into a 0..1 risk index
    # Higher HR -> higher stress; less sleep -> higher stress; low mood -> higher stress.
    hr_component = min(1.0, max(0.0, (req.heart_rate - 55) / 55))  # ~0 at 55bpm, ~1 at 110bpm
    sleep_component = 1.0 - min(1.0, req.sleep_hours / 8.0)  # 0 when 8h sleep, up to 1 when 0h
    steps_component = max(0.0, 1.0 - min(1.0, req.steps / 8000)) * 0.4  # very low activity may increase stress
    mood_component = 1.0 - req.mood_score  # 1 when mood 0, 0 when mood 1
    circadian_component = 0.15 if req.hour in [10, 11, 15, 16] else 0.0  # common peak windows
    baseline_component = 0.0
    if req.pss10_score is not None:
        baseline_component = min(1.0, req.pss10_score / 40.0) * 0.3

    risk = 0.35*hr_component + 0.25*sleep_component + 0.15*mood_component + 0.10*steps_component + circadian_component + baseline_component
    risk = max(0.0, min(1.0, risk))

    band = "low" if risk < 0.33 else "moderate" if risk < 0.66 else "high"

    factors = []
    if hr_component > 0.6: factors.append("elevated heart rate")
    if sleep_component > 0.6: factors.append("short sleep duration")
    if mood_component > 0.6: factors.append("low self-reported mood")
    if steps_component > 0.4: factors.append("very low activity")
    if circadian_component > 0: factors.append("typical peak hours")
    if baseline_component > 0.15: factors.append("high baseline stress")

    return PredictResponse(predicted_level=round(risk, 3), risk_band=band, factors=factors)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
