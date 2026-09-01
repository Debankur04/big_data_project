from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from route import health, predict


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="News Truth & Topic Analyzer API",
    description=(
        "Fake/Real news classification and "
        "LDA topic modeling API."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictRequest(BaseModel):

    text: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service":
            "News Truth & Topic Analyzer",

        "status":
            "running",

        "version":
            "1.0.0"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def api_health():

    try:

        result = health()

        return result

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def api_predict(request: PredictRequest):

    try:

        result = predict(
            request.text
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except TypeError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )