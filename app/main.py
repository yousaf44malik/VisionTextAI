# app/main.py
from fastapi import FastAPI
from app.api.v1 import nlp, image, hybrid

app = FastAPI(title="VisionText AI")

# Include API routers
app.include_router(nlp.router, prefix="/api/v1/nlp", tags=["NLP"])
app.include_router(image.router, prefix="/api/v1/image", tags=["CV"])
app.include_router(hybrid.router, prefix="/api/v1/hybrid", tags=["Hybrid"])
