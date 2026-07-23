from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Sentiment Sense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

zero_shot_pipe = pipeline(
    "zero-shot-classification",
    model="typeform/distilbert-base-uncased-mnli",
)

CANDIDATE_LABELS = ["Food", "Service", "Price", "Atmosphere", "Health"]


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    text = request.text

    topic_results = zero_shot_pipe(text, CANDIDATE_LABELS, multi_label=True)
    aspects = [
        {"label": label, "score": round(float(score), 4)}
        for label, score in zip(topic_results["labels"], topic_results["scores"])
        if score > 0.6
    ]

    sentiment_result = sentiment_pipe(text)[0]

    return {
        "aspects": aspects,
        "sentiment": {
            "label": sentiment_result["label"],
            "confidence": round(float(sentiment_result["score"]), 4),
        },
    }