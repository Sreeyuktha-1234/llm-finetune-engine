"""
FastAPI service exposing model loading and text generation endpoints.
"""

from threading import Lock
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipelines.inference_pipeline import InferencePipeline


app = FastAPI(title="LLM Finetune Engine API", version="1.0.0")

_pipeline_lock = Lock()
_pipeline: Optional[InferencePipeline] = None


class LoadModelRequest(BaseModel):
    """Request body for loading a model into memory."""

    model_name: str = Field(default="gpt2", min_length=1)
    device: Optional[str] = Field(default=None, description="cpu/cuda or null for auto")
    max_length: int = Field(default=100, ge=1)


class LoadModelResponse(BaseModel):
    """Response body returned after model load."""

    message: str
    model_name: str
    device: str
    max_length: int


class GenerateRequest(BaseModel):
    """Request body for text generation."""

    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=50, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    num_return_sequences: int = Field(default=1, ge=1, le=16)


class GenerateResponse(BaseModel):
    """Response body for generated text."""

    model_name: str
    generations: List[str]


@app.get("/health")
def health() -> dict:
    """Simple liveness check endpoint."""

    return {"status": "ok"}


@app.post("/load-model", response_model=LoadModelResponse)
def load_model(request: LoadModelRequest) -> LoadModelResponse:
    """Load or replace the currently active inference model."""

    global _pipeline
    try:
        with _pipeline_lock:
            _pipeline = InferencePipeline(
                model_name=request.model_name,
                device=request.device,
                max_length=request.max_length,
            )
            info = _pipeline.get_pipeline_info()

        return LoadModelResponse(
            message="Model loaded successfully",
            model_name=info.get("model_name", request.model_name),
            device=info.get("device", request.device or "auto"),
            max_length=info.get("max_length", request.max_length),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {exc}") from exc


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate text using the currently loaded model."""

    with _pipeline_lock:
        active_pipeline = _pipeline

    if active_pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Call POST /load-model first.",
        )

    try:
        outputs = active_pipeline.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_return_sequences,
        )
        return GenerateResponse(model_name=active_pipeline.model_name, generations=outputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {exc}") from exc
