from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")


def _load_config() -> dict[str, Any]:
    config_path = Path(os.getenv("VLLM_CONFIG_FILE", DEFAULT_CONFIG_PATH))
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except FileNotFoundError as error:
        raise RuntimeError(f"Config file not found: {config_path}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON in config file: {config_path}") from error

    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise RuntimeError("Config must contain a string 'model' field")
    if not isinstance(config.get("vllm", {}), dict):
        raise RuntimeError("Config 'vllm' field must be an object")
    return config


CONFIG = _load_config()
MODEL_NAME = CONFIG["model"]


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    model: str = MODEL_NAME
    max_tokens: int | None = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False


class CompletionRequest(BaseModel):
    prompt: str | list[str]
    model: str = MODEL_NAME
    max_tokens: int | None = Field(default=512, ge=1)
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False


def _load_engine() -> Any:
    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_config = {"model": MODEL_NAME, **CONFIG.get("vllm", {})}
    return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_config))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.engine = _load_engine()
    yield
    shutdown = getattr(app.state.engine, "shutdown_background_loop", None)
    if shutdown is not None:
        shutdown()


app = FastAPI(title="Gemma vLLM Server", version="1.0.0", lifespan=lifespan)


def _sampling_params(request: ChatCompletionRequest | CompletionRequest) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )


async def _generate(engine: Any, prompt: str, request: Any) -> str:
    request_id = f"cmpl-{uuid4().hex}"
    result = None
    async for result in engine.generate(prompt, _sampling_params(request), request_id):
        pass
    if result is None or not result.outputs:
        return ""
    return result.outputs[0].text


def _check_model(model: str) -> None:
    if model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"Only {MODEL_NAME} is served")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models")
async def models() -> dict[str, list[dict[str, str]]]:
    return {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
    _check_model(request.model)
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented")

    engine = app.state.engine
    prompt = "\n".join(f"{message.role}: {message.content}" for message in request.messages)
    text = await _generate(engine, prompt, request)
    timestamp = int(time())
    return {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": timestamp,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest) -> dict[str, Any]:
    _check_model(request.model)
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not implemented")

    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    texts = [await _generate(app.state.engine, prompt, request) for prompt in prompts]
    return {
        "id": f"cmpl-{uuid4().hex}",
        "object": "text_completion",
        "created": int(time()),
        "model": MODEL_NAME,
        "choices": [{"index": index, "text": text, "finish_reason": "stop"} for index, text in enumerate(texts)],
    }