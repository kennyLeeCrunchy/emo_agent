"""FastAPI entrypoint for the Module 7 local backend."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.agent_service import EmotionCompanionAgent
from backend.agent_orchestrator import AgentOrchestrator, ContextCompactionUnavailable
from backend.backend_service import INDEX_HTML, analyze_audio_bytes, build_health_status
from backend.memory_service import MemoryService
from backend.runtime_pipeline import LazyFullAudioAnalyzer
from backend.security import (
    PersistentDailyQuota,
    SecuritySettings,
    SlidingWindowLimiter,
    proxy_is_authorized,
    trusted_user_id,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def create_app(
    analyzer: Any | None = None,
    agent: EmotionCompanionAgent | None = None,
    orchestrator: AgentOrchestrator | None = None,
    memory: MemoryService | None = None,
    frontend_dist: str | Path | None = DEFAULT_FRONTEND_DIST,
    security: SecuritySettings | None = None,
) -> FastAPI:
    security = security or SecuritySettings.from_env()
    security.validate()
    analyzer = analyzer or LazyFullAudioAnalyzer()
    agent = agent or EmotionCompanionAgent()
    memory = memory or (orchestrator.memory if orchestrator is not None else MemoryService.from_env())
    orchestrator = orchestrator or AgentOrchestrator(agent=agent, memory=memory)
    orchestrator.max_tool_rounds = min(orchestrator.max_tool_rounds, security.llm_max_tool_rounds)
    app = FastAPI(title="Emo Agent Local Backend", version="0.2.0")
    limiter = SlidingWindowLimiter()
    llm_quota = PersistentDailyQuota(security.llm_budget_db)
    analysis_slots = asyncio.Semaphore(security.max_concurrent_analysis)
    frontend_dist_path = Path(frontend_dist) if frontend_dist is not None else None
    frontend_index = frontend_dist_path / "index.html" if frontend_dist_path is not None else None

    @app.middleware("http")
    async def public_security(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/api/") and request.url.path != "/api/health":
            if not proxy_is_authorized(request, security):
                return JSONResponse(status_code=401, content={"ok": False, "error": "unauthorized"})
            if security.public_mode and not request.headers.get("X-Emo-User-Id", "").strip():
                return JSONResponse(status_code=401, content={"ok": False, "error": "trusted_identity_missing"})
            identity = request.headers.get("X-Emo-User-Id") or (
                request.client.host if request.client else "unknown"
            )
            expensive = request.url.path in {"/api/analyze-audio", "/api/agent/chat", "/api/agent/chat/stream"}
            limit = security.expensive_requests_per_minute if expensive else security.requests_per_minute
            if not limiter.allow(f"{identity}:{request.url.path}", limit):
                return JSONResponse(
                    status_code=429,
                    content={"ok": False, "error": "rate_limited", "message": "Too many requests"},
                    headers={"Retry-After": "60"},
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), geolocation=()"
        response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
        return response

    def request_user(request: Request, claimed: object = None) -> str:
        try:
            return trusted_user_id(request, security, claimed)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    def reserve_llm_budget(user_id: str, cost: int | None = None) -> bool:
        """Reserve bounded credits before a paid DeepSeek request leaves this machine."""
        if not agent.config.configured:
            return True
        return llm_quota.allow_many(
            {
                "global": security.llm_upstream_calls_per_day,
                f"user:{user_id}": security.llm_upstream_calls_per_user_per_day,
            },
            cost=cost or security.llm_max_tool_rounds,
        )

    if frontend_dist_path is not None and (frontend_dist_path / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist_path / "assets"), name="frontend-assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        if frontend_index is not None and frontend_index.exists():
            return FileResponse(frontend_index)
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        status = build_health_status(agent.config)
        status["memory"] = {"configured": True, "backend": "sqlite"}
        status["public_security"] = {
            "enabled": security.public_mode,
            "upload_limit_bytes": security.max_upload_bytes,
        }
        return status

    @app.post("/api/analyze-audio", response_model=None)
    async def analyze_audio(request: Request, file: UploadFile | None = File(None)) -> Any:
        try:
            content_length = request.headers.get("Content-Length")
            if content_length and int(content_length) > security.max_upload_bytes + 64 * 1024:
                return JSONResponse(status_code=413, content={"ok": False, "error": "upload_too_large"})
            if file is not None:
                audio_bytes = await file.read(security.max_upload_bytes + 1)
                filename = file.filename
                content_type = file.content_type
            else:
                audio_bytes = await request.body()
                filename = request.headers.get("X-Filename")
                content_type = request.headers.get("Content-Type")
            if len(audio_bytes) > security.max_upload_bytes:
                return JSONResponse(status_code=413, content={"ok": False, "error": "upload_too_large"})
            allowed_types = {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/flac", "audio/x-flac", "audio/webm", "application/octet-stream"}
            normalized_type = str(content_type or "application/octet-stream").split(";", 1)[0].lower()
            if normalized_type not in allowed_types:
                return JSONResponse(status_code=415, content={"ok": False, "error": "unsupported_audio_type"})
            user_id = request_user(request)
            if not reserve_llm_budget(user_id, cost=security.llm_max_tool_rounds):
                return JSONResponse(
                    status_code=429,
                    content={"ok": False, "error": "llm_daily_quota_exceeded", "message": "Daily trial quota reached"},
                )
            async with analysis_slots:
                result = await asyncio.to_thread(
                    analyze_audio_bytes,
                    audio_bytes,
                    filename=filename,
                    content_type=content_type,
                    analyzer=analyzer,
                )
            if isinstance(result, dict) and result.get("ok"):
                analysis = result.get("analysis") or {}
                chat_result = await asyncio.to_thread(
                    orchestrator.chat_from_audio,
                    analysis,
                    session_id=request.headers.get("X-Session-Id"),
                    user_id=user_id,
                    request_id=request.headers.get("X-Request-Id"),
                )
                result["agent_feedback"] = {
                    "companion_response": chat_result.response,
                    "provider": chat_result.provider,
                }
                result["session_id"] = chat_result.session_id
            return result
        except ContextCompactionUnavailable as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": "memory_compaction_unavailable",
                    "message": "记忆整理服务暂时不可用，请10分钟后重新发送消息",
                    "retry_after_seconds": exc.retry_after_seconds,
                },
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": "Audio analysis failed" if security.public_mode else str(exc),
                },
            )

    @app.post("/api/agent/chat", response_model=None)
    def agent_chat(request: Request, payload: dict[str, Any] = Body(...)) -> Any:
        try:
            message = str(payload.get("message") or "")
            if len(message.strip()) > agent.config.max_input_chars:
                return JSONResponse(status_code=422, content={"ok": False, "error": "message_too_long"})
            user_id = request_user(request, payload.get("user_id"))
            quota_cost = (
                security.llm_web_search_cost
                if (
                    orchestrator._requires_web_search(message)
                    or orchestrator._is_information_request(message)
                )
                else security.llm_max_tool_rounds
            )
            if not reserve_llm_budget(user_id, cost=quota_cost):
                return JSONResponse(
                    status_code=429,
                    content={"ok": False, "error": "llm_daily_quota_exceeded", "message": "Daily trial quota reached"},
                )
            result = orchestrator.chat(
                message=message,
                session_id=payload.get("session_id"),
                user_id=user_id,
                request_id=payload.get("request_id"),
                history=payload.get("history") or [],
                latest_analysis=payload.get("latest_analysis") or {},
            )
            return result.to_dict()
        except ContextCompactionUnavailable as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": "memory_compaction_unavailable",
                    "message": "记忆整理服务暂时不可用，请10分钟后重新发送消息",
                    "retry_after_seconds": exc.retry_after_seconds,
                },
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"ok": False, "error": "invalid_request", "message": str(exc)})
        except Exception:
            return JSONResponse(status_code=500, content={"ok": False, "error": "agent_unavailable", "message": "文字陪伴服务暂时不可用"})

    @app.post("/api/agent/chat/stream", response_model=None)
    def agent_chat_stream(request: Request, payload: dict[str, Any] = Body(...)) -> Any:
        message = str(payload.get("message") or "").strip()
        if not message:
            return JSONResponse(status_code=422, content={"ok": False, "error": "invalid_request", "message": "message must not be empty"})
        if len(message) > agent.config.max_input_chars:
            return JSONResponse(status_code=422, content={"ok": False, "error": "message_too_long"})
        user_id = request_user(request, payload.get("user_id"))
        quota_cost = (
            security.llm_web_search_cost
            if (
                orchestrator._requires_web_search(message)
                or orchestrator._is_information_request(message)
            )
            else security.llm_max_tool_rounds
        )
        if not reserve_llm_budget(user_id, cost=quota_cost):
            return JSONResponse(
                status_code=429,
                content={"ok": False, "error": "llm_daily_quota_exceeded", "message": "Daily trial quota reached"},
            )
        try:
            result = orchestrator.chat(
                message=message,
                session_id=payload.get("session_id"),
                user_id=user_id,
                request_id=payload.get("request_id"),
                history=payload.get("history") or [],
                latest_analysis=payload.get("latest_analysis") or {},
            )
        except ContextCompactionUnavailable as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "error": "memory_compaction_unavailable",
                    "message": "记忆整理服务暂时不可用，请10分钟后重新发送消息",
                    "retry_after_seconds": exc.retry_after_seconds,
                },
                headers={"Retry-After": str(exc.retry_after_seconds)},
            )
        return StreamingResponse(
            orchestrator.stream_result(result),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/sessions")
    def list_sessions(request: Request, user_id: str = "local-user") -> dict[str, Any]:
        user_id = request_user(request, user_id)
        return {"ok": True, "sessions": memory.list_sessions(user_id)}

    @app.get("/api/sessions/{session_id}")
    def get_session(request: Request, session_id: str, user_id: str = "local-user") -> Any:
        user_id = request_user(request, user_id)
        session = memory.get_session(user_id, session_id)
        if not session:
            return JSONResponse(status_code=404, content={"ok": False, "error": "session_not_found"})
        return {
            "ok": True,
            "session": session,
            "messages": memory.get_messages(user_id, session_id, limit=200),
        }

    @app.delete("/api/sessions/{session_id}")
    def delete_session(request: Request, session_id: str, user_id: str = "local-user") -> Any:
        user_id = request_user(request, user_id)
        if not memory.delete_session(user_id, session_id):
            return JSONResponse(status_code=404, content={"ok": False, "error": "session_not_found"})
        return {"ok": True, "deleted": True, "session_id": session_id}

    @app.get("/api/memory/summary")
    def memory_summary(request: Request, user_id: str = "local-user", session_id: str | None = None) -> dict[str, Any]:
        user_id = request_user(request, user_id)
        return {"ok": True, **memory.summary(user_id, session_id)}

    @app.post("/api/memory/events", response_model=None)
    def memory_event(request: Request, payload: dict[str, Any] = Body(...)) -> Any:
        user_id = request_user(request, payload.get("user_id"))
        session_id = payload.get("session_id")
        if session_id:
            try:
                memory.ensure_session(user_id, session_id)
            except ValueError as exc:
                return JSONResponse(status_code=403, content={"ok": False, "error": "session_forbidden", "message": str(exc)})
        result = memory.write_memory(user_id, session_id, payload)
        status_code = 200 if result.get("ok") else 422
        return JSONResponse(status_code=status_code, content=result)

    @app.delete("/api/memory")
    def delete_memory(request: Request, user_id: str = "local-user") -> dict[str, Any]:
        user_id = request_user(request, user_id)
        return {"ok": True, "deleted": memory.delete_user_memory(user_id)}

    @app.get("/api/emotion/curve")
    def emotion_curve(
        request: Request,
        scope: str,
        user_id: str = "local-user",
        session_id: str | None = None,
        range: str = "month",
    ) -> Any:
        user_id = request_user(request, user_id)
        if scope == "memory":
            return {"ok": True, **memory.emotion_trend(user_id, range)}
        if scope == "session" and session_id:
            session = memory.get_session(user_id, session_id)
            if not session:
                return JSONResponse(status_code=404, content={"ok": False, "error": "session_not_found"})
            analysis = session.get("latest_analysis") or {}
            return {
                "ok": True,
                "scope": "session_curve",
                "available": bool(analysis.get("emotion_curve")),
                "points": analysis.get("emotion_curve") or [],
                "change_points": analysis.get("emotion_change_points") or [],
            }
        return JSONResponse(status_code=422, content={"ok": False, "error": "invalid_curve_scope"})

    @app.get("/{path:path}", response_class=HTMLResponse)
    def frontend_fallback(path: str) -> Any:
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
        if frontend_index is not None and frontend_index.exists():
            return FileResponse(frontend_index)
        return INDEX_HTML

    return app


app = create_app()
