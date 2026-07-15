"""FastAPI entrypoint for the Module 7 local backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent_service import EmotionCompanionAgent
from backend.backend_service import INDEX_HTML, analyze_audio_bytes, build_health_status
from backend.runtime_pipeline import LazyFullAudioAnalyzer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def create_app(
    analyzer: Any | None = None,
    agent: EmotionCompanionAgent | None = None,
    frontend_dist: str | Path | None = DEFAULT_FRONTEND_DIST,
) -> FastAPI:
    analyzer = analyzer or LazyFullAudioAnalyzer()
    agent = agent or EmotionCompanionAgent()
    app = FastAPI(title="Emo Agent Local Backend", version="0.1.0")
    frontend_dist_path = Path(frontend_dist) if frontend_dist is not None else None
    frontend_index = frontend_dist_path / "index.html" if frontend_dist_path is not None else None

    if frontend_dist_path is not None and (frontend_dist_path / "assets").exists():
        app.mount("/assets", StaticFiles(directory=frontend_dist_path / "assets"), name="frontend-assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> Any:
        if frontend_index is not None and frontend_index.exists():
            return FileResponse(frontend_index)
        return INDEX_HTML

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return build_health_status(agent.config)

    @app.post("/api/analyze-audio", response_model=None)
    async def analyze_audio(request: Request, file: UploadFile | None = File(None)) -> Any:
        try:
            if file is not None:
                audio_bytes = await file.read()
                filename = file.filename
                content_type = file.content_type
            else:
                audio_bytes = await request.body()
                filename = request.headers.get("X-Filename")
                content_type = request.headers.get("Content-Type")
            return analyze_audio_bytes(
                audio_bytes,
                filename=filename,
                content_type=content_type,
                analyzer=analyzer,
                agent=agent,
            )
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                },
            )

    @app.get("/{path:path}", response_class=HTMLResponse)
    def frontend_fallback(path: str) -> Any:
        if path.startswith("api/"):
            return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
        if frontend_index is not None and frontend_index.exists():
            return FileResponse(frontend_index)
        return INDEX_HTML

    return app


app = create_app()
