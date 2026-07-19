# Emo Agent Frontend

Product frontend built with Vite, React, TypeScript/TSX, Tailwind CSS and Radix-style components. The visual direction is based on the Figma Make export, while all displayed analysis data comes from the real FastAPI backend.

The conversation input keeps two first-class audio entry points:

- microphone recording;
- upload an existing `.wav`, `.mp3`, `.m4a`, `.flac`, or `.webm` file.

Both call `POST /api/analyze-audio`. Uploads try multipart first and fall back to a raw request body with `Content-Type` and `X-Filename` when needed.

Text messages call `POST /api/agent/chat/stream` and consume SSE events for incremental replies. If streaming is unavailable, the frontend falls back to `POST /api/agent/chat`. The backend uses the DeepSeek tool-calling loop when configured and a local safe response when no API key is available.

## Install

```powershell
npm install --cache .npm-cache
```

Dependencies are recorded in `package.json` and locked in `package-lock.json`.

## Development

Start the backend first:

```powershell
D:\Anaconda\envs\emotion-agent\python.exe -m backend.run_fastapi --host 127.0.0.1 --port 7860
```

Then start Vite:

```powershell
npm run dev
```

Vite proxies `/api/*` to `http://127.0.0.1:7860`.

## Build for FastAPI Hosting

```powershell
npm run build
```

The build output is written to `frontend/dist/`. FastAPI serves that directory at `/` when it exists.
