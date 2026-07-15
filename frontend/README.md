# Emo Agent Frontend

React + Vite frontend for the Module 7 local FastAPI demo.

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
