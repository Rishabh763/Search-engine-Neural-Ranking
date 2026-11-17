@echo off
REM Live demo: the full-corpus search API on this laptop, reachable from the deployed frontend.
REM
REM   scripts\start_demo.bat https://<your-app>.vercel.app
REM
REM Opens two windows: the API (uvicorn), then a Cloudflare quick tunnel once the API answers.
REM The tunnel window prints https://<random-words>.trycloudflare.com. Share:
REM   https://<your-app>.vercel.app/?api=https://<random-words>.trycloudflare.com
REM Close both windows to stop. The tunnel address changes every time.

setlocal
if "%~1"=="" (
  echo Usage: scripts\start_demo.bat https://your-app.vercel.app
  exit /b 1
)
set "ORIGIN=%~1"
if "%ORIGIN:~-1%"=="/" set "ORIGIN=%ORIGIN:~0,-1%"

cd /d "%~dp0.."

REM cloudflared: installed (on PATH) or the downloaded .exe placed in the project folder.
set "CLOUDFLARED=cloudflared"
if exist "cloudflared-windows-amd64.exe" set "CLOUDFLARED=%CD%\cloudflared-windows-amd64.exe"
if exist "cloudflared.exe" set "CLOUDFLARED=%CD%\cloudflared.exe"
"%CLOUDFLARED%" --version >nul 2>&1
if errorlevel 1 (
  echo cloudflared not found. Either run: winget install --id Cloudflare.cloudflared
  echo and open a new terminal, or download cloudflared-windows-amd64.exe from
  echo https://github.com/cloudflare/cloudflared/releases/latest into this project folder.
  exit /b 1
)

call .venv\Scripts\activate.bat

set SEARCH_COLLECTION=msmarco-full
set SEARCH_BACKEND=dense
set DENSE_INDEX=ivfpq96
set DENSE_NPROBE=1024
set DENSE_REFINE=500
set RERANK_MODEL=models\ms-marco-MiniLM-L-6-v2-onnx
set RERANK_ONNX_VARIANT=fp16
set RERANK_DEVICE=dml
set CORS_ORIGINS=%ORIGIN%,http://localhost:5173

start "Search API" cmd /k uvicorn searchengine.api:app --port 8000

echo Starting the search API (loading the index and the reranker)...
:wait
timeout /t 3 /nobreak >nul
curl -s -o nul http://127.0.0.1:8000/health
if errorlevel 1 goto wait
echo The API is up.

start "Tunnel" cmd /k ""%CLOUDFLARED%" tunnel --url http://localhost:8000"

echo.
echo In the Tunnel window, find the line with https://...trycloudflare.com and share:
echo   %ORIGIN%/?api=https://YOUR-TUNNEL.trycloudflare.com
echo Check it first: open that link and look for "Connected" in the Connection panel.
endlocal
