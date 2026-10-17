@echo off
echo ===================================================
echo   Starting RAG Studio (Backend ^& Frontend)
echo ===================================================
echo.
echo 1. Starting FastAPI Backend on http://127.0.0.1:8000
start cmd /k "python api.py"

echo 2. Starting Vite Frontend on http://localhost:5173
cd frontend
start cmd /k "npm run dev"

echo.
echo Both servers have been launched in separate terminal windows.
echo Please open http://localhost:5173/ in your browser once loaded.
echo.
pause
