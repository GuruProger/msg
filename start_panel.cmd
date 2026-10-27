@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
 set "PANEL_PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
 set "PANEL_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
) else (
 set "PANEL_PYTHON=python"
)
set "PYTHONPATH=%~dp0.ui-deps;%PYTHONPATH%"
"%PANEL_PYTHON%" -c "import fastapi, uvicorn, httpx" >nul 2>&1
if errorlevel 1 (
 "%PANEL_PYTHON%" -m pip --isolated install --target "%~dp0.ui-deps" --upgrade --index-url https://pypi.org/simple --disable-pip-version-check -r requirements.frontend.txt
 if errorlevel 1 goto done
)
echo Panel: http://127.0.0.1:8080/
"%PANEL_PYTHON%" launch_frontend.py --port 8080
:done
pause
endlocal
