@echo off
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 scripts\preflight.py
) else (
    python scripts\preflight.py
)
if errorlevel 1 (
    echo Preflight failed. Verify the bundled inputs.
    pause
    exit /b 1
)
where claude >nul 2>nul
if errorlevel 1 (
    echo Claude Code CLI not found. Install it or run claude in this folder.
    pause
    exit /b 1
)
claude "Read CLAUDE.md and PROMPT_FOR_CLAUDE.md. Implement the RHU Catalog Streamlit chatbot in this folder, run all acceptance tests and app smoke checks, fix failures, and write FINAL_IMPLEMENTATION_REPORT.md. Do not stop after planning."
