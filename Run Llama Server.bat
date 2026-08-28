@echo off
title Llama Server
cd /d "%~dp0"
echo Starting llama.cpp server...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_llama_server.ps1"
pause
