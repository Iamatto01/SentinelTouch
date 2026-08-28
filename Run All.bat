@echo off
title SentinelBonsai - Start All
cd /d "%~dp0"
echo Starting llama.cpp server and Sentinel assistant...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_all.ps1"
pause
