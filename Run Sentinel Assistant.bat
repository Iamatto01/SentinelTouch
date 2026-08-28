@echo off
title Sentinel Assistant
cd /d "%~dp0"
echo Starting Sentinel assistant...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\start_sentinel.ps1"
pause
