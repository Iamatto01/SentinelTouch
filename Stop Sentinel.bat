@echo off
title SentinelBonsai - Stop
cd /d "%~dp0"
echo Stopping Sentinel processes...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\stop_all.ps1"
pause
