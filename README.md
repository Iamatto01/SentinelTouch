# SentinelBonsai

SentinelBonsai is a local Windows voice assistant that uses:

- speech_recognition for microphone capture
- Whisper (local) for speech-to-text
- llama.cpp server for chat responses
- pyttsx3 for text-to-speech

It supports wake word activation ("Sentinel"), action execution tags, and persistent chat memory.

## What Is Included

- SentinelBonsai.py: main assistant runtime
- launcher.py: startup checker + single-exe launcher entry point
- sentinel_memory.json: persistent conversation memory
- scripts/start_llama_server.ps1: start local llama.cpp server
- scripts/start_sentinel.ps1: start assistant
- scripts/start_all.ps1: start server (new window) + assistant
- scripts/stop_all.ps1: stop server and assistant
- Run Llama Server.bat: quick wrapper for server script
- Run Sentinel Assistant.bat: quick wrapper for assistant script
- Run All.bat: quick wrapper for start_all.ps1
- Stop Sentinel.bat: quick wrapper for stop_all.ps1

## Prerequisites

1. Windows 10 or Windows 11
2. Python 3.10+ installed
3. ffmpeg installed
4. llama.cpp built so this file exists:
   .\\llama.cpp\\build\\bin\\llama-server.exe

Install ffmpeg if needed:

winget install -e --id Gyan.FFmpeg

## One-Time Setup

From project root:

py -3.11 -m venv .venv
.\\.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## Easiest Way To Run

If you have already built the packaged launcher, run:

SentinelBonsai.exe

Otherwise, double-click:

Run All.bat

This launches the llama.cpp server and then starts the Sentinel assistant.

The packaged exe is built from `SentinelBonsai.spec` and now starts the server automatically, so it is the closest thing to a single-file launcher.

To build it:

pyinstaller SentinelBonsai.spec

## Manual Two-Window Run

Window 1:

Run Llama Server.bat

Window 2:

Run Sentinel Assistant.bat

## Stop Everything

Double-click:

Stop Sentinel.bat

## How To Use The Assistant

1. Say: Sentinel
2. Assistant replies: Yes sir.
3. Speak your command after that
4. Say exit, quit, stop, or goodbye to close assistant

Inline usage also works:

Sentinel open notepad

## Notes About Actions

The model can return command tags in this format:

[ACTION: &lt;windows command&gt;]

SentinelBonsai executes that command asynchronously on Windows.

Only run models/prompts you trust, because generated commands can launch apps or scripts.

## Troubleshooting

- Error mentions ffmpeg or WinError 2: Install ffmpeg and restart terminal.
- Assistant says server is not reachable: Start server first with Run Llama Server.bat.
- No microphone detected: Check Windows microphone permission and default input device.
- Slow first run: Whisper model download happens the first time.
