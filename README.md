# Sentinel Touch — FRIDAY Edition

Control your computer with hand gestures **and voice commands**, inspired by
Tony Stark's FRIDAY AI assistant from the Iron Man films.

---

## Features

### Gesture Control
| Gesture | Action |
|---------|--------|
| **Index finger only** | MOVE mode — cursor follows your fingertip |
| **Index + Middle fingers** | SCROLL mode — move hand up/down to scroll |
| **Index + Middle + Ring** | DRAG mode — cursor moves; pinch to hold left button |
| **Thumb–Index pinch** (MOVE) | Left click |
| **Thumb–Ring pinch** (MOVE) | Right click |
| **Fist / open palm** | IDLE — no action |
| **Hold pointing direction** | Auto-pan cursor in 8 directions |
| **Two hands outward (2F each)** | Maximize window (Win+↑) |
| **Two hands inward (2F each)** | Minimize window (Win+↓) |
| **Two hands inward (3F each)** | Task View (Win+Tab) |

### Voice Commands
Say any of the following:

| Command | Action |
|---------|--------|
| `screenshot` / `capture` | Save a screenshot |
| `maximize` | Maximize current window |
| `minimize` | Minimize current window |
| `close window` | Close current window (Alt+F4) |
| `new tab` | Open new browser tab (Ctrl+T) |
| `close tab` | Close current tab (Ctrl+W) |
| `copy` | Copy (Ctrl+C) |
| `paste` | Paste (Ctrl+V) |
| `undo` | Undo (Ctrl+Z) |
| `redo` | Redo (Ctrl+Y) |
| `save` | Save (Ctrl+S) |
| `select all` | Select all (Ctrl+A) |
| `volume up` | Volume up |
| `volume down` | Volume down |
| `mute` | Mute/unmute |
| `play` / `pause` | Play/pause media |
| `next track` | Next media track |
| `previous track` | Previous media track |
| `switch window` | Alt+Tab |
| `task view` | Win+Tab |
| `show desktop` | Win+D |
| `search` | Win+S |
| `lock screen` | Win+L |
| `disable` / `freeze` | Disable gesture control |
| `enable` / `unfreeze` | Re-enable gesture control |
| `help` | Hear all available commands |
| `quit` / `stop` | Exit Sentinel Touch |

---

## Requirements

- Python 3.10+
- A webcam
- A microphone (optional — voice commands only)
- Windows, macOS, or Linux

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Linux users:** Install system-level audio libraries first:
> ```bash
> sudo apt-get install portaudio19-dev espeak espeak-ng
> ```

---

## Usage

```bash
python gesture_mouse_control.py
```

The FRIDAY-style HUD overlay shows:

- **Mode badge** — current gesture mode (MOVE / SCROLL / DRAG / IDLE)
- **Listening indicator** — pulsing red dot when microphone is active
- **Last command** — the most recent voice command recognised
- **AI response** — what Sentinel said back to you
- **Bottom bar** — quick-reference gesture cheat-sheet

Press **Q** or say **"quit"** to exit.

---

## Architecture

```
gesture_mouse_control.py   — main loop (webcam + gesture logic + HUD)
voice_assistant.py         — background speech recognition and TTS threads
requirements.txt           — Python dependencies
```
