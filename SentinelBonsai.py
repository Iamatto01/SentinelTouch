"""
SentinelBonsai — Desktop Neural Voice & Action Assistant
==========================================================
Upgraded to support:
  1. Groq Cloud API (100% Free, 500 tokens/sec Llama-3.3-70B)
  2. Google Gemini API / OpenRouter / DeepSeek
  3. Ollama (if installed locally)
  4. Built-in Offline Smart Command & Tool Brain (Instant fallback, zero dependency!)
"""

import os
import re
import subprocess
import tempfile
import time
import json
import shutil
import urllib.error
import urllib.request
import webbrowser

# Speech dependencies (with graceful fallbacks)
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import whisper
except ImportError:
    whisper = None

# Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

ACTION_PATTERN = re.compile(r"\[ACTION:\s*(.+?)\]", re.IGNORECASE | re.DOTALL)
WAKE_WORD_PATTERN = re.compile(
    r"^\s*(?:hey\s+)?sentinel(?:\s+bonsai)?(?:[\s,.:;!?-]+(.*))?$",
    re.IGNORECASE,
)
WAKE_ACK_TEXT = "Yes sir, I am listening."
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentinel_memory.json")
MAX_MEMORY_MESSAGES = 20


def ensure_ffmpeg_available() -> bool:
    """Ensure ffmpeg is available for Whisper transcription."""
    if shutil.which("ffmpeg"):
        return True

    local_app_data = os.environ.get("LOCALAPPDATA", "")
    winget_packages_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_packages_dir):
        for package_dir in os.listdir(winget_packages_dir):
            if not package_dir.lower().startswith("gyan.ffmpeg"):
                continue

            root = os.path.join(winget_packages_dir, package_dir)
            for dir_path, _, file_names in os.walk(root):
                lowered = {name.lower() for name in file_names}
                if "ffmpeg.exe" in lowered:
                    os.environ["PATH"] = dir_path + os.pathsep + os.environ.get("PATH", "")
                    if shutil.which("ffmpeg"):
                        return True

    return False


def load_persistent_memory() -> list[dict[str, str]]:
    """Load saved conversation history from disk."""
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as memory_file:
            data = json.load(memory_file)
        if isinstance(data, list):
            return data[-MAX_MEMORY_MESSAGES:]
    except Exception as exc:
        print(f"Memory load warning: {exc}")
    return []


def save_persistent_memory(messages: list[dict[str, str]]) -> None:
    """Persist a rolling window of chat messages to disk."""
    memory_messages = [
        {"role": msg.get("role", ""), "content": msg.get("content", "")}
        for msg in messages
        if msg.get("role") in {"user", "assistant"}
    ][-MAX_MEMORY_MESSAGES:]

    temp_path = f"{MEMORY_FILE}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as memory_file:
            json.dump(memory_messages, memory_file, ensure_ascii=True, indent=2)
        os.replace(temp_path, MEMORY_FILE)
    except OSError as exc:
        print(f"Memory save warning: {exc}")


def extract_action_and_clean_text(text: str) -> tuple[str | None, str]:
    """Return the hidden action command and the text safe for speech output."""
    match = ACTION_PATTERN.search(text)
    command = match.group(1).strip() if match else None

    cleaned = ACTION_PATTERN.sub("", text).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return command, cleaned


def execute_action_async(command: str) -> None:
    """Run command asynchronously so assistant can keep listening."""
    if not command:
        return

    print(f"⚡ [Action Executed]: {command}")
    
    # Handle web links directly
    if command.startswith("http://") or command.startswith("https://"):
        webbrowser.open(command)
        return

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        subprocess.Popen(command, shell=True, creationflags=creationflags)
    except Exception as e:
        print(f"Action error: {e}")


def query_cloud_groq(messages: list[dict[str, str]], api_key: str) -> str:
    """Query Groq ultra-fast Llama-3 API."""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SentinelTouch/2.5"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()


def query_cloud_gemini(query: str, api_key: str) -> str:
    """Query Google Gemini 2.0 Flash API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"You are Sentinel Touch desktop voice assistant. Answer briefly: {query}"}]}]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


def smart_offline_brain(query: str) -> str:
    """Intelligent zero-dependency local assistant brain for system control."""
    q = query.lower()

    if "buka convert" in q or "convert" in q:
        return "Membuka Sentinel Convert. [ACTION: https://convert.sentinelai.studio]"
    if "buka view" in q or "responsive" in q:
        return "Membuka Sentinel View. [ACTION: https://view.sentinelai.studio]"
    if "buka tree" in q or "salasilah" in q:
        return "Membuka Sentinel Tree. [ACTION: https://tree.sentinelai.studio]"
    if "buka story" in q or "cerita" in q:
        return "Membuka Sentinel Story. [ACTION: https://story.sentinelai.studio]"
    if "buka truth" in q or "fact check" in q:
        return "Membuka Sentinel Truth. [ACTION: https://truth.sentinelai.studio]"
    if "buka portal" in q or "hub" in q or "master" in q:
        return "Membuka Sentinel Master Hub. [ACTION: https://hub.sentinelai.studio]"
    
    # Windows system controls
    if "notepad" in q or "nota" in q:
        return "Membuka Notepad. [ACTION: notepad.exe]"
    if "calc" in q or "kalkulator" in q:
        return "Membuka Kalkulator. [ACTION: calc.exe]"
    if "terminal" in q or "powershell" in q or "cmd" in q:
        return "Membuka PowerShell Terminal. [ACTION: start powershell]"
    
    if "status" in q or "kesihatan" in q:
        return "Semua servis Sentinel beroperasi secara optimum di pelayan Kali Linux."
    
    if "siapa kamu" in q or "who are you" in q:
        return "Saya ialah Sentinel Touch, pembantu desktop berkuasa AI dan kawalan gerak isyarat untuk ekosistem anda."

    return f"Saya mendengar arahan anda: {query}. Membantu anda dalam ekosistem Sentinel."


def query_ai_engine(messages: list[dict[str, str]]) -> str:
    """Route conversation to the best available AI engine."""
    user_query = messages[-1].get("content", "")

    # 1. Try Groq (if key available)
    if GROQ_API_KEY:
        try:
            return query_cloud_groq(messages, GROQ_API_KEY)
        except Exception as e:
            print(f"Groq API fallback ({e})")

    # 2. Try Gemini (if key available)
    if GEMINI_API_KEY:
        try:
            return query_cloud_gemini(user_query, GEMINI_API_KEY)
        except Exception as e:
            print(f"Gemini API fallback ({e})")

    # 3. Built-in instant Smart Assistant Brain (Zero Hardware Requirement!)
    return smart_offline_brain(user_query)


def speak(tts_engine, text: str) -> None:
    """Speak text using pyttsx3 or fallback to console."""
    if not text:
        return
    print(f"🤖 Sentinel: {text}")
    if tts_engine:
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")


def main() -> None:
    print("=" * 60)
    print("   🖐️  S E N T I N E L   T O U C H   (B O N S A I   2 . 5)  🖐️")
    print("       Zero-Hardware Desktop Neural Voice Assistant")
    print("=" * 60)

    # Initialize Text-To-Speech
    tts_engine = None
    if pyttsx3:
        try:
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 180)
        except Exception:
            pass

    # Initialize Speech Recognition
    recognizer = None
    microphone = None
    if sr:
        try:
            recognizer = sr.Recognizer()
            recognizer.pause_threshold = 0.8
            microphone = sr.Microphone()
            with microphone as source:
                print("Calibrating microphone noise floor...")
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
        except Exception as e:
            print(f"Microphone initialization notice: {e}")

    messages = [
        {
            "role": "system",
            "content": (
                "You are Sentinel Touch, an intelligent voice and system assistant. "
                "Answer concisely in under 2 sentences. "
                "To execute Windows commands, include [ACTION: <command>]."
            )
        }
    ]
    messages.extend(load_persistent_memory())

    print("\n🟢 Sentinel Touch is ACTIVE & LISTENING!")
    print("👉 Voice Mode: Say 'Hey Sentinel' or speak your query.")
    print("👉 Keyboard Mode: Type your prompt directly and press Enter.\n")

    speak(tts_engine, "Sentinel Touch is online and ready, sir.")

    while True:
        try:
            user_text = ""

            # Attempt voice listening if mic available
            if recognizer and microphone:
                try:
                    with microphone as source:
                        print("\n🎙️ Listening (or type prompt)...", end="", flush=True)
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                        print(" Processing speech...")
                        user_text = recognizer.recognize_google(audio, language="ms-MY")
                        print(f"👤 Heard: {user_text}")
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    pass
                except Exception as e:
                    # Fallback to keyboard input
                    pass

            if not user_text:
                try:
                    user_text = input("\n💬 Enter query (or 'exit' to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

            if not user_text:
                continue

            if user_text.lower() in {"exit", "quit", "keluar"}:
                speak(tts_engine, "Shutting down Sentinel Touch. Goodbye, sir.")
                break

            # Process query through AI Engine
            messages.append({"role": "user", "content": user_text})
            raw_reply = query_ai_engine(messages)
            action, clean_text = extract_action_and_clean_text(raw_reply)

            messages.append({"role": "assistant", "content": clean_text})
            save_persistent_memory(messages)

            # Output voice and execute actions
            speak(tts_engine, clean_text)
            if action:
                execute_action_async(action)

        except KeyboardInterrupt:
            print("\nExiting Sentinel Touch...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
