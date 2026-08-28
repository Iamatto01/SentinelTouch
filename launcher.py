"""
SentinelBonsai Launcher
=======================
Nice startup wrapper that shows a banner, checks dependencies,
downloads the Whisper model on first run, and then starts the assistant.
This is the entry point for the .exe build.
"""
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

# When running as a PyInstaller bundle, set the working directory
# to where the .exe is located so sentinel_memory.json is found.
if getattr(sys, "frozen", False):
    os.chdir(os.path.dirname(sys.executable))


def print_banner():
    """Show a clean ASCII banner on startup."""
    banner = r"""
  ╔═══════════════════════════════════════════════╗
  ║                                               ║
  ║        ███████╗███████╗███╗   ██╗████████╗    ║
  ║        ██╔════╝██╔════╝████╗  ██║╚══██╔══╝    ║
  ║        ███████╗█████╗  ██╔██╗ ██║   ██║       ║
  ║        ╚════██║██╔══╝  ██║╚██╗██║   ██║       ║
  ║        ███████║███████╗██║ ╚████║   ██║       ║
  ║        ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝       ║
  ║                                               ║
  ║          S E N T I N E L   B O N S A I        ║
  ║            Desktop Voice Assistant             ║
  ║                                               ║
  ╚═══════════════════════════════════════════════╝
    """
    print(banner)


def get_project_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def check_ffmpeg():
    """Check if ffmpeg is available."""
    import shutil
    if shutil.which("ffmpeg"):
        return True

    # Try to find ffmpeg in WinGet packages
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    winget_dir = os.path.join(local_app_data, "Microsoft", "WinGet", "Packages")
    if os.path.isdir(winget_dir):
        for pkg in os.listdir(winget_dir):
            if not pkg.lower().startswith("gyan.ffmpeg"):
                continue
            root = os.path.join(winget_dir, pkg)
            for dirpath, _, filenames in os.walk(root):
                if "ffmpeg.exe" in {f.lower() for f in filenames}:
                    os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")
                    if shutil.which("ffmpeg"):
                        return True
    return False


def test_llama_server() -> bool:
    for endpoint in ["/health", "/v1/models", "/"]:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:8080{endpoint}", method="GET")
            with urllib.request.urlopen(req, timeout=4) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            continue
    return False


def start_llama_server(project_root: str) -> subprocess.Popen | None:
    server_exe = os.path.join(project_root, "llama.cpp", "build", "bin", "llama-server.exe")
    if not os.path.exists(server_exe):
        print("MISSING!")
        print()
        print(f"  llama-server.exe was not found at:\n  {server_exe}")
        print("  Build llama.cpp first, then restart this application.")
        print()
        input("  Press Enter to exit...")
        return None

    print("STARTING!", flush=True)
    print()
    print("  Starting llama.cpp server...")
    print("  Model: prism-ml/Bonsai-8B-gguf:Q1_0")
    print()

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        [server_exe, "-hf", "prism-ml/Bonsai-8B-gguf:Q1_0"],
        cwd=project_root,
        creationflags=creationflags,
    )


def main():
    project_root = get_project_root()
    os.chdir(project_root)

    print_banner()
    print("  Starting up...\n")

    server_process = None

    # Step 1: Check ffmpeg
    print("  [1/3] Checking ffmpeg...", end=" ", flush=True)
    if check_ffmpeg():
        print("OK")
    else:
        print("MISSING!")
        print()
        print("  ffmpeg is required for audio processing.")
        print("  Install it by running this command in a terminal:")
        print()
        print("    winget install -e --id Gyan.FFmpeg")
        print()
        print("  Then restart this application.")
        print()
        input("  Press Enter to exit...")
        return

    # Step 2: Check or start llama.cpp server
    print("  [2/3] Checking llama.cpp server...", end=" ", flush=True)
    if test_llama_server():
        print("OK")
    else:
        server_process = start_llama_server(project_root)
        if server_process is None:
            return

        for _ in range(60):
            if test_llama_server():
                print("READY")
                break
            time.sleep(1)
        else:
            print("NOT READY")
            print()
            print("  The llama.cpp server did not become ready in time.")
            print("  Check the server output for details, then try again.")
            print()
            input("  Press Enter to exit...")
            return

    # Step 3: Load Whisper model (downloads on first run)
    print("  [3/3] Loading Whisper model...", end=" ", flush=True)
    try:
        import whisper
        # This will download the model on first run (~150MB for base)
        _ = whisper.load_model("base")
        print("OK")
    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("  Could not load Whisper model. Check your internet connection")
        print("  for first-time model download.")
        print()
        input("  Press Enter to exit...")
        return

    print()
    print("  All checks passed! Launching SentinelBonsai...")
    print("  " + "=" * 45)
    print()

    # Now run the main assistant
    from SentinelBonsai import main as sentinel_main
    try:
        sentinel_main()
    finally:
        if server_process and server_process.poll() is None:
            try:
                server_process.terminate()
                server_process.wait(timeout=10)
            except Exception:
                try:
                    server_process.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Goodbye!")
    except Exception as e:
        print(f"\n  Fatal error: {e}")
        input("\n  Press Enter to exit...")
