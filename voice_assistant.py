"""Voice assistant module for Sentinel Touch.

Provides background speech recognition and text-to-speech output.
Gracefully disables voice features when no microphone or TTS engine is
available so that gesture control continues to work regardless.
"""

import logging
import queue
import threading

logger = logging.getLogger(__name__)

try:
    import speech_recognition as sr  # type: ignore

    _SR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SR_AVAILABLE = False
    logger.warning("SpeechRecognition not installed – voice input disabled.")

try:
    import pyttsx3  # type: ignore

    _TTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed – voice output disabled.")


class VoiceAssistant:
    """Runs speech recognition and TTS on background daemon threads.

    Usage::

        va = VoiceAssistant()
        va.speak("Sentinel Touch online.")
        ...
        while True:
            cmd = va.get_command()
            if cmd:
                handle(cmd)
        va.stop()
    """

    def __init__(self, name: str = "Sentinel") -> None:
        self.name = name
        self.available: bool = False      # microphone + SR ready
        self.tts_available: bool = False  # TTS engine ready
        self.listening: bool = False      # currently recording audio
        self.enabled: bool = True         # set False to stop threads
        self.last_command: str = ""
        self.last_response: str = ""

        self._command_queue: queue.Queue[str] = queue.Queue()
        self._tts_queue: queue.Queue[str | None] = queue.Queue()

        self._init_tts()
        self._init_mic()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_tts(self) -> None:
        if not _TTS_AVAILABLE:
            return
        try:
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", 165)
            self._tts_engine.setProperty("volume", 0.9)

            # Try to pick a female voice for a more FRIDAY-like feel.
            voices = self._tts_engine.getProperty("voices")
            female = [
                v
                for v in voices
                if "female" in v.name.lower()
                or getattr(v, "gender", "").lower() == "female"
            ]
            if female:
                self._tts_engine.setProperty("voice", female[0].id)

            self._tts_thread = threading.Thread(
                target=self._tts_loop, daemon=True, name="SentinelTTS"
            )
            self._tts_thread.start()
            self.tts_available = True
        except Exception as exc:  # pragma: no cover
            logger.warning("TTS engine init failed: %s", exc)

    def _init_mic(self) -> None:
        if not _SR_AVAILABLE:
            return
        try:
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold = 300
            self._recognizer.dynamic_energy_threshold = True
            self._recognizer.pause_threshold = 0.7

            self._mic = sr.Microphone()
            with self._mic as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)

            self._listen_thread = threading.Thread(
                target=self._listen_loop, daemon=True, name="SentinelListen"
            )
            self._listen_thread.start()
            self.available = True
        except Exception as exc:  # pragma: no cover
            logger.warning("Microphone init failed – voice commands disabled: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def speak(self, text: str) -> None:
        """Queue *text* for asynchronous TTS playback."""
        self.last_response = text
        if self.tts_available:
            self._tts_queue.put(text)

    def get_command(self) -> str | None:
        """Return the next recognised command string, or *None* if the queue is empty."""
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Signal background threads to terminate."""
        self.enabled = False
        if self.tts_available:
            self._tts_queue.put(None)  # sentinel to unblock TTS thread

    # ------------------------------------------------------------------
    # Background thread loops
    # ------------------------------------------------------------------

    def _tts_loop(self) -> None:
        while True:
            text = self._tts_queue.get()
            if text is None:
                break
            try:
                self._tts_engine.say(text)
                self._tts_engine.runAndWait()
            except Exception as exc:
                logger.debug("TTS playback error: %s", exc)

    def _listen_loop(self) -> None:
        while self.enabled:
            try:
                with self._mic as source:
                    self.listening = True
                    try:
                        audio = self._recognizer.listen(
                            source, timeout=2, phrase_time_limit=6
                        )
                    finally:
                        self.listening = False

                try:
                    text: str = self._recognizer.recognize_google(audio).lower().strip()
                    if text:
                        self.last_command = text
                        self._command_queue.put(text)
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as exc:
                    logger.debug("Speech recognition request error: %s", exc)

            except sr.WaitTimeoutError:
                self.listening = False
            except Exception as exc:
                self.listening = False
                logger.debug("Listen loop error: %s", exc)
