"""Sentinel Touch — gesture + voice controlled mouse.

Upgrade: FRIDAY-style AI assistant layer.
  • Voice commands (speech → action) with spoken feedback (TTS).
  • Gesture modes: MOVE (1 finger) | SCROLL (2 fingers) | DRAG (3 fingers).
  • Right-click via thumb–ring-finger pinch in MOVE mode.
  • Drag mode: hold left button while pinching in DRAG mode.
  • Two-hand combo gestures (maximize / minimize / task-view).
  • FRIDAY-style HUD overlay with mode indicator and listening pulse.
"""

import logging
import math
import time
import ctypes

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

from voice_assistant import VoiceAssistant

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Gesture mode constants
# ---------------------------------------------------------------------------
GESTURE_MOVE = "MOVE"      # 1 main finger extended → cursor movement
GESTURE_SCROLL = "SCROLL"  # 2 main fingers extended → vertical scroll
GESTURE_DRAG = "DRAG"      # 3 main fingers extended → drag (hold LMB)
GESTURE_IDLE = "IDLE"      # 0 extended → no action

_MODE_COLORS: dict[str, tuple[int, int, int]] = {
    GESTURE_MOVE: (100, 255, 0),
    GESTURE_SCROLL: (255, 200, 0),
    GESTURE_DRAG: (0, 200, 255),
    GESTURE_IDLE: (128, 128, 128),
}


def get_virtual_screen_bounds() -> tuple[int, int, int, int]:
    """Return virtual desktop bounds (supports multi-monitor on Windows)."""
    if hasattr(ctypes, "windll"):
        user32 = ctypes.windll.user32
        # SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN
        left = int(user32.GetSystemMetrics(76))
        top = int(user32.GetSystemMetrics(77))
        width = int(user32.GetSystemMetrics(78))
        height = int(user32.GetSystemMetrics(79))
        if width > 0 and height > 0:
            return left, top, width, height

    width, height = pyautogui.size()
    return 0, 0, int(width), int(height)


def detect_point_direction(lm: list, frame_width: int, frame_height: int) -> tuple[str, float, float] | None:
    """Estimate 8-way pointing direction from index MCP (5) to index tip (8)."""
    vx = (lm[8].x - lm[5].x) * frame_width
    vy = (lm[8].y - lm[5].y) * frame_height
    mag = math.hypot(vx, vy)
    if mag < 35:
        return None

    angle = math.atan2(vy, vx)
    sector = int(round(angle / (math.pi / 4.0))) % 8
    sqrt2_inv = math.sqrt(0.5)
    sectors = [
        ("RIGHT", 1.0, 0.0),
        ("DOWN-RIGHT", sqrt2_inv, sqrt2_inv),
        ("DOWN", 0.0, 1.0),
        ("DOWN-LEFT", -sqrt2_inv, sqrt2_inv),
        ("LEFT", -1.0, 0.0),
        ("UP-LEFT", -sqrt2_inv, -sqrt2_inv),
        ("UP", 0.0, -1.0),
        ("UP-RIGHT", sqrt2_inv, -sqrt2_inv),
    ]
    return sectors[sector]


def is_dead_man_active(lm: list) -> bool:
    """Enable control when index finger is extended (direction-agnostic)."""

    def dist(a: int, b: int) -> float:
        dx = lm[a].x - lm[b].x
        dy = lm[a].y - lm[b].y
        dz = lm[a].z - lm[b].z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    # Finger extension heuristic based on relative segment lengths.
    index_extended = dist(8, 5) > dist(6, 5) * 1.35
    return index_extended


def count_extended_main_fingers(lm: list) -> int:
    """Count extended fingers among index, middle, and ring."""

    def dist(a: int, b: int) -> float:
        dx = lm[a].x - lm[b].x
        dy = lm[a].y - lm[b].y
        dz = lm[a].z - lm[b].z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    finger_triplets = [
        (8, 6, 5),    # index: tip, pip, mcp
        (12, 10, 9),  # middle
        (16, 14, 13), # ring
    ]

    extended = 0
    for tip, pip, mcp in finger_triplets:
        if dist(tip, mcp) > dist(pip, mcp) * 1.2:
            extended += 1
    return extended


def palm_center_px(lm: list, frame_width: int, frame_height: int) -> tuple[float, float]:
    """Return a stable palm center in pixel coordinates."""
    # Wrist + MCP knuckles for a less jittery center estimate.
    idx = [0, 5, 9, 13, 17]
    x = sum(lm[i].x for i in idx) / len(idx)
    y = sum(lm[i].y for i in idx) / len(idx)
    return x * frame_width, y * frame_height


def detect_gesture_mode(lm: list) -> str:
    """Return the current gesture mode based on extended main-finger count."""
    count = count_extended_main_fingers(lm)
    if count == 1:
        return GESTURE_MOVE
    if count == 2:
        return GESTURE_SCROLL
    if count == 3:
        return GESTURE_DRAG
    return GESTURE_IDLE


def is_right_click_pinch(
    lm: list, frame_width: int, frame_height: int, threshold: int
) -> bool:
    """Detect a thumb–ring-finger pinch for right-click in MOVE mode."""
    tx = lm[4].x * frame_width
    ty = lm[4].y * frame_height
    rx = lm[16].x * frame_width
    ry = lm[16].y * frame_height
    return math.hypot(tx - rx, ty - ry) < threshold


def process_voice_command(
    cmd: str, va: VoiceAssistant, gesture_enabled_ref: list[bool]
) -> tuple[str, bool]:
    """Process a recognised voice command.

    Returns (action_label, should_quit).
    *gesture_enabled_ref* is a one-element mutable list used to toggle
    gesture control from a voice command.
    """
    should_quit = False
    action: str = ""

    if any(k in cmd for k in ("screenshot", "capture screen", "capture")):
        filename = f"sentinel_screenshot_{int(time.time())}.png"
        pyautogui.screenshot(filename)
        va.speak(f"Screenshot saved as {filename}.")
        action = "SCREENSHOT"

    elif "maximize" in cmd:
        pyautogui.hotkey("win", "up")
        va.speak("Maximizing window.")
        action = "MAXIMIZE"

    elif "minimize" in cmd:
        pyautogui.hotkey("win", "down")
        va.speak("Minimizing window.")
        action = "MINIMIZE"

    elif "close window" in cmd or cmd.strip() == "close":
        pyautogui.hotkey("alt", "f4")
        va.speak("Closing window.")
        action = "CLOSE WINDOW"

    elif "new tab" in cmd:
        pyautogui.hotkey("ctrl", "t")
        va.speak("New tab opened.")
        action = "NEW TAB"

    elif "close tab" in cmd:
        pyautogui.hotkey("ctrl", "w")
        va.speak("Tab closed.")
        action = "CLOSE TAB"

    elif "copy" in cmd:
        pyautogui.hotkey("ctrl", "c")
        va.speak("Copied.")
        action = "COPY"

    elif "paste" in cmd:
        pyautogui.hotkey("ctrl", "v")
        va.speak("Pasted.")
        action = "PASTE"

    elif "undo" in cmd:
        pyautogui.hotkey("ctrl", "z")
        va.speak("Undone.")
        action = "UNDO"

    elif "redo" in cmd:
        pyautogui.hotkey("ctrl", "y")
        va.speak("Redone.")
        action = "REDO"

    elif "save" in cmd:
        pyautogui.hotkey("ctrl", "s")
        va.speak("Saved.")
        action = "SAVE"

    elif "select all" in cmd:
        pyautogui.hotkey("ctrl", "a")
        va.speak("All selected.")
        action = "SELECT ALL"

    elif "volume up" in cmd:
        pyautogui.press("volumeup")
        va.speak("Volume up.")
        action = "VOLUME UP"

    elif "volume down" in cmd:
        pyautogui.press("volumedown")
        va.speak("Volume down.")
        action = "VOLUME DOWN"

    elif "mute" in cmd:
        pyautogui.press("volumemute")
        va.speak("Muted.")
        action = "MUTE"

    elif "play" in cmd or "pause" in cmd:
        pyautogui.press("playpause")
        va.speak("Play/pause toggled.")
        action = "PLAY/PAUSE"

    elif "next track" in cmd or "next song" in cmd:
        pyautogui.press("nexttrack")
        va.speak("Next track.")
        action = "NEXT TRACK"

    elif "previous track" in cmd or "previous song" in cmd or "prev track" in cmd:
        pyautogui.press("prevtrack")
        va.speak("Previous track.")
        action = "PREV TRACK"

    elif "switch window" in cmd or "alt tab" in cmd:
        pyautogui.hotkey("alt", "tab")
        va.speak("Switching window.")
        action = "SWITCH WINDOW"

    elif "task view" in cmd:
        pyautogui.hotkey("win", "tab")
        va.speak("Task view.")
        action = "TASK VIEW"

    elif "show desktop" in cmd or "go to desktop" in cmd:
        pyautogui.hotkey("win", "d")
        va.speak("Showing desktop.")
        action = "SHOW DESKTOP"

    elif "search" in cmd:
        pyautogui.hotkey("win", "s")
        va.speak("Opening search.")
        action = "SEARCH"

    elif "lock" in cmd and "screen" in cmd:
        pyautogui.hotkey("win", "l")
        va.speak("Locking screen.")
        action = "LOCK SCREEN"

    elif "help" in cmd or "what can you do" in cmd or "commands" in cmd:
        va.speak(
            "I can control windows, tabs, clipboard, media, volume, and system settings. "
            "Say 'screenshot', 'maximize', 'copy', 'volume up', 'next track', and more. "
            "See the README for the complete list."
        )
        print(
            "[Sentinel] Voice commands: screenshot, maximize, minimize, close window, "
            "new tab, close tab, copy, paste, undo, redo, save, select all, "
            "volume up, volume down, mute, play, pause, next track, previous track, "
            "switch window, task view, show desktop, search, lock screen, "
            "disable, enable, quit."
        )
        action = "HELP"

    elif any(k in cmd for k in ("disable gesture", "freeze gesture", "disable", "freeze")):
        gesture_enabled_ref[0] = False
        va.speak("Gesture control disabled. Say enable to resume.")
        action = "DISABLE GESTURES"

    elif any(k in cmd for k in ("enable gesture", "unfreeze gesture", "enable", "unfreeze")):
        gesture_enabled_ref[0] = True
        va.speak("Gesture control enabled.")
        action = "ENABLE GESTURES"

    elif any(k in cmd for k in ("stop", "quit", "exit", "shutdown", "shut down", "goodbye")):
        va.speak("Sentinel Touch shutting down. Goodbye.")
        time.sleep(1.5)
        should_quit = True
        action = "QUIT"

    return action, should_quit


def draw_hud(
    frame: "np.ndarray",
    mode: str,
    va_listening: bool,
    va_available: bool,
    last_cmd: str,
    last_response: str,
    control_enabled: bool,
    gesture_enabled: bool,
) -> None:
    """Render a FRIDAY-style HUD overlay on *frame* in-place."""
    h, w = frame.shape[:2]

    # --- Top status bar ---------------------------------------------------
    top_overlay = frame.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.addWeighted(top_overlay, 0.55, frame, 0.45, 0, frame)

    # Title
    cv2.putText(
        frame,
        "S.E.N.T.I.N.E.L TOUCH",
        (12, 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        (0, 200, 255),
        2,
    )

    # Gesture mode badge
    mode_color = _MODE_COLORS.get(mode, (255, 255, 255))
    cv2.putText(
        frame,
        f"MODE: {mode}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        mode_color,
        2,
    )

    # Control / gesture-enabled status (top-right)
    if not gesture_enabled:
        status_txt, status_clr = "GESTURES OFF", (0, 80, 255)
    elif control_enabled:
        status_txt, status_clr = "CTRL: ON ", (0, 255, 80)
    else:
        status_txt, status_clr = "CTRL: OFF", (0, 80, 255)
    cv2.putText(
        frame, status_txt, (w - 155, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_clr, 2
    )

    # Voice listening pulse (top-right)
    if va_available:
        pulse_r = int(abs(math.sin(time.time() * 4)) * 7) + 5
        pulse_clr = (0, 40, 230) if va_listening else (60, 60, 60)
        cv2.circle(frame, (w - 18, 55), pulse_r, pulse_clr, -1)
        voice_label = "LISTENING" if va_listening else "VOICE RDY"
        cv2.putText(
            frame,
            voice_label,
            (w - 130, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            pulse_clr,
            1,
        )
    else:
        cv2.putText(
            frame, "VOICE N/A", (w - 120, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 80, 80), 1
        )

    # --- Voice command / response banner (below top bar) ------------------
    if last_cmd:
        cv2.putText(
            frame,
            f"CMD \u25b8 {last_cmd[:60]}",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (190, 190, 190),
            1,
        )
    if last_response:
        cv2.putText(
            frame,
            f"AI  \u25b8 {last_response[:60]}",
            (12, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 230, 180),
            1,
        )

    # --- Bottom info bar --------------------------------------------------
    bot_overlay = frame.copy()
    cv2.rectangle(bot_overlay, (0, h - 68), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bot_overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(
        frame,
        "Index=MOVE | 2F=SCROLL | 3F=DRAG | Pinch=LClick | Thumb+Ring=RClick",
        (10, h - 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 190),
        1,
    )
    cv2.putText(
        frame,
        "2H Outward(2F)=Maximize | 2H Inward(2F)=Minimize | 2H Inward(3F)=Win+Tab",
        (10, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (190, 190, 190),
        1,
    )
    cv2.putText(
        frame,
        "Press Q to quit | Say 'help' for voice commands",
        (10, h - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (140, 140, 140),
        1,
    )


def main() -> None:
    # Safety feature: moving mouse to a screen corner can stop automation.
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

    # --- Voice assistant --------------------------------------------------
    print("[Sentinel Touch] Initialising voice assistant…")
    va = VoiceAssistant(name="Sentinel")
    if va.tts_available:
        va.speak("Sentinel Touch online. All systems ready.")
    else:
        print("[Sentinel Touch] TTS unavailable – running in silent mode.")
    if not va.available:
        print("[Sentinel Touch] Microphone unavailable – voice commands disabled.")

    # gesture_enabled is a mutable container so voice commands can toggle it.
    gesture_enabled: list[bool] = [True]
    last_voice_action: str = ""
    voice_action_display_until: float = 0.0

    # Webcam setup.
    cam_width, cam_height = 1280, 720
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Get virtual desktop resolution for multi-monitor coordinate mapping.
    screen_left, screen_top, screen_width, screen_height = get_virtual_screen_bounds()

    # Hand tracker setup.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils

    # Control parameters.
    active_area_ratio = 0.6  # Smaller active area means less hand travel.
    smoothening = 8          # Higher value = smoother but less responsive.
    jitter_deadzone_px = 6.0 # Ignore very small mapped cursor movements.
    pinch_threshold = 35     # Pixel distance threshold for thumb-index pinch.
    right_click_threshold = 45  # Thumb-ring pinch distance for right-click.
    click_cooldown = 0.25    # Seconds between allowed clicks.
    hold_to_pan_seconds = 1.0
    pan_speed_px_per_sec = 450
    hold_stationary_threshold_px = 45
    pan_ramp_seconds = 0.35
    two_hand_move_threshold_px = 10.0
    two_hand_gap_threshold_px = 18.0
    gesture_action_cooldown = 1.0
    center_smoothing_alpha = 0.35
    combo_confirm_frames = 3
    scroll_sensitivity = 18.0   # pixels of hand movement per scroll unit
    scroll_deadzone_px = 10.0   # minimum movement before scrolling

    start_x, start_y = pyautogui.position()
    prev_x, prev_y = float(start_x), float(start_y)
    last_click_time = 0.0
    last_right_click_time = 0.0

    # Scroll state (SCROLL mode)
    prev_scroll_y: float | None = None

    # Drag state (DRAG mode)
    drag_active = False

    # Fractional scroll accumulator: carry sub-unit scroll deltas across frames.
    scroll_accumulator: float = 0.0

    hold_direction = None
    hold_vector_x = 0.0
    hold_vector_y = 0.0
    hold_started_at = 0.0
    hold_anchor = None
    last_frame_time = time.time()
    pan_offset_x = 0.0
    pan_offset_y = 0.0
    prev_left_center_x = None
    prev_right_center_x = None
    last_combo_action_time = 0.0
    outward_two_score = 0
    inward_three_score = 0
    inward_two_score = 0

    # Tracks current gesture mode for HUD display even when no hand is visible.
    current_gesture_mode: str = GESTURE_IDLE

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Mirror view for natural interaction.
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            # --- Voice command processing (non-blocking queue poll) --------
            voice_cmd = va.get_command()
            if voice_cmd:
                action_label, should_quit = process_voice_command(
                    voice_cmd, va, gesture_enabled
                )
                if should_quit:
                    break
                if action_label:
                    last_voice_action = action_label
                    voice_action_display_until = time.time() + 2.5

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Dynamic active area centered in frame to reduce required hand movement.
            active_w = int(frame_width * active_area_ratio)
            active_h = int(frame_height * active_area_ratio)
            frame_margin_x = (frame_width - active_w) // 2
            frame_margin_y = (frame_height - active_h) // 2

            # Derive gesture mode and control state for HUD.
            current_control_enabled = False

            if results.multi_hand_landmarks:
                hand_infos = []
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = hand_landmarks.landmark
                    cx, cy = palm_center_px(lm, frame_width, frame_height)
                    handedness_label = None
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handedness_label = results.multi_handedness[i].classification[0].label
                    hand_infos.append(
                        {
                            "lm": lm,
                            "center_x": float(cx),
                            "center_y": float(cy),
                            "label": handedness_label,
                            "index_tip_x": int(lm[8].x * frame_width),
                            "index_tip_y": int(lm[8].y * frame_height),
                            "thumb_tip_x": int(lm[4].x * frame_width),
                            "thumb_tip_y": int(lm[4].y * frame_height),
                            "control_enabled": is_dead_man_active(lm),
                            "main_finger_count": count_extended_main_fingers(lm),
                        }
                    )

                # Use the right-most hand as cursor control hand for stability.
                right_labeled_hands = [h for h in hand_infos if h["label"] == "Right"]
                if right_labeled_hands:
                    primary_hand = right_labeled_hands[0]
                else:
                    primary_hand = max(hand_infos, key=lambda h: h["center_x"])
                lm = primary_hand["lm"]
                index_tip_x = primary_hand["index_tip_x"]
                index_tip_y = primary_hand["index_tip_y"]
                thumb_tip_x = primary_hand["thumb_tip_x"]
                thumb_tip_y = primary_hand["thumb_tip_y"]
                control_enabled = bool(primary_hand["control_enabled"])
                current_control_enabled = control_enabled

                # Determine gesture mode from finger count.
                current_gesture_mode = detect_gesture_mode(lm)

                # Draw points of interest.
                cv2.circle(frame, (index_tip_x, index_tip_y), 8, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 8, (0, 255, 255), cv2.FILLED)
                cv2.line(frame, (index_tip_x, index_tip_y), (thumb_tip_x, thumb_tip_y), (0, 255, 0), 2)

                # Map webcam coordinates to screen coordinates.
                mapped_x_base = np.interp(
                    index_tip_x,
                    (frame_margin_x, frame_width - frame_margin_x),
                    (screen_left, screen_left + screen_width - 1),
                )
                mapped_y_base = np.interp(
                    index_tip_y,
                    (frame_margin_y, frame_height - frame_margin_y),
                    (screen_top, screen_top + screen_height - 1),
                )

                # Keep values within screen limits.
                mapped_x_base = float(np.clip(mapped_x_base, screen_left, screen_left + screen_width - 1))
                mapped_y_base = float(np.clip(mapped_y_base, screen_top, screen_top + screen_height - 1))
                mapped_x = float(np.clip(mapped_x_base + pan_offset_x, screen_left, screen_left + screen_width - 1))
                mapped_y = float(np.clip(mapped_y_base + pan_offset_y, screen_top, screen_top + screen_height - 1))

                now = time.time()
                dt = max(0.001, now - last_frame_time)
                last_frame_time = now

                # ---- Mode-specific behaviour --------------------------------
                if gesture_enabled[0] and current_gesture_mode == GESTURE_SCROLL:
                    # Release any held drag button when leaving DRAG mode.
                    if drag_active:
                        pyautogui.mouseUp(button="left")
                        drag_active = False
                    hold_direction = None
                    hold_vector_x, hold_vector_y = 0.0, 0.0
                    hold_started_at = 0.0
                    hold_anchor = None
                    pan_offset_x = 0.0
                    pan_offset_y = 0.0

                    raw_scroll_y = float(index_tip_y)
                    if prev_scroll_y is not None:
                        delta_y = raw_scroll_y - prev_scroll_y
                        if abs(delta_y) > scroll_deadzone_px:
                            scroll_accumulator += delta_y / scroll_sensitivity
                            scroll_units = int(scroll_accumulator)
                            if scroll_units != 0:
                                pyautogui.scroll(-scroll_units)
                                scroll_accumulator -= scroll_units
                    prev_scroll_y = raw_scroll_y

                    cv2.putText(
                        frame,
                        "SCROLL MODE",
                        (frame_margin_x + 10, frame_margin_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 200, 0),
                        2,
                    )

                elif gesture_enabled[0] and current_gesture_mode == GESTURE_DRAG:
                    prev_scroll_y = None
                    scroll_accumulator = 0.0
                    move_delta = math.hypot(mapped_x - prev_x, mapped_y - prev_y)
                    if move_delta >= jitter_deadzone_px:
                        curr_x = prev_x + (mapped_x - prev_x) / smoothening
                        curr_y = prev_y + (mapped_y - prev_y) / smoothening
                    else:
                        curr_x, curr_y = prev_x, prev_y
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                    pinch_dist = math.hypot(index_tip_x - thumb_tip_x, index_tip_y - thumb_tip_y)
                    if pinch_dist < pinch_threshold:
                        if not drag_active:
                            pyautogui.mouseDown(button="left")
                            drag_active = True
                        cv2.putText(
                            frame,
                            "DRAG ACTIVE",
                            (frame_margin_x + 10, frame_margin_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 200, 255),
                            2,
                        )
                    else:
                        if drag_active:
                            pyautogui.mouseUp(button="left")
                            drag_active = False
                        cv2.putText(
                            frame,
                            "DRAG MODE",
                            (frame_margin_x + 10, frame_margin_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 200, 255),
                            2,
                        )

                else:
                    # MOVE mode (or IDLE/disabled).
                    prev_scroll_y = None
                    scroll_accumulator = 0.0
                    if drag_active:
                        pyautogui.mouseUp(button="left")
                        drag_active = False

                    pointing_direction_info = detect_point_direction(lm, frame_width, frame_height)

                    if not control_enabled or not gesture_enabled[0]:
                        hold_direction = None
                        hold_vector_x, hold_vector_y = 0.0, 0.0
                        hold_started_at = 0.0
                        hold_anchor = None
                    elif pointing_direction_info is None:
                        hold_direction = None
                        hold_vector_x, hold_vector_y = 0.0, 0.0
                        hold_started_at = 0.0
                        hold_anchor = None
                    else:
                        pointing_direction, dir_vx, dir_vy = pointing_direction_info
                        if hold_direction != pointing_direction:
                            hold_direction = pointing_direction
                            hold_vector_x, hold_vector_y = dir_vx, dir_vy
                            hold_started_at = now
                            hold_anchor = (index_tip_x, index_tip_y)
                        elif hold_anchor is not None:
                            anchor_dist = math.hypot(
                                index_tip_x - hold_anchor[0],
                                index_tip_y - hold_anchor[1],
                            )
                            if anchor_dist > hold_stationary_threshold_px:
                                hold_direction = None
                                hold_vector_x, hold_vector_y = 0.0, 0.0
                                hold_started_at = 0.0
                                hold_anchor = None

                    pan_active = False
                    if (
                        control_enabled
                        and gesture_enabled[0]
                        and hold_direction is not None
                        and hold_started_at > 0.0
                    ):
                        held_for = now - hold_started_at
                        if held_for >= hold_to_pan_seconds:
                            pan_active = True
                            ramp_elapsed = held_for - hold_to_pan_seconds
                            ramp = min(1.0, max(0.35, ramp_elapsed / pan_ramp_seconds))
                            step = pan_speed_px_per_sec * dt * ramp
                            dx = hold_vector_x * step
                            dy = hold_vector_y * step

                            pos_x, pos_y = pyautogui.position()
                            pan_x = float(np.clip(pos_x + dx, screen_left, screen_left + screen_width - 1))
                            pan_y = float(np.clip(pos_y + dy, screen_top, screen_top + screen_height - 1))
                            pan_offset_x = pan_x - mapped_x_base
                            pan_offset_y = pan_y - mapped_y_base
                            pyautogui.moveTo(pan_x, pan_y)
                            prev_x, prev_y = pan_x, pan_y
                            cv2.putText(
                                frame,
                                f"AUTO PAN: {hold_direction}",
                                (frame_margin_x + 10, frame_margin_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 255),
                                2,
                            )
                        else:
                            cv2.putText(
                                frame,
                                f"HOLD {hold_direction}: {held_for:.1f}/{hold_to_pan_seconds:.1f}s",
                                (frame_margin_x + 10, frame_margin_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (255, 255, 0),
                                2,
                            )

                    if control_enabled and gesture_enabled[0] and not pan_active:
                        move_delta = math.hypot(mapped_x - prev_x, mapped_y - prev_y)
                        if move_delta < jitter_deadzone_px:
                            curr_x, curr_y = prev_x, prev_y
                        else:
                            curr_x = prev_x + (mapped_x - prev_x) / smoothening
                            curr_y = prev_y + (mapped_y - prev_y) / smoothening
                        pyautogui.moveTo(curr_x, curr_y)
                        prev_x, prev_y = curr_x, curr_y
                    elif not control_enabled or not gesture_enabled[0]:
                        pos_x, pos_y = pyautogui.position()
                        prev_x, prev_y = float(pos_x), float(pos_y)

                # ---- Click detection (MOVE mode only) -----------------------
                if gesture_enabled[0] and current_gesture_mode == GESTURE_MOVE and control_enabled:
                    pinch_distance = math.hypot(index_tip_x - thumb_tip_x, index_tip_y - thumb_tip_y)
                    now_c = time.time()

                    if pinch_distance < pinch_threshold:
                        if now_c - last_click_time > click_cooldown:
                            pyautogui.click(button="left")
                            last_click_time = now_c
                            cv2.putText(
                                frame,
                                "L-CLICK",
                                (index_tip_x - 20, index_tip_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 80, 255),
                                2,
                            )

                    if is_right_click_pinch(lm, frame_width, frame_height, right_click_threshold):
                        if now_c - last_right_click_time > click_cooldown:
                            pyautogui.click(button="right")
                            last_right_click_time = now_c
                            cv2.putText(
                                frame,
                                "R-CLICK",
                                (thumb_tip_x + 10, thumb_tip_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 200, 255),
                                2,
                            )

                # Two-hand combo gestures.
                now = time.time()
                if len(hand_infos) >= 2:
                    left_labeled_hands = [h for h in hand_infos if h["label"] == "Left"]
                    right_labeled_hands = [h for h in hand_infos if h["label"] == "Right"]
                    if left_labeled_hands and right_labeled_hands:
                        left_hand = left_labeled_hands[0]
                        right_hand = right_labeled_hands[0]
                    else:
                        sorted_hands = sorted(hand_infos, key=lambda h: h["center_x"])
                        left_hand, right_hand = sorted_hands[0], sorted_hands[-1]

                    raw_left_x = left_hand["center_x"]
                    raw_right_x = right_hand["center_x"]

                    if prev_left_center_x is None:
                        left_x = raw_left_x
                    else:
                        left_x = (1.0 - center_smoothing_alpha) * prev_left_center_x + center_smoothing_alpha * raw_left_x

                    if prev_right_center_x is None:
                        right_x = raw_right_x
                    else:
                        right_x = (1.0 - center_smoothing_alpha) * prev_right_center_x + center_smoothing_alpha * raw_right_x

                    if prev_left_center_x is not None and prev_right_center_x is not None:
                        left_dx = left_x - prev_left_center_x
                        right_dx = right_x - prev_right_center_x
                        prev_gap = prev_right_center_x - prev_left_center_x
                        curr_gap = right_x - left_x
                        gap_change = curr_gap - prev_gap

                        both_two_fingers = (
                            left_hand["main_finger_count"] == 2 and right_hand["main_finger_count"] == 2
                        )
                        both_three_fingers = (
                            left_hand["main_finger_count"] == 3 and right_hand["main_finger_count"] == 3
                        )

                        moving_outward = (
                            left_dx < -two_hand_move_threshold_px
                            and right_dx > two_hand_move_threshold_px
                            and gap_change > two_hand_gap_threshold_px
                        )
                        moving_inward = (
                            left_dx > two_hand_move_threshold_px
                            and right_dx < -two_hand_move_threshold_px
                            and gap_change < -two_hand_gap_threshold_px
                        )

                        can_trigger_combo = (now - last_combo_action_time) >= gesture_action_cooldown

                        outward_two_score = min(
                            combo_confirm_frames,
                            outward_two_score + 1 if (both_two_fingers and moving_outward) else max(0, outward_two_score - 1),
                        )
                        inward_three_score = min(
                            combo_confirm_frames,
                            inward_three_score + 1 if (both_three_fingers and moving_inward) else max(0, inward_three_score - 1),
                        )
                        inward_two_score = min(
                            combo_confirm_frames,
                            inward_two_score + 1 if (both_two_fingers and moving_inward) else max(0, inward_two_score - 1),
                        )

                        if outward_two_score >= combo_confirm_frames and can_trigger_combo:
                            pyautogui.hotkey("win", "up")
                            last_combo_action_time = now
                            outward_two_score = 0
                            inward_three_score = 0
                            inward_two_score = 0
                            cv2.putText(
                                frame,
                                "2H OUTWARD: MAXIMIZE",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                        elif inward_three_score >= combo_confirm_frames and can_trigger_combo:
                            pyautogui.hotkey("win", "tab")
                            last_combo_action_time = now
                            outward_two_score = 0
                            inward_three_score = 0
                            inward_two_score = 0
                            cv2.putText(
                                frame,
                                "2H INWARD: WIN+TAB",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )
                        elif inward_two_score >= combo_confirm_frames and can_trigger_combo:
                            pyautogui.hotkey("win", "down")
                            last_combo_action_time = now
                            outward_two_score = 0
                            inward_three_score = 0
                            inward_two_score = 0
                            cv2.putText(
                                frame,
                                "2H INWARD: MINIMIZE",
                                (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                    prev_left_center_x = left_x
                    prev_right_center_x = right_x
                else:
                    prev_left_center_x = None
                    prev_right_center_x = None
                    outward_two_score = 0
                    inward_three_score = 0
                    inward_two_score = 0

            # Draw interaction boundary box.
            cv2.rectangle(
                frame,
                (frame_margin_x, frame_margin_y),
                (frame_width - frame_margin_x, frame_height - frame_margin_y),
                (255, 255, 0),
                2,
            )

            # Voice action banner (displayed for a few seconds after a command).
            if last_voice_action and time.time() < voice_action_display_until:
                cv2.putText(
                    frame,
                    f"VOICE: {last_voice_action}",
                    (frame_margin_x, frame_margin_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 160),
                    2,
                )

            # FRIDAY-style HUD overlay (drawn last so it sits on top).
            draw_hud(
                frame,
                mode=current_gesture_mode,
                va_listening=va.listening,
                va_available=va.available,
                last_cmd=va.last_command,
                last_response=va.last_response,
                control_enabled=current_control_enabled,
                gesture_enabled=gesture_enabled[0],
            )

            cv2.imshow("Sentinel Touch", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if drag_active:
            pyautogui.mouseUp(button="left")
        va.stop()
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
