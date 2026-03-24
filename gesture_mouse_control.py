import math
import time
import ctypes

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


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


def main() -> None:
    # Safety feature: moving mouse to a screen corner can stop automation.
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0

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

    start_x, start_y = pyautogui.position()
    prev_x, prev_y = float(start_x), float(start_y)
    last_click_time = 0.0

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

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Mirror view for natural interaction.
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Dynamic active area centered in frame to reduce required hand movement.
            active_w = int(frame_width * active_area_ratio)
            active_h = int(frame_height * active_area_ratio)
            frame_margin_x = (frame_width - active_w) // 2
            frame_margin_y = (frame_height - active_h) // 2

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

                # Direction hold logic for long cursor travel with less hand motion.
                pointing_direction_info = detect_point_direction(lm, frame_width, frame_height)
                now = time.time()
                dt = max(0.001, now - last_frame_time)
                last_frame_time = now

                if not control_enabled:
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
                        anchor_dist = math.hypot(index_tip_x - hold_anchor[0], index_tip_y - hold_anchor[1])
                        if anchor_dist > hold_stationary_threshold_px:
                            hold_direction = None
                            hold_vector_x, hold_vector_y = 0.0, 0.0
                            hold_started_at = 0.0
                            hold_anchor = None

                pan_active = False
                if control_enabled and hold_direction is not None and hold_started_at > 0.0:
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
                        # Keep panned position as new reference so stop-pan does not snap back.
                        pan_offset_x = pan_x - mapped_x_base
                        pan_offset_y = pan_y - mapped_y_base
                        pyautogui.moveTo(pan_x, pan_y)
                        prev_x, prev_y = pan_x, pan_y
                        cv2.putText(
                            frame,
                            f"AUTO PAN: {hold_direction}",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                        )
                    else:
                        cv2.putText(
                            frame,
                            f"HOLD {hold_direction}: {held_for:.1f}/{hold_to_pan_seconds:.1f}s",
                            (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2,
                        )

                if control_enabled and not pan_active:
                    # Deadzone + low-pass smoothing for better jitter suppression.
                    move_delta = math.hypot(mapped_x - prev_x, mapped_y - prev_y)
                    if move_delta < jitter_deadzone_px:
                        curr_x, curr_y = prev_x, prev_y
                    else:
                        curr_x = prev_x + (mapped_x - prev_x) / smoothening
                        curr_y = prev_y + (mapped_y - prev_y) / smoothening
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                elif not control_enabled:
                    # Stay idle while safety gesture is off; keep cursor references synced.
                    pos_x, pos_y = pyautogui.position()
                    prev_x, prev_y = float(pos_x), float(pos_y)
                    cv2.putText(
                        frame,
                        "SAFE OFF: show only index finger",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
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

                # Left-click when thumb and index finger pinch together.
                pinch_distance = math.hypot(index_tip_x - thumb_tip_x, index_tip_y - thumb_tip_y)
                if control_enabled and pinch_distance < pinch_threshold:
                    now = time.time()
                    if now - last_click_time > click_cooldown:
                        pyautogui.click(button="left")
                        last_click_time = now
                        cv2.putText(
                            frame,
                            "CLICK",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            3,
                        )

            # Draw interaction boundary box.
            cv2.rectangle(
                frame,
                (frame_margin_x, frame_margin_y),
                (frame_width - frame_margin_x, frame_height - frame_margin_y),
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "2F out=maximize | 2F in=minimize | 3F in=Win+Tab",
                (20, frame_height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "Auto-pan supports 8 directions (including diagonals)",
                (20, frame_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (230, 230, 230),
                2,
            )

            cv2.imshow("Gesture Mouse Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
