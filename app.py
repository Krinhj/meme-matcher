"""
Meme Matcher live viewer.

This script opens the default webcam, mirrors the feed for a natural selfie view,
and shows a split-screen window with a placeholder meme panel. It is the baseline
for plugging in emotion detection + meme matching logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WINDOW_TITLE = "Meme Matcher"
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
MIN_PANEL_WIDTH = 320
MIN_PANEL_HEIGHT = 240
PLACEHOLDER_TEXT = "No meme matched yet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Meme Matcher live viewer.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Index of the camera to use (default: 0).",
    )
    parser.add_argument(
        "--window-width",
        "--panel-width",
        dest="window_width",
        type=int,
        default=DEFAULT_WINDOW_WIDTH,
        help="Total width of the split-screen window (default: 1920).",
    )
    parser.add_argument(
        "--window-height",
        "--panel-height",
        dest="window_height",
        type=int,
        default=DEFAULT_WINDOW_HEIGHT,
        help="Total height of the split-screen window (default: 1080).",
    )
    parser.add_argument(
        "--memes-dir",
        type=Path,
        default=Path("memes"),
        help="Path to the local meme folder (optional for future matching).",
    )
    return parser.parse_args()


def _dynamic_font_metrics(width: int, height: int) -> tuple[float, float, int]:
    """Derive font scales that look good across many panel sizes."""
    reference = min(width, height)
    main_scale = max(0.5, reference / 700.0)
    sub_scale = max(0.4, reference / 1050.0)
    thickness = max(1, int(reference / 450))
    return main_scale, sub_scale, thickness


def _screen_resolution() -> tuple[int, int]:
    """Best-effort detection of the primary display size for clamping window dimensions."""
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception:
        return 1920, 1080


def clamp_window_dimensions(width: int, height: int) -> tuple[int, int]:
    """Ensure the total window fits on screen."""
    screen_w, screen_h = _screen_resolution()
    horizontal_margin = 80
    vertical_margin = 120
    max_width = max(MIN_PANEL_WIDTH * 2, screen_w - horizontal_margin)
    max_height = max(MIN_PANEL_HEIGHT, screen_h - vertical_margin)
    return min(width, max_width), min(height, max_height)


def build_placeholder(width: int, height: int, text: str = PLACEHOLDER_TEXT) -> np.ndarray:
    """Create a neutral gray canvas with centered placeholder text."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)

    font = cv2.FONT_HERSHEY_SIMPLEX
    main_scale, sub_scale, thickness = _dynamic_font_metrics(width, height)

    text_size = cv2.getTextSize(text, font, main_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(panel, text, (text_x, text_y), font, main_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    subtext = "Waiting for confident match..."
    sub_size = cv2.getTextSize(subtext, font, sub_scale, max(1, thickness - 1))[0]
    sub_x = (width - sub_size[0]) // 2
    sub_y = text_y + int(40 * main_scale)
    cv2.putText(panel, subtext, (sub_x, sub_y), font, sub_scale, (200, 200, 200), max(1, thickness - 1), cv2.LINE_AA)
    return panel


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a frame to fit the target panel dimensions."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def compose_split_screen(
    webcam_frame: np.ndarray,
    meme_panel: np.ndarray,
) -> np.ndarray:
    """Concatenate the webcam and meme panels side-by-side."""
    return np.hstack((webcam_frame, meme_panel))


def annotate_frame(frame: np.ndarray, message: str) -> None:
    """Overlay a small instruction string onto the webcam pane."""
    cv2.putText(
        frame,
        message,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def run_viewer(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam. Try a different --camera-index.", file=sys.stderr)
        return 1

    window_width, window_height = clamp_window_dimensions(args.window_width, args.window_height)
    if (window_width, window_height) != (args.window_width, args.window_height):
        print(
            f"Adjusted window size to {window_width}x{window_height} so it fits on your screen.",
            file=sys.stderr,
        )

    panel_width = max(MIN_PANEL_WIDTH, window_width // 2)
    panel_height = max(MIN_PANEL_HEIGHT, window_height)

    placeholder_panel = build_placeholder(panel_width, panel_height)
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, window_width, panel_height)

    try:
        while True:
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to grab frame from webcam.", file=sys.stderr)
                break

            frame = cv2.flip(frame, 1)  # Mirror for selfie view
            resized_webcam = resize_frame(frame, panel_width, panel_height)
            annotate_frame(resized_webcam, "Press 'Q' to quit")

            # Future work: replace placeholder_panel with matched meme frame when available.
            composite = compose_split_screen(resized_webcam, placeholder_panel)
            cv2.imshow(WINDOW_TITLE, composite)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    cli_args = parse_args()
    raise SystemExit(run_viewer(cli_args))
