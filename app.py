"""
Meme Matcher live viewer.

Opens the default webcam, mirrors the feed, and shows a split-screen window with
either the best-matching meme (based on FER emotion vectors) or a friendly
placeholder when confidence is too low.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:  # Prefer top-level import; some installs only expose fer.fer.FER
    from fer import FER  # type: ignore
except Exception:  # pragma: no cover
    try:
        from fer.fer import FER  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FER is required for the live matcher. Please `pip install fer` inside your venv."
        ) from exc

WINDOW_TITLE = "Meme Matcher"
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
MIN_PANEL_WIDTH = 320
MIN_PANEL_HEIGHT = 240
PLACEHOLDER_TEXT = "No meme matched yet"
MEME_BG_COLOR = (35, 35, 35)
EMOTION_ORDER: Sequence[str] = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)


@dataclass
class MemeEntry:
    name: str
    panel: np.ndarray
    vector: np.ndarray
    norm_vector: np.ndarray


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
    parser.add_argument(
        "--index-file",
        type=Path,
        default=Path("memes_index.json"),
        help="JSON index produced by index_memes.py (default: memes_index.json).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Minimum cosine similarity (0-1) required to show a meme.",
    )
    parser.add_argument(
        "--analyze-interval",
        type=int,
        default=5,
        help="Run FER on every Nth frame to save CPU (default: 5).",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Average this many recent emotion vectors before matching (default: 5).",
    )
    parser.add_argument(
        "--mtcnn",
        action="store_true",
        help="Use the slower but slightly more accurate MTCNN detector inside FER.",
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


def _fit_image_to_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize an image with letterboxing so it fills the meme pane nicely."""
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = MEME_BG_COLOR
    if image is None or image.size == 0:
        return panel
    h, w = image.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x = (width - new_w) // 2
    y = (height - new_h) // 2
    panel[y : y + new_h, x : x + new_w] = resized
    return panel


def _caption_panel(panel: np.ndarray, caption: str) -> np.ndarray:
    labeled = panel.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(panel.shape[0], panel.shape[1]) / 900.0)
    thickness = max(1, int(scale * 2))
    text_size = cv2.getTextSize(caption, font, scale, thickness)[0]
    margin = 16
    x = max(margin, (panel.shape[1] - text_size[0]) // 2)
    y = panel.shape[0] - margin
    cv2.putText(
        labeled,
        caption,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return labeled


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


def _normalize_vector(values: Iterable[float]) -> Optional[np.ndarray]:
    vec = np.array(list(values), dtype=np.float32)
    if vec.size == 0:
        return None
    norm = np.linalg.norm(vec)
    if not norm:
        return None
    return vec / norm


def load_meme_entries(
    index_path: Path, memes_dir: Path, panel_width: int, panel_height: int
) -> List[MemeEntry]:
    """Load meme vectors + pre-rendered panels from the JSON index."""
    if not index_path.exists():
        print(f"Warning: index file '{index_path}' not found. Showing placeholder only.", file=sys.stderr)
        return []
    if not memes_dir.exists():
        print(f"Warning: memes directory '{memes_dir}' not found. Showing placeholder only.", file=sys.stderr)
        return []

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: could not parse {index_path}: {exc}", file=sys.stderr)
        return []

    emotion_count = len(EMOTION_ORDER)
    entries: List[MemeEntry] = []
    for item in data.get("items", []):
        vec = np.array(item.get("emotion_vec", []), dtype=np.float32)
        if vec.size != emotion_count:
            continue
        norm_vec = _normalize_vector(vec)
        if norm_vec is None:
            continue
        rel_path = Path(item.get("relative_path", item.get("file_name", "")))
        full_path = (memes_dir / rel_path).resolve()
        image = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: skipped {rel_path} (image missing or unreadable).", file=sys.stderr)
            continue
        panel = _fit_image_to_panel(image, panel_width, panel_height)
        caption = item.get("file_name") or rel_path.name
        panel = _caption_panel(panel, caption)
        entries.append(MemeEntry(name=caption, panel=panel, vector=vec, norm_vector=norm_vec))

    if not entries:
        print("Warning: no valid meme entries found in the index.", file=sys.stderr)
    else:
        print(f"Loaded {len(entries)} meme entries from {index_path}.")
    return entries


def _select_face(detections: Sequence[dict]) -> Optional[dict]:
    if not detections:
        return None
    return max(
        detections,
        key=lambda d: max(d.get("box", [0, 0, 0, 0])[2], 0) * max(d.get("box", [0, 0, 0, 0])[3], 0),
    )


def detect_emotion_vector(detector: FER, frame: np.ndarray) -> Optional[np.ndarray]:
    """Run FER on the current frame and return a normalized emotion vector."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        detections = detector.detect_emotions(rgb)
    except Exception as exc:
        print(f"FER failed on a frame: {exc}", file=sys.stderr)
        return None

    face = _select_face(detections)
    if not face:
        return None

    emotions = face.get("emotions") or {}
    vec = np.array([float(emotions.get(label, 0.0)) for label in EMOTION_ORDER], dtype=np.float32)
    if not np.any(vec):
        return None
    return _normalize_vector(vec)


def find_best_match(face_vec: np.ndarray, memes: Sequence[MemeEntry]) -> Tuple[Optional[MemeEntry], float]:
    best_entry: Optional[MemeEntry] = None
    best_score = -1.0
    for entry in memes:
        score = float(np.dot(face_vec, entry.norm_vector))
        if score > best_score:
            best_entry = entry
            best_score = score
    return best_entry, best_score


def _smoothed_vector(buffer: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    if not buffer:
        return None
    stacked = np.stack(buffer)
    mean_vec = stacked.mean(axis=0)
    return _normalize_vector(mean_vec)


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
    meme_entries = load_meme_entries(args.index_file, args.memes_dir, panel_width, panel_height)
    detector = FER(mtcnn=args.mtcnn)
    analyze_interval = max(1, args.analyze_interval)
    smoothing_window = max(1, args.smoothing_window)
    vector_buffer: deque[np.ndarray] = deque(maxlen=smoothing_window)

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, window_width, panel_height)

    active_panel = placeholder_panel
    active_label: Optional[str] = None
    frame_counter = 0

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
            frame_counter += 1

            if meme_entries and frame_counter % analyze_interval == 0:
                vec = detect_emotion_vector(detector, frame)
                if vec is None:
                    vector_buffer.clear()
                    active_panel = placeholder_panel
                    active_label = None
                else:
                    vector_buffer.append(vec)
                    smooth_vec = _smoothed_vector(vector_buffer)
                    if smooth_vec is None:
                        active_panel = placeholder_panel
                        active_label = None
                        continue
                    best_entry, score = find_best_match(smooth_vec, meme_entries)
                    if best_entry and score >= args.similarity_threshold:
                        active_panel = best_entry.panel
                        active_label = f"{best_entry.name} ({score:.2f})"
                    else:
                        active_panel = placeholder_panel
                        active_label = None

            status = f"Match: {active_label}" if active_label else "Match: (none yet)"
            annotate_frame(resized_webcam, f"{status} | Press 'Q' to quit")

            composite = compose_split_screen(resized_webcam, active_panel)
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
