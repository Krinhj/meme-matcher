"""
Index local memes by extracting facial emotion vectors and optional manual tags.

Usage:
    python index_memes.py --memes-dir memes --output memes_index.json
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np

# Prefer top-level import; some installs break on "from fer.fer import FER"
try:
    from fer import FER  # type: ignore
except Exception:  # pragma: no cover
    from fer.fer import FER  # fallback

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
EMOTION_ORDER = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index memes with facial emotion metadata.")
    p.add_argument("--memes-dir", type=Path, default=Path("memes"),
                   help="Directory containing meme images (default: memes).")
    p.add_argument("--output", type=Path, default=Path("memes_index.json"),
                   help="Path to the output JSON file (default: memes_index.json).")
    p.add_argument("--tags-file", type=Path, default=None,
                   help="Optional JSON file providing manual tags/notes per meme.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing output file.")
    p.add_argument("--mtcnn", action="store_true",
                   help="Use the slower but more accurate MTCNN face detector in FER.")
    p.add_argument("--verbose", action="store_true", help="Increase logging verbosity.")
    p.add_argument("--infer-gesture-from-dir", action="store_true",
                   help="Infer gesture tag from immediate parent directory name.")
    p.add_argument("--face-pick", choices=["largest","strongest"], default="largest",
                   help="How to choose the face if multiple are detected.")
    return p.parse_args()

def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

def find_images(memes_dir: Path) -> List[Path]:
    if not memes_dir.exists():
        raise FileNotFoundError(f"Meme directory not found: {memes_dir}")
    images = sorted(
        p for p in memes_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not images:
        logging.warning("No images found in %s", memes_dir)
    return images

def load_manual_tags(tags_file: Optional[Path], memes_dir: Path) -> Dict[str, Dict[str, Any]]:
    if not tags_file:
        return {}
    if not tags_file.exists():
        raise FileNotFoundError(f"Tags metadata file not found: {tags_file}")
    raw = json.loads(tags_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Tags file must contain a JSON object mapping filenames to metadata objects.")
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in raw.items():
        md = _validate_manual_entry(k, v)
        rel_key = _normalize_key(k, memes_dir)
        out[rel_key] = md
        out.setdefault(Path(rel_key).name, md)
    logging.info("Loaded manual tags for %d meme(s).", len(raw))
    return out

def _validate_manual_entry(key: str, value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Metadata for '{key}' must be an object.")
    tags = value.get("tags", [])
    gesture_tags = value.get("gesture_tags", [])
    notes = value.get("notes")
    if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
        raise ValueError(f"'tags' for '{key}' must be a list of strings.")
    if not isinstance(gesture_tags, list) or any(not isinstance(t, str) for t in gesture_tags):
        raise ValueError(f"'gesture_tags' for '{key}' must be a list of strings.")
    if notes is not None and not isinstance(notes, str):
        raise ValueError(f"'notes' for '{key}' must be a string if provided.")
    return {"tags": tags, "gesture_tags": gesture_tags, "notes": notes}

def _normalize_key(key: str, memes_dir: Path) -> str:
    p = Path(key)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(memes_dir).as_posix()
    except ValueError:
        return p.name

def _file_hash(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _pick_best_face(detections: List[Dict[str, Any]], mode: str = "largest") -> Dict[str, Any]:
    if not detections:
        return {}
    if mode == "strongest":
        def strength(d: Dict[str, Any]) -> float:
            emo = d.get("emotions") or {}
            return max(emo.values()) if emo else 0.0
        return max(detections, key=strength)
    # largest box (default)
    def area(d: Dict[str, Any]) -> float:
        x, y, w, h = d.get("box", [0,0,0,0])
        return float(max(w,0) * max(h,0))
    return max(detections, key=area)

def _to_emotion_vec(emotions: Dict[str, float]) -> List[float]:
    if not emotions:
        # neutral fallback
        vec = [0.0]*len(EMOTION_ORDER)
        vec[-1] = 1.0
        return vec
    vec = [float(emotions.get(k, 0.0)) for k in EMOTION_ORDER]
    s = sum(vec)
    if s <= 0:
        vec = [0.0]*len(EMOTION_ORDER); vec[-1] = 1.0
        return vec
    return [v/s for v in vec]

def process_image(
    path: Path,
    base_dir: Path,
    detector: FER,
    manual_metadata: Dict[str, Dict[str, Any]],
    infer_gesture_from_dir: bool,
    face_pick_mode: str,
) -> Dict[str, Any]:
    rel = path.relative_to(base_dir).as_posix()
    logging.debug("Processing %s", rel)

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        logging.warning("Skipping %s (unreadable; check codec support for GIF/WEBP).", rel)
        return _build_record(rel, path.name, None, {}, path)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        detections = detector.detect_emotions(rgb)
    except Exception as exc:  # very broad—FER backend differences
        logging.error("FER detection failed for %s: %s", rel, exc)
        detection = {"error": str(exc), "emotions": {}}
    else:
        detection = _pick_best_face(detections, mode=face_pick_mode) if detections else None

    manual = manual_metadata.get(rel) or manual_metadata.get(path.name) or {}
    # Optional gesture inference from parent directory name
    if infer_gesture_from_dir:
        parent = Path(rel).parent.name
        if parent and parent not in manual.get("gesture_tags", []):
            # simple heuristic: assume parent is a gesture tag
            manual = {
                **manual,
                "gesture_tags": list({*manual.get("gesture_tags", []), parent})
            }

    rec = _build_record(rel, path.name, detection, manual, path)
    rec["image_size"] = {"width": int(img.shape[1]), "height": int(img.shape[0])}
    rec["indexed_at"] = datetime.now(timezone.utc).isoformat()
    return rec

def _build_record(
    relative_path: str,
    file_name: str,
    detection: Optional[Dict[str, Any]],
    manual_metadata: Dict[str, Any],
    file_path: Path,
) -> Dict[str, Any]:
    emotions = (detection or {}).get("emotions") or {}
    emotion_vec = _to_emotion_vec(emotions)
    dominant = _dominant_emotion(emotions) if emotions else None

    try:
        st = file_path.stat()
        file_meta = {
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "sha1": _file_hash(file_path)
        }
    except Exception:
        file_meta = {"size": None, "mtime": None, "sha1": None}

    return {
        "file_name": file_name,
        "relative_path": relative_path,
        "detected_face": bool(detection) and not (detection or {}).get("error"),
        "emotions": emotions,
        "emotion_vec": emotion_vec,            # cosine-ready, follows EMOTION_ORDER
        "dominant_emotion": dominant,
        "bbox": _serialize_bbox((detection or {}).get("box")),
        "tags": manual_metadata.get("tags", []),
        "gesture_tags": manual_metadata.get("gesture_tags", []),
        "notes": manual_metadata.get("notes"),
        "errors": (detection or {}).get("error"),
        "file": file_meta,
    }

def _dominant_emotion(emotions: Dict[str, float]) -> Dict[str, Any]:
    label, score = max(emotions.items(), key=lambda kv: kv[1])
    return {"label": label, "score": float(score)}


def _serialize_bbox(box: Any) -> List[int] | None:
    if box is None:
        return None
    if isinstance(box, np.ndarray):
        return [int(v) for v in box.tolist()]
    if isinstance(box, (list, tuple)):
        return [int(v) for v in box]
    return None

def write_output(records: List[Dict[str, Any]], output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "emotion_order": EMOTION_ORDER,
        "items": records,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Wrote %d meme record(s) to %s", len(records), output_path)

def _make_detector(mtcnn: bool) -> FER:
    # Graceful fallback if MTCNN can't be created due to missing TF
    try:
        return FER(mtcnn=mtcnn)
    except Exception as exc:
        if mtcnn:
            logging.warning("MTCNN requested but unavailable (%s). Falling back to mtcnn=False.", exc)
            return FER(mtcnn=False)
        raise

def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    images = find_images(args.memes_dir)
    if not images:
        logging.warning("Nothing to index. Exiting.")
        return 0

    manual_metadata = load_manual_tags(args.tags_file, args.memes_dir)
    detector = _make_detector(args.mtcnn)

    # Sequential is fine for small sets; parallelize if needed.
    records = [
        process_image(
            path, args.memes_dir, detector, manual_metadata,
            infer_gesture_from_dir=args.infer_gesture_from_dir,
            face_pick_mode=args.face_pick
        )
        for path in images
    ]

    write_output(records, args.output, args.overwrite)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
