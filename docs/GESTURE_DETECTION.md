# Gesture Detection Implementation Guide

## Overview

Gesture detection has been successfully integrated into the Meme Matcher app to improve matching accuracy by combining facial emotion recognition with body/hand pose detection.

## Architecture

### Components

1. **`gesture_detector.py`** - Core gesture detection module
   - Uses OpenCV Haar Cascades for face detection
   - Implements skin color segmentation (YCrCb color space) for hand detection
   - Spatial heuristics to classify gestures based on hand positions relative to face
   - Compatible with Python 3.13+ (no MediaPipe dependency)

2. **`app.py`** - Main viewer with gesture integration
   - Optional gesture detection via `--enable-gestures` flag
   - Configurable scoring weights for emotion vs gesture
   - Real-time gesture display in match overlay

3. **`test_gestures.py`** - Standalone testing tool
   - Visual feedback with bounding boxes
   - Real-time gesture classification
   - Useful for debugging and calibration

## Supported Gestures

| Gesture | Detection Logic | Use Case |
|---------|----------------|----------|
| **hands_up** | Hands detected significantly above face | Celebration, excitement, "absolute cinema" |
| **temple_tap** | Index finger near temple/side of head at eye level | "Roll Safe" thinking pose, smart/clever |
| **thinking** | Hand near chin/mouth area | Contemplative, pondering, decision-making |
| **eyebrow_raise** | Eyebrow-to-eye distance exceeds threshold | The Rock's signature move, skeptical, suspicious |
| **neutral** | No specific gesture detected | Default fallback |

## Usage

### 1. Tag Your Memes

**Option A: Manual tagging** (recommended for accuracy)

Create or edit `docs/manual_tags.json`:

```json
{
  "thinking_monkey.png": {
    "tags": ["thinking", "contemplative"],
    "gesture_tags": ["thinking"],
    "notes": "Hand near chin pose"
  },
  "roll_safe.jpg": {
    "tags": ["smart", "confident"],
    "gesture_tags": ["temple_tap"],
    "notes": "Finger pointing at temple"
  }
}
```

Then index with tags:
```bash
python index_memes.py --tags-file docs/manual_tags.json --overwrite
```

**Option B: Directory-based inference**

Organize memes into gesture-named folders:
```
memes/
  thinking/
    monkey_thinking.png
    person_pondering.jpg
  temple_tap/
    roll_safe.jpg
  hands_up/
    absolute_cinema.png
```

Then index with directory inference:
```bash
python index_memes.py --infer-gesture-from-dir --overwrite
```

### 2. Run with Gesture Detection

**Basic usage:**
```bash
python app.py --enable-gestures
```

**Custom weights:**
```bash
python app.py --enable-gestures --emotion-weight 0.6 --gesture-weight 0.35
```

**Test gestures first:**
```bash
python test_gestures.py
```

## Scoring Algorithm

The matching score combines multiple signals:

```python
combined_score = (
    emotion_weight * emotion_similarity +    # default: 0.7
    gesture_weight * gesture_overlap +       # default: 0.25
    clip_weight * clip_similarity            # default: 0.05 (reserved for future)
)
```

### Emotion Similarity
- Cosine similarity between user's emotion vector and meme's emotion vector
- Range: -1.0 to 1.0 (typically 0.0 to 1.0)

### Gesture Overlap
- Jaccard similarity: `intersection / union` of gesture tag sets
- Range: 0.0 to 1.0
- Examples:
  - User: `['thinking']`, Meme: `['thinking']` → 1.0
  - User: `['thinking', 'neutral']`, Meme: `['thinking']` → 0.5
  - User: `['hands_up']`, Meme: `['thinking']` → 0.0

## Technical Details

### Gesture Detection Pipeline

1. **Face Detection**
   - Haar Cascade classifier (frontal face)
   - Selects largest face if multiple detected
   - Provides spatial reference for hand position analysis

2. **Hand Detection**
   - Convert frame to YCrCb color space
   - Apply skin color mask (calibrated for various skin tones)
   - Morphological operations to reduce noise
   - Contour detection and filtering by size/aspect ratio
   - Keep top 2 largest hand regions

3. **Gesture Classification**
   - Calculate hand positions relative to face bounding box
   - Apply spatial heuristics:
     - **hands_up**: hand center Y < face top - 0.3 * face height
     - **temple_tap**: hand at eye level, to the side of face
     - **thinking**: hand near chin (face center Y to bottom + margin)

### Performance Considerations

- Gesture detection runs on the same frame as emotion detection
- Controlled by `--analyze-interval` (default: every 5 frames)
- Minimal CPU overhead (~10-15ms per frame on modern hardware)
- Skin detection works best with good lighting

### Limitations & Future Improvements

**Current Limitations:**
- Skin color detection can be affected by lighting conditions
- No finger/hand pose detection (only position-based)
- Works best with clear, unobstructed hand visibility
- Single-person detection only

**Future Enhancements:**
- Upgrade to MediaPipe Holistic (requires Python 3.12)
  - 33 pose landmarks + 21 hand landmarks per hand
  - More accurate finger tracking (e.g., index finger for temple_tap)
  - Better handling of occlusions
- Add more gesture types (peace sign, thumbs up, etc.)
- Temporal smoothing for gesture detection (similar to emotion smoothing)
- Multi-person gesture detection

## Troubleshooting

### Gestures not detected
1. Check lighting - ensure hands are well-lit
2. Run `test_gestures.py` to see what's being detected
3. Verify face is detected (green box should appear)
4. Adjust hand position - try more exaggerated gestures

### False positives
1. Increase `--similarity-threshold` to require higher confidence
2. Adjust `--gesture-weight` lower (e.g., 0.15) to rely more on emotions
3. Use manual tagging instead of directory inference for better accuracy

### Performance issues
1. Increase `--analyze-interval` to analyze fewer frames
2. Disable gestures if not needed: remove `--enable-gestures` flag
3. Use smaller window size: `--window-width 1280 --window-height 720`

## Example Workflows

### Workflow 1: Emotion-only matching (default)
```bash
python index_memes.py --memes-dir memes --output memes_index.json --overwrite
python app.py
```

### Workflow 2: Emotion + gesture matching
```bash
# Index with gesture tags
python index_memes.py --infer-gesture-from-dir --overwrite

# Run with gestures enabled
python app.py --enable-gestures
```

### Workflow 3: Fine-tuned matching
```bash
# Manual tagging for precision
python index_memes.py --tags-file docs/manual_tags.json --mtcnn --overwrite

# Custom weights favoring gestures
python app.py --enable-gestures --emotion-weight 0.5 --gesture-weight 0.45
```

## API Reference

### GestureDetector Class

```python
from gesture_detector import GestureDetector

detector = GestureDetector(
    min_detection_confidence=0.5,  # Unused in OpenCV impl
    min_tracking_confidence=0.5,   # Unused in OpenCV impl
    enable_face_mesh=True,         # Enable face detection
)

# Detect gestures in a frame
result = detector.detect(frame)  # frame: BGR numpy array

# Access results
print(result.gesture_tags)        # ['thinking', ...]
print(result.confidence)          # 0.65
print(result.landmarks_detected)  # {'face': True, 'hands': True, ...}

# Cleanup
detector.close()
```

### gesture_overlap_score Function

```python
from gesture_detector import gesture_overlap_score

score = gesture_overlap_score(
    user_gestures=['thinking', 'neutral'],
    meme_gestures=['thinking']
)
# Returns: 0.5 (Jaccard similarity)
```

## Contributing

When adding new gesture types:

1. Define detection logic in `GestureDetector._detect_pose_gestures()`
2. Add gesture name to supported list in docstring
3. Update `docs/manual_tags_example.json` with examples
4. Test with `test_gestures.py`
5. Update this guide with the new gesture

## References

- [OpenCV Haar Cascades](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [Skin Detection in YCrCb](https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html)
- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) (future upgrade)
