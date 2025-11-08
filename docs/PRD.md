# Meme Matcher - Product Requirements Document (PRD)

## 1. Overview

**Meme Matcher** is a fun computer vision desktop app that matches your facial expression and gestures to a meme from a local meme folder. The app uses emotion detection, optional gesture cues, and similarity scoring to find the closest vibe meme.

### Core Concept

You take a picture (or use your webcam) and the app detects your expression, then finds which meme best matches your mood.

---

## 2. Goals

- Recognize user facial expressions and basic gestures.
- Match those to memes in a local folder (e.g., "Squint Fry", "Thinking Monkey").
- Show the top meme match instantly in a desktop window.
- Keep everything local (no cloud uploads).
- Remain lightweight and easy to run for fun and experimentation.
- Launching the script should simply open the viewer; closing the window ends the session.

---

## 3. Core Features

| Feature                          | Description                                                               |
| -------------------------------- | ------------------------------------------------------------------------- |
| **Facial Emotion Recognition**   | Detects emotions like happy, sad, angry, neutral, surprised, etc.         |
| **Gesture Detection (Optional)** | Detects simple gestures like hands-up, temple-tap, or thinking pose.      |
| **Meme Indexing Script**         | Preprocesses meme folder and stores emotion/gesture tags.                 |
| **Similarity Scoring**           | Uses cosine similarity to compare emotion vectors between user and memes. |
| **Desktop Split-Screen Viewer**  | Split-screen window with live webcam feed on one side and the matched meme (or graceful no-match message) on the other. |
| **Privacy Mode**                 | Fully local processing. No external APIs or cloud calls.                  |

---

## 4. Extended Features (Optional)

| Feature                     | Description                                                      |
| --------------------------- | ---------------------------------------------------------------- |
| **CLIP Embedding Matching** | Use CLIP image embeddings to improve matches for stylized memes. |
| **Manual Tagging UI**       | Add custom tags like "smug", "confused", or "thinking".         |
| **Top-K Result Grid**       | Display top 3-5 closest memes ranked by similarity.              |
| **Download Match Card**     | Combine user photo + meme for sharing.                           |

---

## 5. User Flow

### Step 1: Meme Indexing

1. Add memes to the `/memes` folder.
2. Run `index_memes.py`.
   - Extracts facial emotion vectors from each meme and stores normalized `emotion_vec` arrays for cosine scoring.
   - Detects gestures if visible (manual tags or parent-folder inference).
   - Stores results in `memes_index.json` with schema version, timestamps, and file metadata.
   - Optional: merge manual tags/notes via JSON (`tags`, `gesture_tags`, `notes` fields) for memes with ambiguous expressions.

### Step 2: Desktop App Usage

1. Launch the desktop app: `python app.py` (no web interface; a window opens immediately).
2. The webcam feed starts automatically (live-only, no upload mode).
3. The viewer loads `memes_index.json`, pre-renders each meme panel, and then runs FER on every _N_th frame (configurable via `--analyze-interval`).
4. The viewer averages the last `--smoothing-window` emotion vectors to cut down jitter, then compares the smoothed vector against the cached meme vectors; if the best cosine score clears the `--similarity-threshold`, that meme is shown, otherwise the placeholder stays visible.
5. The window renders:
   - Left/Top half: live webcam preview with an overlay that states the latest meme match or “none yet”.
   - Right/Bottom half: current matched meme or the friendly "No meme matched yet" placeholder.
   - Optional overlay text such as emotion breakdown or match score (future enhancement).

### UI Layout

- Split-screen layout (horizontal or vertical) with the live webcam preview occupying one half and the selected meme occupying the other.
- The webcam side stays live to encourage real-time expression changes; the meme pane updates instantly as scores change.
- Display a friendly "No meme matched yet" placeholder in the meme pane until a confident match appears.
- Additional UI elements (emotion stats, controls) sit unobtrusively around or below the split so core visuals stay primary.
- Close the window to stop the session; no extra controls are required for v1.

---

## 6. Technical Architecture

**Desktop Display (OpenCV HighGUI or lightweight GUI toolkit)**
- Live webcam capture
- Split-screen rendering + overlay text/graphics

**Processing Backend (Python)**
- Emotion and optional gesture detection
- Meme index lookup + similarity scoring (NumPy / scikit-learn)

**Storage**
- Local filesystem

---

## 7. Matching Algorithm

```python
score = (
    0.7 * cosine(emotion_user, emotion_meme)
    + 0.25 * gesture_overlap(user, meme)
    + 0.05 * clip_similarity(user, meme)  # optional
)
```

> **Current implementation:** the live viewer already uses the emotion cosine term with a configurable threshold; gesture overlap and CLIP blending remain optional roadmap items for later phases.

**Gesture Tags**

| Tag          | Detection Heuristic               |
| ------------ | --------------------------------- |
| `hands_up`   | Both wrists above shoulders       |
| `temple_tap` | Index finger near temple landmark |
| `thinking`   | Hand near mouth/chin              |
| `neutral`    | No gesture detected               |

---

## 8. Example Meme Set

| Meme                  | Emotion             | Gesture      | Description          |
| --------------------- | ------------------- | ------------ | -------------------- |
| Fry "Not Sure If"     | Skeptical / Neutral | None         | Squinty "sus" look   |
| Absolute Cinema       | Awe / Pride         | Hands up     | "Masterpiece moment" |
| Roll Safe             | Confident           | Temple tap   | "Smart move"         |
| Thinking Monkey       | Pensive             | Hand on chin | "Deep in thought"    |
| Monkey Idea           | Excited             | One hand up  | "Eureka!"            |

---

## 9. Tech Stack

| Layer      | Technology                                        |
| ---------- | ------------------------------------------------- |
| Frontend   | OpenCV HighGUI window (optionally PyQt/PySide)    |
| Backend    | OpenCV, FER, MediaPipe (optional gestures)        |
| Storage    | Local files (JSON, image folders)                 |
| Similarity | NumPy / scikit-learn                              |
| Optional   | CLIP via Hugging Face Transformers                |
| Deployment | Local script launched from terminal               |

---

## 10. Risks & Mitigations

| Risk              | Mitigation                               |
| ----------------- | ---------------------------------------- |
| Face not detected | Manual tagging or fallback CLIP          |
| Gesture noise     | Temporal smoothing / majority voting     |
| Meme mismatch     | Adjust similarity weights                |
| CPU load          | Use lightweight FER model (no TensorFlow)|
| Privacy           | No image upload; keep processing local   |
| Gesture tooling compatibility | MediaPipe lacks Py3.13 wheels today; pin to Python 3.12 or swap in alternative landmark libraries until official support ships |

---

## 11. Metrics of Success

| Metric                  | Target                  |
| ----------------------- | ----------------------- |
| Meme match satisfaction | >= 70% (subjective fun) |
| Indexing speed          | < 1 s per meme          |
| Response latency        | < 300 ms                |
| Supported memes         | >= 500 images           |

---

## 12. Deliverables

| Deliverable        | Description               |
| ------------------ | ------------------------- |
| `index_memes.py`   | Meme indexing script      |
| `app.py`           | Desktop split-screen viewer |
| `memes_index.json` | Indexed meme metadata     |
| `/memes/`          | Folder of meme images     |
| `README.md`        | Setup + run guide         |
| `docs/PHASES.md`   | Step-by-step implementation roadmap |

---

## 13. Future Directions

- Add emotion + text pairing for meme captions.
- Crowd-source meme collections.
- Build a mobile PWA camera version.
- Offer "mood history" tracking for fun analytics.

## 14. Learning Objectives

- **End-to-end CV practice** - capture webcam frames, pre-process them, run FER/landmark models, and interpret confidence scores in real time.
- **Gesture heuristics** - experiment with MediaPipe or lightweight keypoint rules to derive tags like `hands_up` or `temple_tap`.
- **Similarity tuning** - encode meme/user emotions as vectors, blend scores with gesture overlap, and tweak weights/thresholds to minimize false matches.
- **Multimodal embeddings** - optionally add CLIP to see how modern vision-language representations complement rule-based scoring.
- **Dataset curation** - automate meme-feature extraction, cache outputs in `memes_index.json`, and iterate on tagging quality.
- **Realtime UX constraints** - balance latency (<300 ms), smoothing, and split-screen rendering to make the experience feel responsive for shareable demos.

This is a fun, local-first side project built for laughter, expression, and a bit of AI magic.
