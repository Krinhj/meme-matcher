# Meme Matcher

Meme Matcher is a local-first, split-screen desktop app that pairs your live webcam expressions with the closest meme from a local collection - perfect for quick LinkedIn or Instagram story demos while learning a bit of computer vision.

## Highlights

- Real-time webcam capture rendered next to the best meme match, with a friendly "No meme matched yet" state until a confident result appears.
- Facial emotion detection plus optional gesture cues, blended into a configurable similarity score (gestures/CLIP are future add-ons; emotion cosine is live today).
- Fully offline processing with a local meme index cached in `memes_index.json` for fast lookups.
- Lightweight OpenCV-based window: launch `python app.py`, emote, and close the window when you are done.

## Getting Started

1. **Install dependencies**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate  # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Prepare memes** - drop image files into the `memes/` folder (kept out of git for copyright/privacy).
3. **Index memes** - run:

   ```bash
   python index_memes.py \
     --memes-dir memes \
     --output memes_index.json \
     --infer-gesture-from-dir \
     --face-pick strongest \
     --tags-file docs/manual_tags.json \
     --overwrite
   ```

   Flags to know:
   - `--mtcnn`: slower but more accurate FER detector.
   - `--infer-gesture-from-dir`: uses parent folder names as gesture tags.
   - `--face-pick {largest,strongest}`: how to choose the face when multiple exist (defaults to largest).
   - `--tags-file`: optional manual metadata JSON (see below).
4. **Go live** - run `python app.py` to open the split-screen viewer; closing the window stops the session. The left pane shows your mirrored webcam feed plus the current match overlay (`Match: (none yet)` until a meme clears the similarity threshold), and the right pane swaps between the winning meme panel and the placeholder message.

### Optional manual tags

Create a JSON file (e.g., `docs/manual_tags.json`) to label memes with extra cues:

```json
{
  "thinking_monkey.png": {
    "tags": ["thinking", "classic"],
    "gesture_tags": ["hand_on_chin"],
    "notes": "Use when the user looks contemplative."
  }
}
```

Then run `python index_memes.py --tags-file docs/manual_tags.json --overwrite`.

### Viewer CLI options

| Flag | Description |
| ---- | ----------- |
| `--index-file PATH` | Pick a different `memes_index.json` if you keep multiple variants. |
| `--memes-dir PATH` | Where to load the meme images referenced in the index (default: `memes/`). |
| `--similarity-threshold 0.35` | Minimum cosine score (0–1). Raise it if you want stricter matches. |
| `--analyze-interval 5` | Run FER on every _N_ frames to save CPU. Set to 1 for maximum responsiveness. |
| `--mtcnn` | Enable FER's MTCNN detector (slower but more accurate; matches the CLI flag in `index_memes.py`). |
| `--camera-index` | Pick a different webcam if your machine exposes multiple capture devices. |

Example launch:

```bash
python app.py --similarity-threshold 0.4 --analyze-interval 3
```

### How to test things now

1. **Index smoke test** – run the `index_memes.py` command above after adding new memes. The script logs skipped files and regenerates `memes_index.json`; inspect a few entries to confirm `emotion_vec` and `detected_face` look reasonable.
2. **Viewer dry run** – execute `python app.py --help` to confirm dependencies import cleanly, then `python app.py` with your webcam covered/uncovered to watch the placeholder flip to a meme once a confident expression is detected.
3. **Manual QA loop** – exaggerate an expression (e.g., yell or look skeptical) and verify that the overlay reports the intended meme. If it does not, tweak the corresponding `emotion_vec` entry manually or via `index_memes.py --mtcnn`.

## Learn by Building

- Capture/preprocess frames, run FER/landmark models, and interpret their confidences in real time.
- Experiment with MediaPipe or heuristic gesture tags and see how they influence matches.
- Tune cosine-similarity weights and thresholds to reduce false positives.
- Optionally plug in CLIP embeddings to compare classic CV scoring with modern multimodal features.

## Documentation

- Product requirements and architecture: [`docs/PRD.md`](docs/PRD.md)
- Phased step-by-step roadmap: [`docs/PHASES.md`](docs/PHASES.md)
- Contributor/agent guide: [`AGENTS.md`](AGENTS.md)
