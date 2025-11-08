# Meme Matcher

Meme Matcher is a local-first, split-screen desktop app that pairs your live webcam expressions with the closest meme from a local collection - perfect for quick LinkedIn or Instagram story demos while learning a bit of computer vision.

## Highlights

- Real-time webcam capture rendered next to the matched meme (with a friendly "No meme matched yet" state until a confident result appears).
- Facial emotion detection plus optional gesture cues, blended into a configurable similarity score.
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
4. **Go live** - run `python app.py` to open the split-screen viewer; closing the window stops the session.

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

## Learn by Building

- Capture/preprocess frames, run FER/landmark models, and interpret their confidences in real time.
- Experiment with MediaPipe or heuristic gesture tags and see how they influence matches.
- Tune cosine-similarity weights and thresholds to reduce false positives.
- Optionally plug in CLIP embeddings to compare classic CV scoring with modern multimodal features.

## Documentation

- Product requirements and architecture: [`docs/PRD.md`](docs/PRD.md)
- Phased step-by-step roadmap: [`docs/PHASES.md`](docs/PHASES.md)
- Contributor/agent guide: [`AGENTS.md`](AGENTS.md)
